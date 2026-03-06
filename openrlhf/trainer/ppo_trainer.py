"""
PPO trainer (Ray actor) with optional multi-agent (C3 / MAPPO / MAGRPO) integrations.

Owns:
- Dataset prep + step planning (incl. C3 task datasets + merge_epoch semantics)
- Rollout -> Experience creation (shared / per-role)
- Critic warmup stage with strict rollout-cache contract (MAPPO needs prompt_text)
- Q-critic (C3) + V-critic training orchestration
- Main training loop (fit)

Notes:
- Keep hot paths simple; Ray + torch workloads are host-memory sensitive.
- Warmup cache is contract-checked: slim caches are only valid for fast-path warmup.
"""

# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

from __future__ import annotations

import math
import os
import queue
import threading
import time
from abc import ABC
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple

import ray
import torch
from tqdm import tqdm

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ppo_trainer_plugins import (
    PPOTrainerPluginsMixin,
    _append_jsonl,
    _first,
    _is_rank0,
    _safe_print,
    _sample_role_name,
    _tensor1_float,
    _to_jsonable,
    _unpack_prompt_batch,
)
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.dynamic_filtering import dyn_filter_update
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)

# =============================================================================
# Small helpers
# =============================================================================


def _get_train_epochs(args) -> int:
    try:
        return max(1, int(getattr(args, "train_epochs", 1) or 1))
    except Exception:
        return 1


def _as_stats_dict(x) -> Optional[dict]:
    """Unwrap Ray-returned nested list/tuple of dict-like stats."""
    for _ in range(2):
        x = _first(x)
    return dict(x) if isinstance(x, Mapping) else None


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return bool(default)
    return str(v).strip().lower() not in {"0", "false", "no"}


def _strategy_rank(strategy: DeepspeedStrategy) -> int:
    try:
        return int(getattr(strategy, "get_rank", lambda: 0)())
    except Exception:
        return 0


def _is_missing(x) -> bool:
    """True if a CLI arg is effectively unset."""
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() == ""
    if isinstance(x, (list, tuple)):
        return (not x) or all(isinstance(v, str) and v.strip() == "" for v in x)
    return False


def _safe_select(ds, max_n: int):
    """Select first max_n rows (map-style datasets), best-effort."""
    if max_n is None or int(max_n) <= 0:
        return ds
    if not hasattr(ds, "select"):
        return ds
    try:
        n = len(ds)
    except Exception:
        return ds
    return ds.select(range(min(int(max_n), int(n))))


def _int_scalar(x, default: int = 0) -> int:
    """Parse a scalar-ish value to int (torch scalar, bool/int/str, 0/1 tensors)."""
    if x is None:
        return int(default)
    if isinstance(x, torch.Tensor):
        try:
            if x.numel() <= 0:
                return int(default)
            return int(x.view(-1)[0].item())
        except Exception:
            return int(default)
    try:
        return int(x)
    except Exception:
        return int(default)


# C3 Q-critic "prefix-only" view config.
_Q_CRITIC_VIEW_CFG = {
    "include_full": False,
    "expand_prefix": "all_roles",
    "max_texts_per_sample": 0,
    "prefix_scope": "topo_prefix",
}


def _c3_dump_cfg(args) -> dict:
    """Attach C3-specific run context into dumps (best-effort)."""
    if str(getattr(args, "marl_algorithm", "") or "").lower().strip() != "c3":
        return {}

    out: dict = {
        "c3_credit_variant": str(getattr(args, "c3_credit_variant", "") or ""),
        "q_critic_view_policy": "prefix_only_all_roles_no_full",
    }

    try:
        out["c3_va_alpha"] = float(getattr(args, "c3_va_alpha", 1.0))
    except Exception:
        out["c3_va_alpha"] = 1.0

    fanout = getattr(args, "c3_fanout", None)
    if isinstance(fanout, (list, tuple)):
        out["c3_fanout"] = [int(x) for x in fanout]
        k = 1
        for x in out["c3_fanout"]:
            k *= int(x)
        out["c3_k_from_fanout"] = int(k)
    elif fanout is not None:
        out["c3_fanout"] = fanout

    roles_topo = getattr(args, "c3_roles_topo", None)
    if isinstance(roles_topo, (list, tuple)):
        out["c3_roles_topo"] = [str(x) for x in roles_topo]
    elif roles_topo is not None:
        out["c3_roles_topo"] = roles_topo

    try:
        out["n_samples_per_prompt"] = int(getattr(args, "n_samples_per_prompt", 1) or 1)
    except Exception:
        out["n_samples_per_prompt"] = 1

    if getattr(args, "c3_task", None) is not None:
        out["c3_task"] = str(getattr(args, "c3_task"))

    return out


# =============================================================================
# Warmup cache IO (critic warmup)
# =============================================================================


class _WarmupCacheIO:
    """
    Critic warmup rollout cache.

    Contracts:
      - slim cache: only valid for fast-path warmup (C3 MAS) that doesn't need token tensors
      - full cache: required when warmup uses ExperienceMaker (needs sequences/attention/action)
      - MAPPO warmup requires info['prompt_text'] (cache must preserve it)
    """

    _FMT = "warmup_cache_v2"

    def __init__(
        self,
        *,
        cache_dir: str,
        cache_mode: str,
        cache_slim: bool,
        require_full: bool,
        require_prompt_text: bool,
        rank: int,
    ) -> None:
        self.cache_dir = str(cache_dir or "").strip()
        self.cache_mode = str(cache_mode or "auto").strip().lower()
        self.cache_slim = bool(cache_slim)
        self.require_full = bool(require_full)
        self.require_prompt_text = bool(require_prompt_text)
        self.rank = int(rank)

        self.enabled = bool(self.cache_dir)
        if self.cache_mode not in {"auto", "read", "write", "refresh"}:
            self.cache_mode = "auto"

        if self.require_full and self.cache_slim:
            logger.warning(
                "Warmup cache: forcing FULL cache (need sequences/attention/action). "
                "Set CRITIC_WARMUP_ROLLOUT_CACHE_SLIM=0 if desired."
            )
            self.cache_slim = False

    def _path(self, step_idx: int) -> str:
        return os.path.join(self.cache_dir, f"critic_warmup_rank{self.rank:03d}_step{int(step_idx):06d}.pt")

    @staticmethod
    def _cpu(x):
        if torch.is_tensor(x):
            return x.detach().to("cpu")
        return x

    @staticmethod
    def _ensure_tensor(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x)

    def _build_keep_info(self, info: dict) -> dict:
        keep_keys = (
            "question_id",
            "k_id",
            "role",
            "traj_role_outputs",
            "reward",
            "reward_source",
            "env_info_json",
            "task_name",
            "env_name",
            "prompt_text",  # MAPPO warmup
            "output_text",
            "roles_topo",
            "traj_id",
            "is_leaf",
            "num_roles",
        )
        out: dict = {}
        for k in keep_keys:
            if k in info and info.get(k, None) is not None:
                out[k] = self._cpu(info.get(k))
        return out

    def _to_entries(self, rollout_samples, *, slim: bool):
        """Serialize to list[dict] (portable, avoids pickling Experience)."""
        if rollout_samples is None:
            return None

        out: list = []
        for s in rollout_samples or []:
            if isinstance(s, dict):
                out.append(s)
                continue

            info = getattr(s, "info", None)
            info = info if isinstance(info, dict) else {}
            keep_info = self._build_keep_info(info)

            rewards = getattr(s, "rewards", None)
            if rewards is None:
                rewards = keep_info.get("reward", None)
            scores = getattr(s, "scores", None)
            if scores is None:
                scores = rewards

            d = {
                "_format": self._FMT,
                "slim": bool(slim),
                "prompts": list(getattr(s, "prompts", None) or []),
                "labels": list(getattr(s, "labels", None) or []),
                "rewards": self._cpu(rewards),
                "scores": self._cpu(scores),
                "info": keep_info,
            }

            if not slim:
                d.update(
                    {
                        "sequences": self._cpu(getattr(s, "sequences", None)),
                        "attention_mask": self._cpu(getattr(s, "attention_mask", None)),
                        "action_mask": self._cpu(getattr(s, "action_mask", None)),
                        "rollout_log_probs": self._cpu(getattr(s, "rollout_log_probs", None)),
                    }
                )

            out.append(d)

        return out

    def _is_compatible_entry(self, d: dict) -> bool:
        slim = bool(d.get("slim", True))
        if self.require_full and slim:
            return False

        info = d.get("info", None)
        if self.require_prompt_text:
            if not isinstance(info, dict):
                return False
            if info.get("prompt_text", None) is None:
                return False

        if self.require_full and not slim:
            if (
                d.get("sequences", None) is None
                or d.get("attention_mask", None) is None
                or d.get("action_mask", None) is None
            ):
                return False

        return True

    def _from_entries(self, obj):
        """Deserialize to list[Experience] or None (miss/incompatible)."""
        from openrlhf.trainer.ppo_utils.experience_maker import Experience

        if obj is None:
            return None

        # Legacy: list[Experience] pickled
        if isinstance(obj, list) and obj and isinstance(obj[0], Experience):
            if self.require_full:
                for e in obj:
                    if e.sequences is None or e.attention_mask is None or e.action_mask is None:
                        return None
            if self.require_prompt_text:
                for e in obj:
                    info = getattr(e, "info", None)
                    if (not isinstance(info, dict)) or (info.get("prompt_text", None) is None):
                        return None
            return obj

        # New: list[dict]
        if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
            out: list = []
            for d in obj:
                if not isinstance(d, dict) or not self._is_compatible_entry(d):
                    return None

                slim = bool(d.get("slim", True))
                info = d.get("info", None)

                if not slim:
                    out.append(
                        Experience(
                            sequences=self._ensure_tensor(d.get("sequences", None)),
                            attention_mask=self._ensure_tensor(d.get("attention_mask", None)),
                            action_mask=self._ensure_tensor(d.get("action_mask", None)),
                            rollout_log_probs=self._ensure_tensor(d.get("rollout_log_probs", None)),
                            prompts=list(d.get("prompts") or []),
                            labels=list(d.get("labels") or []),
                            rewards=d.get("rewards", None),
                            scores=d.get("scores", None),
                            info=info,
                        )
                    )
                else:
                    out.append(
                        Experience(
                            prompts=list(d.get("prompts") or []),
                            labels=list(d.get("labels") or []),
                            rewards=d.get("rewards", None),
                            scores=d.get("scores", None),
                            info=info,
                        )
                    )
            return out

        return None

    def load(self, step_idx: int):
        if not self.enabled or self.cache_mode not in {"auto", "read"}:
            return None

        p = self._path(step_idx)
        if not os.path.exists(p):
            return None

        legacy_loaded = False
        try:
            obj = torch.load(p, map_location="cpu")
        except Exception:
            try:
                obj = torch.load(p, map_location="cpu", weights_only=False)
                legacy_loaded = True
                logger.info(f"[WarmupCache] loaded legacy cache with weights_only=False; will migrate: {p}")
            except Exception as e:
                logger.warning(f"[WarmupCache] failed to load {p}: {e}")
                return None

        loaded = self._from_entries(obj)
        if loaded is not None:
            if legacy_loaded:
                try:
                    tmp = p + ".migrated.tmp"
                    torch.save(self._to_entries(loaded, slim=self.cache_slim), tmp)
                    os.replace(tmp, p)
                    logger.info(f"[WarmupCache] migrated legacy cache to safe format: {p}")
                except Exception as e:
                    logger.warning(f"[WarmupCache] migration failed for {p}: {e}")
            return loaded

        if self.cache_mode == "auto":
            try:
                os.replace(p, p + ".incompatible")
                logger.warning(f"[WarmupCache] moved incompatible cache aside: {p}.incompatible")
            except Exception as e:
                logger.warning(f"[WarmupCache] failed to move incompatible cache {p}: {e}")

        logger.warning(
            "[WarmupCache] ignoring incompatible cache entry "
            f"(require_full={self.require_full}, require_prompt_text={self.require_prompt_text}): {p}"
        )
        return None

    def maybe_save(self, step_idx: int, rollout_samples) -> None:
        if not self.enabled or self.cache_mode not in {"auto", "write", "refresh"}:
            return

        p = self._path(step_idx)
        if self.cache_mode == "auto" and os.path.exists(p):
            return

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = p + ".tmp"
            torch.save(self._to_entries(rollout_samples, slim=self.cache_slim), tmp)
            os.replace(tmp, p)
        except Exception as e:
            logger.warning(f"[WarmupCache] failed to save {p}: {e}")


class _WarmupScheduleIO:
    """Deterministic warmup prompt schedule (best-effort, per-rank)."""

    def __init__(self, *, cache_dir: str, rank: int) -> None:
        self.cache_dir = str(cache_dir or "").strip()
        self.rank = int(rank)
        self.enabled = bool(self.cache_dir)
        self._obj = None

    def _path(self) -> str:
        return os.path.join(self.cache_dir, f"critic_warmup_schedule_rank{self.rank:03d}.pt")

    def load_or_create(self) -> None:
        if not self.enabled:
            self._obj = None
            return

        p = self._path()
        if os.path.exists(p):
            try:
                obj = torch.load(p, map_location="cpu")
                if isinstance(obj, dict) and "prompts" in obj and "labels" in obj and "metas" in obj:
                    self._obj = obj
                    return
            except Exception as e:
                logger.warning(f"[WarmupSchedule] failed to load {p}: {e}")

        self._obj = {"prompts": [], "labels": [], "metas": []}

    def __len__(self) -> int:
        if not self.enabled or self._obj is None:
            return 0
        try:
            return int(len(self._obj["prompts"]))
        except Exception:
            return 0

    def get(self, step_idx: int):
        if not self.enabled or self._obj is None:
            return None
        idx = int(step_idx)
        if idx < len(self):
            return (self._obj["prompts"][idx], self._obj["labels"][idx], self._obj["metas"][idx])
        return None

    def append(self, prompts, labels, metas) -> None:
        if not self.enabled or self._obj is None:
            return

        self._obj["prompts"].append(prompts)
        self._obj["labels"].append(labels)
        self._obj["metas"].append(metas)

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            p = self._path()
            tmp = p + ".tmp"
            torch.save(self._obj, tmp)
            os.replace(tmp, p)
        except Exception as e:
            logger.warning(f"[WarmupSchedule] failed to save schedule: {e}")


# =============================================================================
# Base trainer (non-Ray-specific core)
# =============================================================================


class BasePPOTrainer(PPOTrainerPluginsMixin, ABC):
    # Back-compat: some call sites reference BasePPOTrainer._to_jsonable
    _to_jsonable = staticmethod(_to_jsonable)

    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.args = strategy.args

        self.tokenizer = get_tokenizer(
            pretrain,
            None,
            "left",
            strategy,
            use_fast=not self.args.disable_fast_tokenizer,
        )

        self.policy_sharing_mode = getattr(self.args, "policy_sharing_mode", "shared")

        # Shared vs per-role layout
        self.actor_model_group = None
        self.reference_model_group = None
        self.actor_model_groups = None
        self.reference_model_groups = None
        self.role_names = None

        # vLLM layout
        self.vllm_engines = None
        self._vllm_engines_for_generator = vllm_engines
        self.vllm_engines_by_role = None

        self._init_groups(actor_model_group, reference_model_group, vllm_engines)

        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group

        self.dataloader_pin_memory = dataloader_pin_memory
        self.prompt_split = prompt_split
        self.eval_split = eval_split

        self.prompt_max_len = prompt_max_len
        self.generate_kwargs = generate_kwargs

        # Global knobs
        self.max_epochs = self.args.max_epochs
        self.remote_rm_url = self.args.remote_rm_url
        self.init_kl_coef = self.args.init_kl_coef
        self.kl_target = self.args.kl_target
        self.kl_horizon = self.args.kl_horizon

        # Critic scheduling
        self.critic_target_raw = str(getattr(self.args, "critic_target", "auto") or "auto").lower().strip()
        self.critic_target = self.critic_target_raw
        self._critic_target_eff = self.critic_target_raw
        self.critic_warmup_steps = int(getattr(self.args, "critic_warmup_steps", 0) or 0)
        self.critic_train_steps_per_iter = int(getattr(self.args, "critic_train_steps_per_iter", 1) or 1)

        # Runtime state
        self.prompts_dataloader = None
        self.eval_dataloader = None
        self.max_steps = None

        self.samples_generator = None
        self.experience_maker = None
        self.experience_makers_by_role = None
        self.remote_reward_model = None

        # vLLM weights version tracking
        self.policy_version = 0
        self.policy_versions_by_role = {rn: 0 for rn in (self.role_names or [])} if self.actor_model_groups else None

        # Task sampling bookkeeping (P0)
        self._c3_reshuffle_each_epoch = False
        self._c3_task_sampling_mode = None
        self._c3_task_train_builder = None
        self._use_task_train = False
        self._use_task_eval = False

        # Cache for warmup-stage dataloader recreation
        self._train_prompts_dataset = None
        self._train_prompts_eff_bs = None
        self._train_prompts_shuffle = None
        self._train_prompts_drop_last = None

        # Samples generator class
        if self.args.agent_func_path:
            from openrlhf.trainer.ppo_utils.experience_maker_async import SamplesGeneratorAsync as SamplesGenerator
        else:
            from openrlhf.trainer.ppo_utils.experience_maker import SamplesGenerator
        self.generator_cls = SamplesGenerator

        rollout_cls_path = getattr(self.args, "rollout_generator_cls", None)
        if rollout_cls_path:
            if self.args.agent_func_path:
                raise ValueError("rollout_generator_cls is not supported with --agent_func_path/async_train yet")
            import importlib

            mod_name, cls_name = rollout_cls_path.rsplit(".", 1)
            self.generator_cls = getattr(importlib.import_module(mod_name), cls_name)
            logger.info(f"Using custom rollout generator: {rollout_cls_path}")

    def _init_groups(self, actor_model_group, reference_model_group, vllm_engines) -> None:
        if isinstance(actor_model_group, dict):
            self.actor_model_groups = actor_model_group
            self.role_names = list(self.actor_model_groups.keys())
            self.reference_model_groups = reference_model_group if isinstance(reference_model_group, dict) else None
            self.vllm_engines_by_role = vllm_engines if isinstance(vllm_engines, dict) else None
            self.vllm_engines = (
                [e for lst in self.vllm_engines_by_role.values() for e in lst] if self.vllm_engines_by_role else None
            )
            return

        self.actor_model_group = actor_model_group
        self.reference_model_group = reference_model_group
        self.vllm_engines = vllm_engines

    def fit(self):
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Dataset / steps planning (P0: merge_epoch semantics)
    # -------------------------------------------------------------------------

    def _merge_epoch_enabled(self) -> bool:
        return (
            bool(getattr(self, "_c3_task_train_builder", None) is not None)
            and str(getattr(self, "_c3_task_sampling_mode", "") or "").lower().strip() == "merge_epoch"
        )

    def _install_train_dataloader(
        self,
        prompts_dataset: PromptDataset,
        *,
        eff_bs: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        strategy = self.strategy
        args = self.args

        try:
            ds_len = int(len(prompts_dataset))
        except Exception:
            ds_len = 0

        eff_bs = max(1, int(eff_bs or 1))
        if drop_last and ds_len > 0 and eff_bs > ds_len:
            _safe_print(
                strategy,
                (
                    f"[WARN] vllm_generate_batch_size={eff_bs} > dataset_size={ds_len} with drop_last=True; "
                    f"auto-setting vllm_generate_batch_size={ds_len}."
                ),
            )
            eff_bs = ds_len
            args.vllm_generate_batch_size = ds_len

        self.prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset,
            eff_bs,
            shuffle=bool(shuffle),
            drop_last=bool(drop_last),
        )

        # Cache for warmup-stage recreation
        self._train_prompts_dataset = prompts_dataset
        self._train_prompts_eff_bs = int(eff_bs)
        self._train_prompts_shuffle = bool(shuffle)
        self._train_prompts_drop_last = bool(drop_last)

    def _rebuild_task_train_epoch(self, epoch: int) -> None:
        """merge_epoch: rebuild train dataset/dataloader at epoch boundary."""
        if not self._merge_epoch_enabled():
            return

        builder = getattr(self, "_c3_task_train_builder", None)
        if builder is None:
            return

        train_data = builder.build(int(epoch))
        prompts_dataset = PromptDataset(train_data, self.tokenizer, self.strategy, input_template=self.args.input_template)

        # merge_epoch: builder controls epoch-level shuffling; keep dataloader shuffle off
        eff_bs = int(getattr(self.args, "vllm_generate_batch_size", 1) or 1)
        self._install_train_dataloader(prompts_dataset, eff_bs=eff_bs, shuffle=False, drop_last=True)

        try:
            if _is_rank0(self.strategy):
                logger.info(f"[merge_epoch] rebuilt train dataloader for epoch={int(epoch)} (len={len(prompts_dataset)}).")
        except Exception:
            pass

    def prepare_datasets(self) -> None:
        args = self.args
        strategy = self.strategy
        is_eval_only = bool(getattr(args, "eval_only", False))

        # C3 tasks often need meta json.
        if getattr(args, "c3_task", None) and not bool(getattr(args, "return_meta_json", False)):
            setattr(args, "return_meta_json", True)

        use_task = bool(getattr(args, "c3_task", None))
        use_task_train = (not is_eval_only) and use_task and _is_missing(getattr(args, "prompt_data", None))
        use_task_eval = use_task and _is_missing(getattr(args, "eval_dataset", None))

        self._use_task_train = bool(use_task_train)
        self._use_task_eval = bool(use_task_eval)

        task_train_data = None
        task_eval_dict = None

        if use_task_train or use_task_eval:
            from c3.integration.marl_specs import load_task
            from c3.integration.task_datasets import load_task_datasets

            task_spec = load_task(args.c3_task)

            env = getattr(task_spec, "environment", None)
            if isinstance(env, dict):
                self._c3_reshuffle_each_epoch = bool(env.get("reshuffle_each_epoch", False))
                self._c3_task_sampling_mode = str(env.get("sampling_mode", "concat") or "concat").strip().lower()
            else:
                self._c3_reshuffle_each_epoch = bool(getattr(env, "reshuffle_each_epoch", False))
                self._c3_task_sampling_mode = str(getattr(env, "sampling_mode", "concat") or "concat").strip().lower()

            td = load_task_datasets(
                task_spec,
                cache_dir=getattr(args, "dataset_cache_dir", None),
                default_train_split=self.prompt_split,
                default_eval_split=self.eval_split,
                max_train_samples=getattr(args, "max_samples", None),
                max_eval_samples=getattr(args, "max_samples", None),
            )
            task_train_data = td.train
            task_eval_dict = td.evals
            self._c3_task_train_builder = getattr(td, "train_builder", None)

        # ---- train dataset ----
        prompts_dataset = None
        if not is_eval_only:
            if use_task_train:
                train_data = task_train_data
            else:
                # Do not per-source truncate; apply global cap after mixing.
                train_data = blending_datasets(
                    args.prompt_data,
                    args.prompt_data_probs,
                    strategy,
                    args.seed,
                    max_count=int(1e8),
                    dataset_split=self.prompt_split,
                )
                train_data = _safe_select(train_data, getattr(args, "max_samples", -1))

            prompts_dataset = PromptDataset(train_data, self.tokenizer, strategy, input_template=args.input_template)

            if use_task_train and self._merge_epoch_enabled():
                train_shuffle = False
            else:
                train_shuffle = bool(self._c3_reshuffle_each_epoch) if use_task_train else True

            eff_bs = int(getattr(args, "vllm_generate_batch_size", 1) or 1)
            self._install_train_dataloader(prompts_dataset, eff_bs=eff_bs, shuffle=train_shuffle, drop_last=True)
        else:
            self.prompts_dataloader = None

        # ---- eval dataset ----
        eval_dataloader = None
        if use_task_eval:
            eval_data = None
            if task_eval_dict and isinstance(task_eval_dict, dict):
                eval_list = list(task_eval_dict.values())
                if len(eval_list) == 1:
                    eval_data = eval_list[0]
                elif len(eval_list) > 1:
                    try:
                        import datasets  # type: ignore

                        eval_data = datasets.concatenate_datasets(eval_list)
                    except Exception:
                        eval_data = eval_list[0]

            if eval_data is not None:
                eval_data = _safe_select(eval_data, getattr(args, "max_samples", -1))
                eval_dataset = PromptDataset(eval_data, self.tokenizer, strategy, input_template=args.input_template)
                eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, shuffle=True, drop_last=False)

        elif getattr(args, "eval_dataset", None):
            eval_data = blending_datasets(args.eval_dataset, None, strategy, dataset_split=self.eval_split)
            eval_data = _safe_select(eval_data, getattr(args, "max_samples", -1))
            eval_dataset = PromptDataset(eval_data, self.tokenizer, strategy, input_template=args.input_template)
            eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, shuffle=True, drop_last=False)

        self.eval_dataloader = eval_dataloader

        if is_eval_only:
            self.max_steps = 1
            return

        # ---- max_steps ----
        train_epochs = _get_train_epochs(args)
        try:
            num_batches = int(self.prompts_dataloader.__len__())
        except Exception:
            try:
                ds_len = int(len(prompts_dataset)) if prompts_dataset is not None else 0
                bs = int(getattr(args, "vllm_generate_batch_size", 0) or getattr(args, "rollout_batch_size", 1) or 1)
                bs = max(1, bs)
                num_batches = ds_len // bs
            except Exception:
                num_batches = 0

        self.max_steps = max(1, int(num_batches)) * train_epochs

    def get_max_steps(self):
        return self.max_steps


# =============================================================================
# PPOTrainer (Ray actor)
# =============================================================================


@ray.remote
class PPOTrainer(BasePPOTrainer):
    """Trainer for PPO / REINFORCE++ / GRPO / RLOO and variants (Ray actor)."""

    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            **generate_kwargs,
        )

        self.kl_ctl = (
            AdaptiveKLController(self.init_kl_coef, self.kl_target, self.kl_horizon)
            if self.kl_target
            else FixedKLController(self.init_kl_coef)
        )

        # Optional remote RM
        if self.args.remote_rm_url and not self.args.remote_rm_url[0] == "agent":
            from openrlhf.utils.remote_rm_utils import RemoteRewardModel

            self.remote_reward_model = RemoteRewardModel.remote(self.args, self.remote_rm_url)
        else:
            self.remote_reward_model = None

        self.q_critic_model_group = generate_kwargs.pop("q_critic_model_group", None)

        # C3 MAS marker
        self._is_c3_mas = bool(
            getattr(self.args, "rollout_generator_cls", None) and getattr(self.args, "c3_task", None)
        )
        self._c3_roles = None
        self._c3_layers = None
        self._c3_parents = None

        marl_alg0 = self._marl_alg()
        if self._is_c3_mas and marl_alg0 == "c3":
            from c3.integration.marl_specs import load_task
            from c3.credit.c3.baselines import build_dependency_from_roles

            task_spec = load_task(self.args.c3_task)
            self._c3_roles = [r.name for r in task_spec.roles]
            parents, layers, _, _, _, _ = build_dependency_from_roles(task_spec.roles)
            self._c3_parents = parents
            self._c3_layers = layers

        self.samples_generator = self.generator_cls(
            self._vllm_engines_for_generator,
            self.strategy,
            self.tokenizer,
            self.prompt_max_len,
        )

        self._init_experience_makers()
        self.prepare_datasets()
        self._init_wandb()

        if self.args.eval_steps == -1:
            self.args.eval_steps = float("inf")
        if self.args.save_steps == -1:
            self.args.save_steps = float("inf")

    # -------------------------------------------------------------------------
    # Common accessors
    # -------------------------------------------------------------------------

    def _marl_alg(self) -> str:
        return str(getattr(self.args, "marl_algorithm", "auto") or "auto").lower().strip()

    # -------------------------------------------------------------------------
    # Experience makers
    # -------------------------------------------------------------------------

    def _init_experience_makers(self) -> None:
        if self.actor_model_groups is not None:
            self.experience_makers_by_role = {}
            for rn in self.role_names:
                ref_g = self.reference_model_groups.get(rn) if self.reference_model_groups is not None else None
                self.experience_makers_by_role[rn] = RemoteExperienceMaker(
                    self.actor_model_groups[rn],
                    self.critic_model_group,
                    self.reward_model_group,
                    ref_g,
                    self.kl_ctl,
                    self.strategy,
                    self.tokenizer,
                    remote_reward_model=self.remote_reward_model,
                    q_critic_model_group=self.q_critic_model_group,
                )
        else:
            self.experience_maker = RemoteExperienceMaker(
                self.actor_model_group,
                self.critic_model_group,
                self.reward_model_group,
                self.reference_model_group,
                self.kl_ctl,
                self.strategy,
                self.tokenizer,
                remote_reward_model=self.remote_reward_model,
                q_critic_model_group=self.q_critic_model_group,
            )

    # -------------------------------------------------------------------------
    # Rollout -> Experience utilities
    # -------------------------------------------------------------------------

    def _route_samples_by_role(self, rollout_samples) -> Dict[str, list]:
        samples_by_role = {rn: [] for rn in self.role_names}
        for s in rollout_samples:
            rn = _sample_role_name(s)
            if rn not in samples_by_role:
                raise RuntimeError(f"per_role mode got unknown role={rn!r}, expected={self.role_names}")
            samples_by_role[rn].append(s)

        for rn in self.role_names:
            if len(samples_by_role[rn]) == 0:
                raise RuntimeError(f"per_role mode: got 0 rollout samples for role={rn!r}. Check rollout_generator.")
        return samples_by_role

    def _make_experiences(self, rollout_samples):
        if self.actor_model_groups is None:
            return self.experience_maker.make_experience_batch(rollout_samples), None

        samples_by_role = self._route_samples_by_role(rollout_samples)
        experiences_by_role = {
            rn: self.experience_makers_by_role[rn].make_experience_batch(samples_by_role[rn], compute_advantages=False)
            for rn in self.role_names
        }
        experiences_all = sum(experiences_by_role.values(), [])

        any_maker = next(iter(self.experience_makers_by_role.values()))
        experiences_all = any_maker.compute_advantages_and_returns(experiences_all)
        return experiences_all, experiences_by_role

    def _maybe_balance_experiences(self, experiences_all, experiences_by_role):
        if not self.args.use_dynamic_batch:
            return experiences_all, experiences_by_role

        if self.actor_model_groups is not None:
            for rn in self.role_names:
                experiences_by_role[rn] = balance_experiences(experiences_by_role[rn], self.args)
            return sum(experiences_by_role.values(), []), experiences_by_role

        return balance_experiences(experiences_all, self.args), experiences_by_role

    # -------------------------------------------------------------------------
    # Debug dumps
    # -------------------------------------------------------------------------

    def _extract_experience_reward_scalar(self, exp) -> Optional[float]:
        for k in ("reward", "rewards", "scores", "answer_reward"):
            if hasattr(exp, k):
                v = _tensor1_float(getattr(exp, k), default=None)
                if v is not None:
                    return float(v)

        info = getattr(exp, "info", None)
        if isinstance(info, dict):
            for k in ("answer_reward", "reward"):
                v = _tensor1_float(info.get(k, None), default=None)
                if v is not None:
                    return float(v)
        return None

    def _maybe_dump_rollouts_stats_jsonl(
        self,
        *,
        dump_path: Optional[str],
        dump_every: int,
        rollout_iter: int,
        steps: int,
        epoch_idx: int,
        iter_in_epoch: int,
        train_epochs: int,
        rand_prompts,
        experiences_all,
        q_critic_status_log: Optional[dict],
    ) -> None:
        if not dump_path or dump_every <= 0 or not _is_rank0(self.strategy):
            return
        if (rollout_iter % int(dump_every)) != 0:
            return

        def _mean_tensor(x):
            try:
                if isinstance(x, torch.Tensor):
                    return float(x.detach().float().mean().cpu().item())
            except Exception:
                pass
            return None

        try:
            prompt0 = str(rand_prompts[0]) if rand_prompts is not None and len(rand_prompts) > 0 else ""
            rewards, clip_flags, resp_lens, tot_lens = [], [], [], []
            for e in experiences_all:
                info = getattr(e, "info", None) or {}
                if not isinstance(info, dict):
                    continue

                r = info.get("reward", None)
                if r is not None:
                    mr = _mean_tensor(r)
                    if mr is not None:
                        rewards.append(mr)

                cf = info.get("response_clip_ratio", None)
                if cf is not None:
                    mc = _mean_tensor(cf)
                    if mc is not None:
                        clip_flags.append(mc)

                rl = info.get("response_length", None)
                if rl is not None:
                    mrl = _mean_tensor(rl)
                    if mrl is not None:
                        resp_lens.append(mrl)

                tl = info.get("total_length", None)
                if tl is not None:
                    mtl = _mean_tensor(tl)
                    if mtl is not None:
                        tot_lens.append(mtl)

            payload = {
                "kind": "train_rollouts_stats",
                "ts": float(time.time()),
                "rollout_iter": int(rollout_iter),
                "global_step": int(steps),
                "epoch": int(epoch_idx),
                "iter_in_epoch": int(iter_in_epoch),
                "train_epochs": int(train_epochs),
                "marl_algorithm": str(getattr(self.args, "marl_algorithm", "")),
                "prompt": prompt0,
                "reward/mean": (sum(rewards) / len(rewards)) if rewards else None,
                "response_clip_ratio/mean": (sum(clip_flags) / len(clip_flags)) if clip_flags else None,
                "response_length/mean": (sum(resp_lens) / len(resp_lens)) if resp_lens else None,
                "total_length/mean": (sum(tot_lens) / len(tot_lens)) if tot_lens else None,
            }
            payload.update(_c3_dump_cfg(self.args))
            if q_critic_status_log is not None:
                payload["q_critic_status"] = _to_jsonable(q_critic_status_log)
            payload = self._with_run_context(payload)

            _append_jsonl(str(dump_path), payload)
        except Exception as e:
            _safe_print(self.strategy, f"[WARN] dump_rollouts_stats_jsonl failed: {e}")

    def _maybe_dump_rollouts_jsonl(
        self,
        *,
        dump_path: Optional[str],
        dump_every: int,
        rollout_iter: int,
        steps: int,
        epoch_idx: int,
        iter_in_epoch: int,
        train_epochs: int,
        rand_prompts,
        experiences_all,
        q_critic_status_log: Optional[dict],
    ) -> None:
        if not dump_path or dump_every <= 0 or not _is_rank0(self.strategy):
            return
        if (rollout_iter % int(dump_every)) != 0:
            return

        try:
            prompt0 = str(rand_prompts[0]) if rand_prompts is not None and len(rand_prompts) > 0 else ""
            rewards = []
            for e in experiences_all:
                r = self._extract_experience_reward_scalar(e)
                if r is not None:
                    rewards.append(float(r))
            reward_mean = float(sum(rewards) / len(rewards)) if rewards else None

            payload = {
                "kind": "train_rollouts",
                "ts": float(time.time()),
                "rollout_iter": int(rollout_iter),
                "steps": int(steps),
                "global_step": int(steps),
                "epoch": int(epoch_idx),
                "iter_in_epoch": int(iter_in_epoch),
                "train_epochs": int(train_epochs),
                "marl_algorithm": str(getattr(self.args, "marl_algorithm", "")),
                "num_experiences": int(len(experiences_all)),
                "prompt": prompt0,
                "reward": reward_mean,
                "policy_sharing_mode": str(getattr(self.args, "policy_sharing_mode", "shared") or "shared"),
            }
            payload.update(_c3_dump_cfg(self.args))
            if q_critic_status_log is not None:
                payload["q_critic_status"] = _to_jsonable(q_critic_status_log)

            payload = self._with_run_context(payload)
            payload["experiences"] = [self._summarize_experience(e) for e in experiences_all]
            _append_jsonl(str(dump_path), payload)
        except Exception as e:
            _safe_print(self.strategy, f"[WARN] dump_rollouts_jsonl failed: {e}")

    def _maybe_dump_c3_batch_data_jsonl(
        self,
        *,
        dump_batch_path: Optional[str],
        dump_every: int,
        rollout_iter: int,
        steps: int,
        epoch_idx: int,
        iter_in_epoch: int,
        train_epochs: int,
        c3_batch_data,
    ) -> None:
        if not dump_batch_path or dump_every <= 0 or not _is_rank0(self.strategy):
            return
        if (rollout_iter % int(dump_every)) != 0:
            return
        if not isinstance(c3_batch_data, list):
            return

        try:
            _append_jsonl(
                str(dump_batch_path),
                self._with_run_context(
                    {
                        "ts": float(time.time()),
                        "rollout_iter": int(rollout_iter),
                        "steps": int(steps),
                        "global_step": int(steps),
                        "epoch": int(epoch_idx),
                        "iter_in_epoch": int(iter_in_epoch),
                        "train_epochs": int(train_epochs),
                        "marl_algorithm": str(getattr(self.args, "marl_algorithm", "")),
                        **_c3_dump_cfg(self.args),
                        "batch_data": _to_jsonable(c3_batch_data),
                    }
                ),
            )
        except Exception as e:
            _safe_print(self.strategy, f"[WARN] dump_c3_batch_data failed: {e}")

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------

    def _decode_first_experience(self, experiences_all) -> str:
        if not experiences_all:
            return ""
        try:
            from openrlhf.utils.utils import remove_pad_token

            exp0 = experiences_all[0]
            if not isinstance(getattr(exp0, "sequences", None), torch.Tensor):
                return ""

            seq = exp0.sequences
            attn = getattr(exp0, "attention_mask", None)
            if seq.dim() == 1:
                seq = seq.unsqueeze(0)
            if isinstance(attn, torch.Tensor) and attn.dim() == 1:
                attn = attn.unsqueeze(0)

            if isinstance(attn, torch.Tensor):
                ids0 = remove_pad_token(seq[:1], attn[:1])
                return self.tokenizer.batch_decode(ids0, skip_special_tokens=False)[0]
            return self.tokenizer.decode(seq[0].detach().cpu().tolist(), skip_special_tokens=False)
        except Exception:
            return ""

    # -------------------------------------------------------------------------
    # Dynamic filtering
    # -------------------------------------------------------------------------

    def _dyn_filter_update(self, rollout_samples, state: Tuple[list, int]):
        if not self.args.dynamic_filtering:
            return rollout_samples, None, state

        return dyn_filter_update(
            rollout_samples,
            k=int(self.args.n_samples_per_prompt),
            rollout_batch_size=int(self.args.rollout_batch_size),
            reward_range=tuple(self.args.dynamic_filtering_reward_range),
            state=state,
        )

    # -------------------------------------------------------------------------
    # Critic-only warmup (cache contract enforced)
    # -------------------------------------------------------------------------

    def _run_critic_warmup_stage(self, warmup_total: int, warmup_done: int = 0) -> int:
        warmup_total = int(warmup_total or 0)
        warmup_done = int(warmup_done or 0)
        if warmup_total <= 0 or warmup_done >= warmup_total:
            return max(warmup_done, warmup_total)

        if self.prompts_dataloader is None:
            raise RuntimeError("critic_warmup_steps > 0 but prompts_dataloader is None")

        ds = getattr(self, "_train_prompts_dataset", None) or getattr(self.prompts_dataloader, "dataset", None)
        if ds is None:
            raise RuntimeError("cannot build warmup dataloader: train PromptDataset is missing")

        eff_bs = int(getattr(self, "_train_prompts_eff_bs", 1) or 1)
        train_shuffle = bool(getattr(self, "_train_prompts_shuffle", True))
        train_drop_last = bool(getattr(self, "_train_prompts_drop_last", True))
        warmup_dl = self.strategy.setup_dataloader(ds, eff_bs, shuffle=train_shuffle, drop_last=train_drop_last)

        # Prefetch (optional)
        prefetch_depth = int(getattr(self.args, "critic_warmup_prefetch_depth", 0) or 0)
        prefetch_timeout = float(getattr(self.args, "critic_warmup_prefetch_timeout", 60.0) or 60.0)
        use_prefetch = prefetch_depth > 0

        # Cache config
        cache_dir = str(
            getattr(self.args, "critic_warmup_rollout_cache_dir", "")
            or os.environ.get("CRITIC_WARMUP_ROLLOUT_CACHE_DIR", "")
        ).strip()
        cache_mode = str(
            getattr(self.args, "critic_warmup_rollout_cache_mode", "")
            or os.environ.get("CRITIC_WARMUP_ROLLOUT_CACHE_MODE", "auto")
        ).strip().lower()

        marl_alg = self._marl_alg()
        fast_path_expected = bool(
            int(getattr(self.args, "critic_warmup_fast_path", 1) or 1) and self._is_c3_mas and marl_alg == "c3"
        )
        need_full_cache = not fast_path_expected
        require_prompt_text = marl_alg == "mappo"

        slim_default = "1" if fast_path_expected else "0"
        cache_slim = str(os.environ.get("CRITIC_WARMUP_ROLLOUT_CACHE_SLIM", slim_default)).strip().lower() not in {
            "0",
            "false",
            "no",
        }

        rank = _strategy_rank(self.strategy)
        cache_io = _WarmupCacheIO(
            cache_dir=cache_dir,
            cache_mode=cache_mode,
            cache_slim=cache_slim,
            require_full=need_full_cache,
            require_prompt_text=require_prompt_text,
            rank=rank,
        )

        # Schedule: best-effort determinism, only when cache enabled + no dyn filtering + no prefetch.
        schedule_enabled = bool(
            cache_io.enabled
            and (not bool(getattr(self.args, "dynamic_filtering", False)))
            and (not use_prefetch)
            and _env_flag("CRITIC_WARMUP_ROLLOUT_SCHEDULE", True)
        )
        schedule_io = _WarmupScheduleIO(cache_dir=cache_dir, rank=rank)
        if schedule_enabled:
            schedule_io.load_or_create()

        # Rollout generator wrapper (shared by prefetch + main loop)
        def _generate_rollout(rand_prompts, labels, meta_jsons, *, timing: Optional[dict], key: str):
            self._strict_check_vllm_before_rollout()
            gen_kwargs = dict(self.generate_kwargs)
            if meta_jsons is not None:
                gen_kwargs["all_metas"] = list(meta_jsons)

            t0 = time.perf_counter() if timing is not None else None
            rs = self.samples_generator.generate_samples(
                rand_prompts,
                labels,
                remote_reward_model=self.remote_reward_model,
                phase="train",
                **gen_kwargs,
            )
            if timing is not None and t0 is not None:
                timing[key] = (time.perf_counter() - t0) * 1000.0
            return rs

        # Prefetch producer
        stop_evt = threading.Event()
        producer_err = {"exc": None}
        q = queue.Queue(maxsize=prefetch_depth) if use_prefetch else None
        producer_th = None

        def _producer_put(item) -> bool:
            if q is None:
                return False
            while not stop_evt.is_set():
                try:
                    q.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    continue
            return False

        def _warmup_producer_loop():
            try:
                dyn_state_p: Tuple[list, int] = ([], 0)
                epoch_p = 0
                self._maybe_set_dataloader_epoch(warmup_dl, epoch_p, resume_has_state=False, start_epoch=0)
                itp = iter(warmup_dl)

                while not stop_evt.is_set():
                    try:
                        batch = next(itp)
                    except StopIteration:
                        epoch_p += 1
                        self._maybe_set_dataloader_epoch(warmup_dl, epoch_p, resume_has_state=False, start_epoch=0)
                        itp = iter(warmup_dl)
                        continue

                    _, rand_prompts, labels, meta_jsons = _unpack_prompt_batch(batch, where="critic warmup producer")
                    rollout_samples = _generate_rollout(
                        rand_prompts,
                        labels,
                        meta_jsons,
                        timing=None,
                        key="",
                    )

                    selected, pass_rate, dyn_state_p = self._dyn_filter_update(rollout_samples, dyn_state_p)
                    if self.args.dynamic_filtering:
                        if selected is None:
                            continue
                        rollout_samples = selected

                    if not _producer_put({"rollout_samples": rollout_samples, "pass_rate": pass_rate}):
                        break

            except Exception as e:
                producer_err["exc"] = e
                stop_evt.set()
            finally:
                if q is not None:
                    try:
                        q.put(None, timeout=1)
                    except Exception:
                        pass

        if use_prefetch:
            logger.info(f"Warmup rollout prefetch enabled: depth={prefetch_depth}, timeout={prefetch_timeout}s")
            producer_th = threading.Thread(target=_warmup_producer_loop, name="warmup_rollout_prefetch", daemon=True)
            producer_th.start()

        # No-prefetch prompt iterator
        epoch = 0
        dyn_state: Tuple[list, int] = ([], 0)
        if not use_prefetch:
            self._maybe_set_dataloader_epoch(warmup_dl, epoch, resume_has_state=False, start_epoch=0)
            it = iter(warmup_dl)

            def _next_prompts(where: str):
                nonlocal epoch, it
                while True:
                    try:
                        batch = next(it)
                    except StopIteration:
                        epoch += 1
                        self._maybe_set_dataloader_epoch(warmup_dl, epoch, resume_has_state=False, start_epoch=0)
                        it = iter(warmup_dl)
                        continue
                    _, rand_prompts, labels, meta_jsons = _unpack_prompt_batch(batch, where=where)
                    return rand_prompts, labels, meta_jsons

        # Warmup loop
        self._in_critic_warmup_stage = True
        pbar = tqdm(
            total=int(warmup_total - warmup_done),
            desc=f"CriticWarmup [{warmup_done}/{warmup_total}]",
            disable=not _is_rank0(self.strategy),
        )

        try:
            fast_path = bool(int(getattr(self.args, "critic_warmup_fast_path", 1) or 1) and self._is_c3_mas and marl_alg == "c3")

            while warmup_done < warmup_total:
                timing = {} if self._timing_enabled() else None
                t_iter0 = time.perf_counter() if timing is not None else None

                pass_rate = None
                rollout_samples = None

                # 1) Get rollout samples
                if use_prefetch:
                    if producer_err.get("exc") is not None:
                        raise RuntimeError("Warmup rollout prefetch producer failed") from producer_err["exc"]

                    try:
                        item = q.get(timeout=prefetch_timeout)  # type: ignore[union-attr]
                    except queue.Empty as e:
                        if producer_err.get("exc") is not None:
                            raise RuntimeError("Warmup rollout prefetch producer failed") from producer_err["exc"]
                        raise RuntimeError(
                            f"Warmup rollout prefetch timeout after {prefetch_timeout}s. "
                            "Increase --critic_warmup_prefetch_timeout or disable prefetch (depth=0)."
                        ) from e

                    if item is None:
                        if producer_err.get("exc") is not None:
                            raise RuntimeError("Warmup rollout prefetch producer failed") from producer_err["exc"]
                        raise RuntimeError("Warmup rollout prefetcher stopped unexpectedly")

                    rollout_samples = item.get("rollout_samples")
                    pass_rate = item.get("pass_rate")

                    if timing is not None and t_iter0 is not None:
                        timing["time/warmup_rollout_wait_ms"] = (time.perf_counter() - t_iter0) * 1000.0

                else:
                    cache_hit = False
                    if cache_io.enabled and cache_io.cache_mode in {"auto", "read"}:
                        rollout_samples = cache_io.load(warmup_done)
                        cache_hit = rollout_samples is not None
                        if timing is not None:
                            timing["warmup_rollout_cache_hit"] = 1.0 if cache_hit else 0.0

                    rand_prompts = labels = meta_jsons = None

                    if schedule_enabled:
                        sched = schedule_io.get(warmup_done)
                        if sched is not None:
                            rand_prompts, labels, meta_jsons = sched
                            if timing is not None:
                                timing["warmup_schedule_hit"] = 1.0
                        else:
                            rand_prompts, labels, meta_jsons = _next_prompts(where="critic warmup schedule")
                            schedule_io.append(
                                list(rand_prompts) if rand_prompts is not None else rand_prompts,
                                list(labels) if labels is not None else labels,
                                (list(meta_jsons) if meta_jsons is not None else None),
                            )
                            if timing is not None:
                                timing["warmup_schedule_hit"] = 0.0

                    if rollout_samples is None:
                        if rand_prompts is None or labels is None:
                            rand_prompts, labels, meta_jsons = _next_prompts(where="critic warmup")

                        rollout_samples = _generate_rollout(
                            rand_prompts,
                            labels,
                            meta_jsons,
                            timing=timing,
                            key="time/warmup_rollout_generate_ms",
                        )

                    selected, pass_rate, dyn_state = self._dyn_filter_update(rollout_samples, dyn_state)
                    if self.args.dynamic_filtering:
                        if selected is None:
                            buf_groups, _n_total = dyn_state
                            logger.info(
                                "[Warmup] filtered_prompts %d < rollout_batch_size %s, continue sampling",
                                len(buf_groups),
                                self.args.rollout_batch_size,
                            )
                            continue
                        rollout_samples = selected
                        logger.info(f"[Warmup] Dynamic filtering pass rate: {pass_rate:.2f}%")

                    if cache_io.enabled and rollout_samples is not None:
                        if not (cache_hit and cache_io.cache_mode == "auto"):
                            cache_io.maybe_save(warmup_done, rollout_samples)

                # 2) Make experiences (or fast_path)
                if fast_path:
                    experiences_all = rollout_samples
                    experiences_by_role = None
                    if timing is not None:
                        timing["time/warmup_make_experiences_ms"] = 0.0
                else:
                    t_exp0 = time.perf_counter() if timing is not None else None
                    experiences_all, experiences_by_role = self._make_experiences(rollout_samples)
                    if timing is not None and t_exp0 is not None:
                        timing["time/warmup_make_experiences_ms"] = (time.perf_counter() - t_exp0) * 1000.0

                    experiences_all, experiences_by_role = self._maybe_balance_experiences(
                        experiences_all, experiences_by_role
                    )

                # 3) Train critics (Q then V)
                t_q0 = time.perf_counter() if timing is not None else None
                q_critic_status, _ = self._train_q_critic_if_needed(experiences_all)
                if timing is not None and t_q0 is not None:
                    timing["time/warmup_q_critic_ms"] = (time.perf_counter() - t_q0) * 1000.0

                t_v0 = time.perf_counter() if timing is not None else None
                v_critic_stats_early, v_critic_status_ref = self._train_v_critic_if_needed(experiences_all)
                if timing is not None and t_v0 is not None:
                    timing["time/warmup_v_critic_call_ms"] = (time.perf_counter() - t_v0) * 1000.0

                # 4) Log
                status = {
                    "critic_warmup": 1,
                    "critic_warmup_step": int(warmup_done + 1),
                    "critic_warmup_total": int(warmup_total),
                    "critic_warmup_fast_path": 1 if fast_path else 0,
                }
                if self.args.dynamic_filtering and pass_rate is not None:
                    status["dynamic_filtering_pass_rate"] = pass_rate

                status.update(self._rewardprovider_rollout_metrics(experiences_all))
                if isinstance(v_critic_stats_early, dict):
                    status.update(v_critic_stats_early)

                if v_critic_status_ref is not None:
                    t_vw0 = time.perf_counter() if timing is not None else None
                    v_stats = ray.get(v_critic_status_ref)[0]
                    if timing is not None and t_vw0 is not None:
                        timing["time/warmup_v_critic_wait_ms"] = (time.perf_counter() - t_vw0) * 1000.0

                    if isinstance(v_stats, dict):
                        status.update(v_stats)
                    if self.strategy.args.deepspeed_enable_sleep:
                        ray.get(self.critic_model_group.async_run_method(method_name="offload_states"))

                if isinstance(q_critic_status, dict):
                    status.update(q_critic_status)

                if timing is not None and t_iter0 is not None:
                    timing["time/warmup_iter_ms"] = (time.perf_counter() - t_iter0) * 1000.0
                    status.update(timing)

                logger.info(f"Critic warmup {warmup_done + 1}/{warmup_total}: {status}")

                warmup_done += 1
                pbar.update(1)

        finally:
            self._in_critic_warmup_stage = False

            if use_prefetch:
                stop_evt.set()
                try:
                    if producer_th is not None:
                        producer_th.join(timeout=10)
                except Exception:
                    pass
                try:
                    while True:
                        _ = q.get_nowait()  # type: ignore[union-attr]
                except Exception:
                    pass

            try:
                pbar.close()
            except Exception:
                pass

        return int(warmup_done)

    # -------------------------------------------------------------------------
    # Q / V critic training
    # -------------------------------------------------------------------------

    def _c3_failfast_missing_traj_maps(self, experiences_all, *args, **kwargs) -> None:
        """
        C3 Q-critic / credit materialization requires info['traj_role_outputs'].
        """
        where = None
        if args:
            where = args[0]
        where = kwargs.get("where", where)
        if where is None:
            where = "unknown"

        if not isinstance(experiences_all, list):
            return
        for exp in experiences_all:
            info = getattr(exp, "info", None)
            if not isinstance(info, dict):
                continue
            marl_enabled = _int_scalar(info.get("marl_enabled", 0), default=0)
            if marl_enabled == 1 and "traj_role_outputs" not in info:
                raise KeyError(
                    "[C3][FAIL-FAST] Experience.info missing 'traj_role_outputs' mapping before Q-critic "
                    "materialization. This usually means it was pruned earlier for Ray memory safety. "
                    "Fix ExperienceMaker._prune_rollout_text_fields_inplace to preserve 'traj_role_outputs' for C3."
                    f" (where={where})"
                )

    def _c3_regroup_experiences_by_prompt(self, experiences_all):
        """C3-only: validate leaf Full-K then regroup when each Experience is qid-pure."""
        if not experiences_all or not isinstance(experiences_all, list):
            return experiences_all

        info0 = getattr(experiences_all[0], "info", None)
        if not isinstance(info0, dict) or "question_id" not in info0:
            return experiences_all

        try:
            k_rollouts = max(1, int(getattr(self.args, "n_samples_per_prompt", 1) or 1))
        except Exception:
            k_rollouts = 1

        def _batch_size(exp) -> int:
            seq = getattr(exp, "sequences", None)
            if isinstance(seq, torch.Tensor) and seq.dim() >= 1:
                return int(seq.shape[0])
            if isinstance(seq, list):
                return len(seq)

            info = getattr(exp, "info", None) or {}
            if isinstance(info, dict):
                for key in ("question_id", "k_id", "role"):
                    v = info.get(key, None)
                    if isinstance(v, torch.Tensor) and v.dim() >= 1:
                        return int(v.shape[0])
                    if isinstance(v, list):
                        return len(v)
                for v in info.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 1:
                        return int(v.shape[0])
                    if isinstance(v, list):
                        return len(v)
            return 1

        def _value_at_row(v, row: int, B: int):
            if v is None:
                return None
            if isinstance(v, torch.Tensor):
                try:
                    if v.dim() >= 1 and int(v.shape[0]) == int(B):
                        return v[row]
                except Exception:
                    pass
                vv = v.view(-1)
                if vv.numel() == 0:
                    return None
                if vv.numel() == 1:
                    return vv[0]
                return vv[row] if 0 <= row < int(vv.numel()) else vv[0]
            if isinstance(v, list):
                if not v:
                    return None
                return v[row] if 0 <= row < len(v) else v[0]
            return v

        def _int_at_row(info: dict, key: str, row: int, B: int) -> int:
            if key not in info:
                raise RuntimeError(f"[C3][FAIL-FAST] Missing key {key!r} in Experience.info.")
            vv = _value_at_row(info.get(key, None), row, B)
            if vv is None:
                raise RuntimeError(f"[C3][FAIL-FAST] info[{key!r}] is None at row={row}")
            if isinstance(vv, torch.Tensor) and vv.numel() == 1:
                return int(vv.view(-1)[0].item())
            return int(vv)

        def _leaf_at_row(info: dict, row: int, B: int, kid: int) -> int:
            # Nested-C3: k_id is leaf-only => kid < 0 means non-leaf
            if int(kid) < 0:
                return 0
            vv = _value_at_row(info.get("is_leaf", None), row, B) if isinstance(info, dict) else None
            if vv is None:
                return 1
            if isinstance(vv, torch.Tensor) and vv.numel() == 1:
                return int(vv.view(-1)[0].item())
            return int(vv)

        leaf_k_by_qid: Dict[int, set] = {}
        rows_by_qid: Dict[int, int] = {}
        leaf_rows_by_qid: Dict[int, int] = {}
        exp_qid_pure: List[Optional[int]] = []

        for ei, exp in enumerate(experiences_all):
            info = getattr(exp, "info", None)
            if not isinstance(info, dict):
                raise RuntimeError(f"[C3][FAIL-FAST] Experience.info missing/not dict at exp_idx={ei}")

            B = _batch_size(exp)
            qids_in_exp: set[int] = set()

            for row in range(B):
                qid = _int_at_row(info, "question_id", row, B)
                kid = _int_at_row(info, "k_id", row, B)
                is_leaf = _leaf_at_row(info, row, B, kid)

                qid = int(qid)
                qids_in_exp.add(qid)

                rows_by_qid[qid] = rows_by_qid.get(qid, 0) + 1
                if int(is_leaf) == 1:
                    leaf_rows_by_qid[qid] = leaf_rows_by_qid.get(qid, 0) + 1
                    if int(kid) >= 0:
                        leaf_k_by_qid.setdefault(qid, set()).add(int(kid))

            exp_qid_pure.append(next(iter(qids_in_exp)) if len(qids_in_exp) == 1 else None)

        bad: List[Tuple[int, List[int], List[int]]] = []
        qids_seen = list(rows_by_qid.keys())

        for qid in qids_seen:
            got_set = leaf_k_by_qid.get(qid, set())
            got = sorted(list(got_set))
            missing = [k for k in range(k_rollouts) if k not in got_set]
            extra = [k for k in got if (k < 0 or k >= k_rollouts)]
            if missing or extra:
                bad.append((qid, missing, extra))

        if bad:
            ex = []
            for (qid, missing, extra) in bad[:5]:
                ex.append(
                    f"qid={qid} missing={missing[:12]} extra={extra[:12]} "
                    f"rows={rows_by_qid.get(qid, 0)} leaf_rows={leaf_rows_by_qid.get(qid, 0)} "
                    f"got_k={sorted(list(leaf_k_by_qid.get(qid, set())))[:12]}"
                )
            raise RuntimeError(
                "Full-K violation in C3 experiences before materialize. "
                f"expected leaf k_id in [0..{k_rollouts-1}] for every question_id. "
                f"bad={len(bad)}/{len(qids_seen)} examples: " + "; ".join(ex)
            )

        if all(q is not None for q in exp_qid_pure):
            seen: set[int] = set()
            qid_order: List[int] = []
            for q in exp_qid_pure:
                qq = int(q)  # type: ignore[arg-type]
                if qq not in seen:
                    seen.add(qq)
                    qid_order.append(qq)

            buckets: Dict[int, List] = {qid: [] for qid in qid_order}
            for exp, q in zip(experiences_all, exp_qid_pure):
                buckets[int(q)].append(exp)  # type: ignore[arg-type]

            out = []
            for qid in qid_order:
                out.extend(buckets.get(qid, []))
            return out

        return experiences_all

    def _train_q_critic_if_needed(self, experiences_all) -> Tuple[Optional[dict], Optional[list]]:
        args = self.args
        marl_alg = self._marl_alg()
        critic_tgt_eff = self._resolve_critic_target_eff(marl_alg)
        self._critic_target_eff = critic_tgt_eff

        if _is_rank0(self.strategy):
            logger.info(f"[CriticTarget] raw={self.critic_target_raw} eff={critic_tgt_eff} marl={marl_alg}")

        if not (self._is_c3_mas and marl_alg == "c3" and critic_tgt_eff in {"q", "all"}):
            return None, None
        if self.q_critic_model_group is None:
            raise RuntimeError("critic_target resolves to 'q' but q_critic_model_group is None.")
        if not self._c3_roles or self._c3_layers is None:
            raise RuntimeError("C3 roles/layers are not initialized; c3_task parsing failed.")

        experiences_all = self._c3_regroup_experiences_by_prompt(experiences_all)

        # Fail-fast: C3 requires traj_role_outputs; if missing, pruning likely dropped it.
        self._c3_failfast_missing_traj_maps(experiences_all, where="trainer._train_q_critic_if_needed/pre_materialize")

        from c3.credit.c3.materialize import materialize_c3_batch_data

        batch_data, _, _ = materialize_c3_batch_data(
            experiences_all,
            roles=self._c3_roles,
            k_rollouts=int(args.n_samples_per_prompt),
            require_full_k=True,
        )

        q_cfg = {
            **_Q_CRITIC_VIEW_CFG,
            "loss_type": str(getattr(args, "c3_critic_loss_type", "bce")),
            "train_batch_size": int(getattr(args, "q_critic_train_batch_size", 64)),
            "ctx_limit": int(getattr(args, "critic_ctx_limit", 2560) or 2560),
            "critic_preamble_path": str(getattr(args, "critic_preamble_path", "") or ""),
            "parents": self._c3_parents,
            "disable_ckpt_during_train": False,
            "max_grad_norm": float(getattr(args, "max_grad_norm", 1) or 1),
        }

        q_steps = int(self.critic_train_steps_per_iter)
        if q_steps <= 0:
            return None, batch_data

        use_multi = bool(int(getattr(args, "q_critic_use_multi_steps", 1) or 1))

        if self._q_critic_async_overlap_enabled():
            if getattr(self, "_pending_q_critic_ref", None) is not None:
                self._barrier_pending_q_critic(where="pre_train_q_critic")

            if use_multi:
                last_refs = self.q_critic_model_group.async_run_method(
                    method_name="train_q_critic_multi_steps",
                    batch_data=batch_data,
                    roles=self._c3_roles,
                    layers=self._c3_layers,
                    cfg=q_cfg,
                    num_steps=int(q_steps),
                )
            else:
                last_refs = None
                for _ in range(int(q_steps)):
                    last_refs = self.q_critic_model_group.async_run_method(
                        method_name="train_q_critic",
                        batch_data=batch_data,
                        roles=self._c3_roles,
                        layers=self._c3_layers,
                        cfg=q_cfg,
                    )

            self._pending_q_critic_ref = last_refs
            return None, batch_data

        if use_multi:
            q_refs = self.q_critic_model_group.async_run_method(
                method_name="train_q_critic_multi_steps",
                batch_data=batch_data,
                roles=self._c3_roles,
                layers=self._c3_layers,
                cfg=q_cfg,
                num_steps=int(q_steps),
            )
            q_critic_status = ray.get(q_refs)[0]
        else:
            q_critic_status = None
            for _ in range(int(q_steps)):
                q_refs = self.q_critic_model_group.async_run_method(
                    method_name="train_q_critic",
                    batch_data=batch_data,
                    roles=self._c3_roles,
                    layers=self._c3_layers,
                    cfg=q_cfg,
                )
                q_critic_status = ray.get(q_refs)[0]

        return q_critic_status, batch_data

    def _assert_mappo_vcritic_contract(self, experiences_all, *, where: str) -> None:
        """MAPPO V-critic must use centralized macro-step fields."""
        if self._marl_alg() != "mappo":
            return

        required = (
            "critic_input_ids",
            "critic_attention_mask",
            "critic_action_mask",
            "critic_values",
            "critic_returns",
        )

        if not isinstance(experiences_all, list) or not experiences_all:
            raise RuntimeError(f"[MAPPO][FAIL-FAST] Empty experiences for V-critic at {where}.")

        for e in experiences_all:
            for k in required:
                if getattr(e, k, None) is None:
                    raise RuntimeError(
                        f"[MAPPO][FAIL-FAST] Experience missing {k!r} before V-critic ({where}). "
                        "Centralized critic contract violated."
                    )

            cam = getattr(e, "critic_action_mask", None)
            if isinstance(cam, torch.Tensor):
                if cam.dim() != 2 or cam.shape[1] != 1 or cam.shape[0] <= 0:
                    raise RuntimeError(
                        f"[MAPPO][FAIL-FAST] critic_action_mask must be [B,1], got {tuple(cam.shape)} at {where}."
                    )

    # ---------------------------------------------------------------------
    # Memory safety: prune heavy rollout texts + split actor/critic payloads
    # ---------------------------------------------------------------------

    def _should_keep_rollout_texts(self) -> bool:
        try:
            if bool(getattr(self.args, "keep_rollout_texts", False)):
                return True
        except Exception:
            pass
        try:
            if (
                getattr(self.args, "dump_rollouts_jsonl_path", None)
                and int(getattr(self.args, "dump_rollouts_every", 0) or 0) > 0
            ):
                return True
            if getattr(self.args, "dump_c3_batch_data_path", None):
                return True
        except Exception:
            pass
        return False

    def _prune_experience_inplace(self, exps) -> None:
        if self._should_keep_rollout_texts():
            return
        if not isinstance(exps, list):
            return

        for e in exps:
            try:
                e.prompts = []
                e.labels = []
            except Exception:
                pass

            info = getattr(e, "info", None)
            if not isinstance(info, dict):
                continue

            # Common large blobs / nested text maps.
            heavy_keys = (
                "prompt_text",
                "state_text",
                "output_text",
                "question",
                "traj_role_outputs",
                "traj_role_prompts",
                "c3_path",
                "messages",
            )
            for k in heavy_keys:
                info.pop(k, None)

            # Defensive drop: any remaining strings / nested strings.
            for k in list(info.keys()):
                v = info.get(k, None)
                if isinstance(v, (str, bytes)):
                    info.pop(k, None)
                    continue
                if isinstance(v, (list, tuple)) and v and isinstance(v[0], (str, bytes)):
                    info.pop(k, None)
                    continue
                if isinstance(v, dict):
                    stack = [v]
                    bad = False
                    while stack and not bad:
                        cur = stack.pop()
                        for vv in cur.values():
                            if isinstance(vv, (str, bytes)):
                                bad = True
                                break
                            if isinstance(vv, (list, tuple)) and vv and isinstance(vv[0], (str, bytes)):
                                bad = True
                                break
                            if isinstance(vv, dict):
                                stack.append(vv)
                    if bad:
                        info.pop(k, None)

    def _make_actor_append_payload(self, exps):
        """Actor does not need centralized critic tensors."""
        from openrlhf.trainer.ppo_utils.experience_maker import Experience

        if not isinstance(exps, list) or not exps:
            return exps
        fields = [
            "index",
            "sequences",
            "attention_mask",
            "action_mask",
            "action_log_probs",
            "base_action_log_probs",
            "rollout_log_probs",
            "advantages",
            "returns",
            "info",
        ]
        out = Experience.select(exps, fields)
        self._prune_experience_inplace(out)
        return out

    def _make_critic_append_payload(self, exps):
        """Critic does not need actor logprobs."""
        from openrlhf.trainer.ppo_utils.experience_maker import Experience

        if not isinstance(exps, list) or not exps:
            return exps
        fields = [
            "index",
            "sequences",
            "attention_mask",
            "action_mask",
            "values",
            "returns",
            "critic_input_ids",
            "critic_attention_mask",
            "critic_action_mask",
            "critic_values",
            "critic_returns",
            "info",
        ]
        out = Experience.select(exps, fields)
        self._prune_experience_inplace(out)
        return out

    def _train_v_critic_if_needed(self, experiences_all) -> Tuple[Optional[dict], Optional[object]]:
        args = self.args
        tgt = str(getattr(self, "_critic_target_eff", self.critic_target) or self.critic_target).lower().strip()
        if tgt == "auto":
            tgt = "v"
        if not (self.critic_model_group is not None and tgt in {"v", "all"}):
            return None, None

        critic_payload = self._make_critic_append_payload(experiences_all)
        self._assert_mappo_vcritic_contract(critic_payload, where="trainer._train_v_critic_if_needed/append_payload")

        ray.get(self.critic_model_group.async_run_method_batch(method_name="append", experience=critic_payload))

        if self.strategy.args.deepspeed_enable_sleep:
            ray.get(self.critic_model_group.async_run_method(method_name="reload_states"))

        ref = self.critic_model_group.async_run_method(method_name="fit", num_steps=int(self.critic_train_steps_per_iter))

        if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
            stats = ray.get(ref)[0]
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="offload_states"))
            return stats, None

        return None, ref

    # -------------------------------------------------------------------------
    # fit()
    # -------------------------------------------------------------------------

    def fit(self) -> None:
        args = self.args

        dump_path = getattr(args, "dump_rollouts_jsonl_path", None)
        dump_every = int(getattr(args, "dump_rollouts_every", 0) or 0)
        dump_batch_path = getattr(args, "dump_c3_batch_data_path", None)

        if self.actor_model_groups is not None and getattr(self.args, "dynamic_filtering", False):
            raise RuntimeError("dynamic_filtering is not supported with per_role policy mode yet.")

        checkpoint_states = self._load_checkpoint_states()
        self._sync_actor_weights_to_vllm()

        steps = int(checkpoint_states.get("global_step", 0)) + 1
        last_eval_step = int(checkpoint_states.get("last_eval_step", 0) or 0)

        rollout_iter = 0
        self._pending_q_critic_ref = None
        self._last_q_critic_status = None

        train_epochs = _get_train_epochs(args)
        start_epoch = max(0, int(checkpoint_states.get("episode", 0) or 0))
        last_client_states = None

        data_loader_state_dict = checkpoint_states.get("data_loader_state_dict", {}) or {}

        # merge_epoch resume: rebuild epoch view first, then load dataloader state.
        if self._merge_epoch_enabled() and int(start_epoch) > 0:
            self._rebuild_task_train_epoch(int(start_epoch))

        # Tolerant dataloader state restore (old ckpts may be incompatible).
        resume_has_state = False
        if data_loader_state_dict:
            try:
                self.prompts_dataloader.load_state_dict(data_loader_state_dict)
                resume_has_state = True
            except Exception as e:
                logger.warning(f"[WARN] failed to load prompts_dataloader state_dict (resume without it): {e}")
                data_loader_state_dict = {}
                resume_has_state = False

        # ---- Optional: eval at start ----
        if bool(getattr(args, "eval_at_start", False)):
            try:
                already_done = bool(checkpoint_states.get("eval_at_start_done", False))
            except Exception:
                already_done = False

            if not already_done:
                # Auto-enable save_on_eval for progress-based schedules.
                try:
                    progress_ratio = float(getattr(args, "eval_every_ratio", 0.0) or 0.0)
                    progress_percent = float(getattr(args, "eval_every_percent", 0.0) or 0.0)
                    progress_eval_enabled = (progress_ratio > 0.0) or (progress_percent > 0.0)
                except Exception:
                    progress_eval_enabled = False

                if progress_eval_enabled and not bool(getattr(args, "save_on_eval", False)):
                    for tgt in (args, getattr(self.strategy, "args", None)):
                        if tgt is None:
                            continue
                        try:
                            tgt.save_on_eval = True
                        except Exception:
                            pass
                    logger.info("[INFO] auto-enabled save_on_eval (eval_at_start + eval_every_percent/ratio).")

                try:
                    has_eval = (
                        self.eval_dataloader is not None
                        and hasattr(self.eval_dataloader, "__len__")
                        and len(self.eval_dataloader) > 0
                    )
                except Exception:
                    has_eval = False

                if has_eval:
                    try:
                        gs0 = int(checkpoint_states.get("global_step", 0) or 0)
                    except Exception:
                        gs0 = 0

                    logger.info(f"[EvalAtStart] running initial evaluation at global_step={gs0}.")
                    try:
                        self._sync_actor_weights_to_vllm()
                    except Exception:
                        pass

                    _ = self.evaluate(
                        self.eval_dataloader,
                        global_step=int(gs0),
                        temperature=float(
                            0.7 if getattr(args, "eval_temperature", None) is None
                            else getattr(args, "eval_temperature")
                        ),
                        n_samples_per_prompt=int(getattr(args, "eval_n_samples_per_prompt", 1) or 1),
                    )

                    last_eval_step = int(gs0)
                    checkpoint_states["last_eval_step"] = int(last_eval_step)
                    checkpoint_states["eval_at_start_done"] = True

                    if bool(getattr(args, "save_on_eval", False)):
                        client_states0 = {
                            "global_step": int(gs0),
                            "critic_warmup_done": bool(checkpoint_states.get("critic_warmup_done", False)),
                            "critic_warmup_step": int(checkpoint_states.get("critic_warmup_step", 0) or 0),
                            "critic_warmup_total": int(getattr(self, "critic_warmup_steps", 0) or 0),
                            "episode": int(checkpoint_states.get("episode", 0) or 0),
                            "train_epoch": int(checkpoint_states.get("episode", 0) or 0),
                            "iter_in_epoch": int(checkpoint_states.get("iter_in_epoch", 0) or 0),
                            "train_epochs": int(_get_train_epochs(args)),
                            "data_loader_state_dict": None,
                            "last_eval_step": int(last_eval_step),
                            "eval_at_start_done": bool(checkpoint_states.get("eval_at_start_done", False)),
                        }
                        self._barrier_pending_q_critic(where="before_save_on_eval_at_start")
                        self._save_checkpoints(tag=f"eval_start_step{gs0}", client_states=client_states0)
                else:
                    logger.info("[EvalAtStart] skipped: eval_dataloader is empty or None.")

        # ---- critic warmup gate ----
        warmup_total = int(getattr(self, "critic_warmup_steps", 0) or 0)
        warmup_done = int(checkpoint_states.get("critic_warmup_step", 0) or 0)
        warmup_done_flag = bool(checkpoint_states.get("critic_warmup_done", False))

        if warmup_total > 0:
            raw = str(getattr(self, "critic_target_raw", getattr(self.args, "critic_target", "auto")) or "auto").lower()
            marl_alg = self._marl_alg()

            eff = raw
            if raw == "auto":
                if marl_alg == "c3":
                    eff = "q" if getattr(self, "q_critic_model_group", None) is not None else "none"
                elif marl_alg == "magrpo":
                    eff = "none"
                else:
                    eff = "v" if getattr(self, "critic_model_group", None) is not None else "none"

            if eff == "q" and getattr(self, "q_critic_model_group", None) is None:
                eff = "none"
            if eff == "v" and getattr(self, "critic_model_group", None) is None:
                eff = "none"

            if eff not in {"q", "v"}:
                logger.info(f"[CriticWarmup] disabled: critic_target={raw} marl={marl_alg}")
                warmup_total = 0
                warmup_done = 0
                warmup_done_flag = True

        # Older checkpoints did warmup inline.
        if (
            warmup_total > 0
            and ("critic_warmup_step" not in checkpoint_states)
            and ("critic_warmup_done" not in checkpoint_states)
            and int(checkpoint_states.get("global_step", 0) or 0) > 0
        ):
            warmup_done = warmup_total
            warmup_done_flag = True

        if warmup_total > 0 and (not warmup_done_flag) and warmup_done < warmup_total:
            logger.info(f"Starting critic-only warmup: {warmup_done}/{warmup_total}")
            warmup_done = self._run_critic_warmup_stage(warmup_total=warmup_total, warmup_done=warmup_done)
            checkpoint_states["critic_warmup_step"] = int(warmup_done)
            checkpoint_states["critic_warmup_done"] = bool(warmup_done >= warmup_total)
            logger.info(f"Critic-only warmup finished: {warmup_done}/{warmup_total}")

        dyn_state: Tuple[list, int] = ([], 0)

        # ---- main loop ----
        for epoch_idx in range(start_epoch, train_epochs):
            is_resume_epoch = bool(resume_has_state) and int(epoch_idx) == int(start_epoch)

            # merge_epoch: rebuild each epoch; resume epoch keeps loaded dataloader state
            if self._merge_epoch_enabled() and (not is_resume_epoch):
                self._rebuild_task_train_epoch(int(epoch_idx))

            self._maybe_set_dataloader_epoch(self.prompts_dataloader, epoch_idx, resume_has_state, start_epoch)

            try:
                total_batches = int(self.prompts_dataloader.__len__())
            except Exception:
                total_batches = None

            pbar = tqdm(
                total=total_batches,
                desc=f"Epoch [{epoch_idx + 1}/{train_epochs}]",
                disable=not _is_rank0(self.strategy),
            )

            for iter_in_epoch, batch in enumerate(self.prompts_dataloader):
                timing = {} if self._timing_enabled() else None
                t_iter0 = time.perf_counter() if timing is not None else None

                _, rand_prompts, labels, meta_jsons = _unpack_prompt_batch(batch, where="train fit() loop")
                self._strict_check_vllm_before_rollout()

                gen_kwargs = dict(self.generate_kwargs)
                if meta_jsons is not None:
                    gen_kwargs["all_metas"] = list(meta_jsons)

                t_roll0 = time.perf_counter() if timing is not None else None
                rollout_samples = self.samples_generator.generate_samples(
                    rand_prompts,
                    labels,
                    remote_reward_model=self.remote_reward_model,
                    phase="train",
                    global_step=int(steps),
                    epoch_idx=int(epoch_idx),
                    iter_in_epoch=int(iter_in_epoch),
                    **gen_kwargs,
                )
                if timing is not None and t_roll0 is not None:
                    timing["time/rollout_generate_ms"] = (time.perf_counter() - t_roll0) * 1000.0

                pbar.update(1)

                selected, pass_rate, dyn_state = self._dyn_filter_update(rollout_samples, dyn_state)
                if self.args.dynamic_filtering:
                    if selected is None:
                        buf_groups, _n_total = dyn_state
                        logger.info(
                            "filtered_prompts %d < rollout_batch_size %s, continue sampling",
                            len(buf_groups),
                            self.args.rollout_batch_size,
                        )
                        continue
                    rollout_samples = selected
                    logger.info(f"Dynamic filtering pass rate: {pass_rate:.2f}%")

                t_qb0 = time.perf_counter() if timing is not None else None
                q_critic_status_prev = self._barrier_pending_q_critic(where="before_make_experiences")
                if timing is not None and t_qb0 is not None:
                    timing["time/q_critic_barrier_ms"] = (time.perf_counter() - t_qb0) * 1000.0

                t_exp0 = time.perf_counter() if timing is not None else None
                experiences_all, experiences_by_role = self._make_experiences(rollout_samples)
                if timing is not None and t_exp0 is not None:
                    timing["time/make_experiences_ms"] = (time.perf_counter() - t_exp0) * 1000.0

                experiences_all, experiences_by_role = self._maybe_balance_experiences(experiences_all, experiences_by_role)

                t_q0 = time.perf_counter() if timing is not None else None
                q_critic_status, c3_batch_data = self._train_q_critic_if_needed(experiences_all)
                if timing is not None and t_q0 is not None:
                    timing["time/q_critic_train_or_submit_ms"] = (time.perf_counter() - t_q0) * 1000.0

                q_critic_status_log = _as_stats_dict(q_critic_status) or _as_stats_dict(q_critic_status_prev)

                self._maybe_dump_rollouts_jsonl(
                    dump_path=str(dump_path) if dump_path else None,
                    dump_every=int(dump_every),
                    rollout_iter=int(rollout_iter),
                    steps=int(steps),
                    epoch_idx=int(epoch_idx),
                    iter_in_epoch=int(iter_in_epoch),
                    train_epochs=int(train_epochs),
                    rand_prompts=rand_prompts,
                    experiences_all=experiences_all,
                    q_critic_status_log=q_critic_status_log,
                )

                self._maybe_dump_c3_batch_data_jsonl(
                    dump_batch_path=str(dump_batch_path) if dump_batch_path else None,
                    dump_every=int(dump_every),
                    rollout_iter=int(rollout_iter),
                    steps=int(steps),
                    epoch_idx=int(epoch_idx),
                    iter_in_epoch=int(iter_in_epoch),
                    train_epochs=int(train_epochs),
                    c3_batch_data=c3_batch_data,
                )

                # After optional dumps, drop heavy texts ASAP.
                self._prune_experience_inplace(experiences_all)

                t_v0 = time.perf_counter() if timing is not None else None
                v_critic_stats_early, v_critic_status_ref = self._train_v_critic_if_needed(experiences_all)
                if timing is not None and t_v0 is not None:
                    timing["time/v_critic_call_ms"] = (time.perf_counter() - t_v0) * 1000.0

                # Append actor buffer (slim payload)
                if self.actor_model_groups is not None:
                    refs = []
                    for rn in self.role_names:
                        payload = self._make_actor_append_payload(experiences_by_role[rn])
                        refs.extend(
                            self.actor_model_groups[rn].async_run_method_batch(method_name="append", experience=payload)
                        )
                    ray.get(refs)
                else:
                    payload = self._make_actor_append_payload(experiences_all)
                    ray.get(self.actor_model_group.async_run_method_batch(method_name="append", experience=payload))

                # Train actor
                t_a0 = time.perf_counter() if timing is not None else None
                status = self.ppo_train(steps)
                if timing is not None and t_a0 is not None:
                    timing["time/actor_train_ms"] = (time.perf_counter() - t_a0) * 1000.0

                if isinstance(v_critic_stats_early, dict):
                    status.update(v_critic_stats_early)

                if v_critic_status_ref is not None:
                    t_vw0 = time.perf_counter() if timing is not None else None
                    v_stats = ray.get(v_critic_status_ref)[0]
                    if timing is not None and t_vw0 is not None:
                        timing["time/v_critic_wait_ms"] = (time.perf_counter() - t_vw0) * 1000.0

                    if isinstance(v_stats, dict):
                        status.update(v_stats)
                    if self.strategy.args.deepspeed_enable_sleep:
                        ray.get(self.critic_model_group.async_run_method(method_name="offload_states"))

                if isinstance(q_critic_status_log, dict):
                    status.update(q_critic_status_log)

                rollout_iter += 1

                # Update KL
                if "kl" in status:
                    n_steps = int(args.rollout_batch_size) * int(args.n_samples_per_prompt)
                    n_roles = 1
                    try:
                        if experiences_all and isinstance(getattr(experiences_all[0], "info", None), dict):
                            nr = experiences_all[0].info.get("num_roles", None)
                            if isinstance(nr, torch.Tensor):
                                n_roles = int(nr[0].item())
                    except Exception:
                        pass
                    if self.actor_model_groups is not None:
                        n_roles = max(n_roles, len(self.role_names))
                    self.kl_ctl.update(status["kl"], n_steps * max(1, int(n_roles)))

                if self.args.dynamic_filtering and pass_rate is not None:
                    status["dynamic_filtering_pass_rate"] = pass_rate

                status.update(self._rewardprovider_rollout_metrics(experiences_all))
                if steps % args.logging_steps == 0:
                    status.update(self._marl_response_clip_metrics(experiences_all))
                    status.update(self._c3_credit_rollout_metrics(experiences_all))

                if timing is not None and t_iter0 is not None:
                    timing["time/iter_ms"] = (time.perf_counter() - t_iter0) * 1000.0
                    status.update(timing)

                logger.info(
                    f"Global step {steps} | epoch {epoch_idx + 1}/{train_epochs} | iter {iter_in_epoch + 1}"
                    + (f"/{total_batches}" if total_batches is not None else "")
                    + f": {status}"
                )

                decoded0 = self._decode_first_experience(experiences_all)
                if experiences_all:
                    status["generated_samples"] = self._make_generated_sample_meta(experiences_all[0], decoded0)

                client_states = {
                    "global_step": int(steps),
                    "critic_warmup_done": bool(checkpoint_states.get("critic_warmup_done", False)),
                    "critic_warmup_step": int(checkpoint_states.get("critic_warmup_step", 0) or 0),
                    "critic_warmup_total": int(getattr(self, "critic_warmup_steps", 0) or 0),
                    "episode": int(epoch_idx),
                    "train_epoch": int(epoch_idx),
                    "iter_in_epoch": int(iter_in_epoch),
                    "train_epochs": int(train_epochs),
                    "data_loader_state_dict": None,
                    "last_eval_step": int(last_eval_step),
                }

                last_client_states = client_states
                try:
                    last_eval_step = self.save_logs_and_checkpoints(
                        args,
                        steps,
                        pbar,
                        status,
                        client_states,
                        last_step=int(last_eval_step),
                    )
                except TypeError:
                    self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)
                    last_eval_step = int(client_states.get("last_eval_step", last_eval_step) or last_eval_step)

                client_states["last_eval_step"] = int(last_eval_step)
                steps += 1

            try:
                pbar.close()
            except Exception:
                pass

        # Final eval + save-on-eval
        try:
            last_step = int(steps) - 1
        except Exception:
            last_step = None

        try:
            eval_steps = getattr(args, "eval_steps", None)
            eval_enabled = (
                last_step is not None
                and self.eval_dataloader is not None
                and hasattr(self.eval_dataloader, "__len__")
                and len(self.eval_dataloader) > 0
                and eval_steps is not None
                and not (isinstance(eval_steps, float) and math.isinf(eval_steps))
                and int(eval_steps) > 0
            )
        except Exception:
            eval_enabled = False

        if eval_enabled:
            try:
                need_final_eval = (int(last_step) % int(eval_steps)) != 0
            except Exception:
                need_final_eval = False

            if need_final_eval:
                try:
                    self.evaluate(
                        self.eval_dataloader,
                        int(last_step),
                        args.eval_temperature,
                        args.eval_n_samples_per_prompt,
                    )
                    if bool(getattr(args, "save_on_eval", False)):
                        self._barrier_pending_q_critic(where="before_save_on_eval_final")
                        self._save_checkpoints(tag=f"eval_step{int(last_step)}_final", client_states=last_client_states)
                except Exception as e:
                    logger.warning(f"[WARN] final eval/save_on_eval failed: {e}")

        self._barrier_pending_q_critic(where="train_end")

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()
