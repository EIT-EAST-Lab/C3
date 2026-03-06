# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

from __future__ import annotations

import ast
import json
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field, fields
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
import torch

from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from openrlhf.utils.utils import remove_pad_token, zero_pad_sequences

logger = init_logger(__name__)

# =============================================================================
# Tensor tree utils
# =============================================================================
def _tree_map(fn, x: Any):
    if isinstance(x, torch.Tensor):
        return fn(x)
    if isinstance(x, list):
        return [_tree_map(fn, v) for v in x]
    if isinstance(x, tuple):
        return tuple(_tree_map(fn, v) for v in x)
    if isinstance(x, dict):
        return {k: _tree_map(fn, v) for k, v in x.items()}
    return x


def to(x: Any, device: torch.device):
    return _tree_map(lambda t: t.to(device), x)


def pin_memory(x: Any):
    return _tree_map(lambda t: t.pin_memory(), x)


# =============================================================================
# Small helpers
# =============================================================================
_TEXT_TYPES = (str, bytes)
_MAS_HINT_KEYS = ("question_id", "k_id", "is_leaf", "c3_node_id")

# Common heavy blobs stored in Experience.info; drop unless explicitly kept.
# NOTE: C3/MAS may depend on these; ensure keep_rollout_texts=True when needed.
_HEAVY_ROLLOUT_INFO_KEYS = (
    "prompt_text",
    "state_text",
    "output_text",
    "traj_role_outputs",
    "traj_role_prompts",
)


def _ensure_info_dict(obj) -> Dict[str, Any]:
    info = getattr(obj, "info", None)
    if isinstance(info, dict):
        return info
    info = {}
    try:
        obj.info = info
    except Exception:
        pass
    return info


def _looks_like_mas(info: Dict[str, Any]) -> bool:
    return isinstance(info, dict) and any(k in info for k in _MAS_HINT_KEYS)


def _as_int_scalar(v) -> Optional[int]:
    if v is None:
        return None
    try:
        if isinstance(v, torch.Tensor):
            vv = v.view(-1)
            return int(vv[0].item()) if vv.numel() else None
        if isinstance(v, (list, tuple)):
            return int(v[0]) if v else None
        return int(v)
    except Exception:
        return None


def _as_len_scalar(v) -> int:
    return int(_as_int_scalar(v) or 0)


def _as_long_1d(x: Any, *, B: int, fallback: torch.Tensor) -> torch.Tensor:
    """Best-effort: x -> int64[B]. Return fallback on mismatch/error."""
    try:
        t = x.to(torch.long).view(-1) if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.long).view(-1)
        return t if int(t.numel()) == int(B) else fallback
    except Exception:
        return fallback


def _normalize_text_list(x: Any, *, B: int) -> List[str]:
    """Normalize x to list[str] of length B."""
    if x is None:
        return [""] * B

    if isinstance(x, torch.Tensor):
        try:
            v = x.detach().cpu().view(-1)
            if int(v.numel()) == B:
                return [str(v[i].item()) for i in range(B)]
            if int(v.numel()) == 1:
                return [str(v[0].item())] * B
        except Exception:
            pass
        return [""] * B

    if isinstance(x, (list, tuple)):
        xs = [str(v) for v in x]
        if len(xs) == B:
            return xs
        if len(xs) == 1 and B > 1:
            return xs * B
        if not xs:
            return [""] * B
        xs = xs[:B]
        if len(xs) < B:
            xs.extend([xs[-1]] * (B - len(xs)))
        return xs

    return [str(x)] * B


def _parse_roles_topo(rt: Any) -> Optional[List[str]]:
    """roles_topo: list[str] or repr string like 'A->B', 'A,B', '[...]'."""
    if rt is None:
        return None

    def _parse_str(s: str) -> Optional[List[str]]:
        s = (s or "").strip()
        if not s:
            return None

        if s.startswith("[") and s.endswith("]"):
            for loader in (json.loads, ast.literal_eval):
                try:
                    v = loader(s)
                    if isinstance(v, list):
                        out = [str(x).strip() for x in v if str(x).strip()]
                        return out or None
                except Exception:
                    pass

        if "->" in s:
            out = [p.strip() for p in s.split("->") if p.strip()]
            return out or None
        if "," in s:
            out = [p.strip() for p in s.split(",") if p.strip()]
            return out or None
        return [s]

    if isinstance(rt, str):
        return _parse_str(rt)

    if isinstance(rt, list):
        if not rt:
            return None
        if all(isinstance(x, str) for x in rt):
            uniq = [x.strip() for x in rt if (x or "").strip()]
            if not uniq:
                return None
            if len(uniq) == 1:
                return _parse_str(uniq[0])
            simple = all(("->" not in t and "," not in t and not (t.startswith("[") and t.endswith("]"))) for t in uniq)
            return uniq if simple else _parse_str(uniq[0])

        for x in rt:
            if isinstance(x, (list, tuple)):
                out = [str(y).strip() for y in x if str(y).strip()]
                if out:
                    return out
            if isinstance(x, str):
                v = _parse_str(x)
                if v:
                    return v
        return None

    return None


def _canon_marl(name: Any) -> str:
    try:
        from c3.algorithms.registry import canonical_name

        return canonical_name(name)
    except Exception:
        return str(name or "auto").lower().strip()


def _is_text_blob(v: Any) -> bool:
    """True if v contains nested string blobs (for Ray object size safety)."""
    if isinstance(v, _TEXT_TYPES):
        return True

    if isinstance(v, (list, tuple)):
        if not v:
            return False
        # Fast paths
        if any(isinstance(x, _TEXT_TYPES) for x in v):
            return True
        if all(isinstance(x, _TEXT_TYPES) for x in v):
            return True
        # Nested containers
        stack: List[Any] = [x for x in v if isinstance(x, (dict, list, tuple))]
        while stack:
            cur = stack.pop()
            if isinstance(cur, _TEXT_TYPES):
                return True
            if isinstance(cur, dict):
                vals = cur.values()
                if any(isinstance(x, _TEXT_TYPES) for x in vals):
                    return True
                for x in vals:
                    if isinstance(x, (dict, list, tuple)):
                        stack.append(x)
            elif isinstance(cur, (list, tuple)):
                if any(isinstance(x, _TEXT_TYPES) for x in cur):
                    return True
                for x in cur:
                    if isinstance(x, (dict, list, tuple)):
                        stack.append(x)
        return False

    if isinstance(v, dict):
        stack: List[dict] = [v]
        while stack:
            cur = stack.pop()
            for vv in cur.values():
                if isinstance(vv, _TEXT_TYPES):
                    return True
                if isinstance(vv, (list, tuple)):
                    if any(isinstance(x, _TEXT_TYPES) for x in vv):
                        return True
                    for x in vv:
                        if isinstance(x, (dict, list, tuple)):
                            # recurse into nested containers
                            if isinstance(x, dict):
                                stack.append(x)
                            else:
                                # list/tuple nested
                                if _is_text_blob(x):
                                    return True
                elif isinstance(vv, dict):
                    stack.append(vv)
        return False

    return False


def _prune_info_text_blobs_inplace(info: Dict[str, Any], *, preserve: Optional[set] = None) -> None:
    preserve = preserve or set()
    for k in _HEAVY_ROLLOUT_INFO_KEYS:
        if k not in preserve:
            info.pop(k, None)

    for k in list(info.keys()):
        if k in preserve:
            continue
        v = info.get(k, None)
        if _is_text_blob(v):
            info.pop(k, None)


# =============================================================================
# Experience container
# =============================================================================
@dataclass
class Experience:
    """A micro-batch container (mostly [B,S]) + batch-aligned info."""

    index: List[int] = field(default_factory=list)

    sequences: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    action_mask: Optional[torch.Tensor] = None

    # Centralized critic (optional; e.g. MAPPO state critic)
    critic_input_ids: Optional[torch.Tensor] = None
    critic_attention_mask: Optional[torch.Tensor] = None
    critic_action_mask: Optional[torch.Tensor] = None
    critic_values: Optional[torch.Tensor] = None
    critic_returns: Optional[torch.Tensor] = None

    action_log_probs: Optional[torch.Tensor] = None
    base_action_log_probs: Optional[torch.Tensor] = None
    rollout_log_probs: Optional[torch.Tensor] = None

    values: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    kl: Optional[torch.Tensor] = None

    prompts: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)

    rewards: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    info: Dict[str, Any] = field(default_factory=dict)

    @torch.no_grad()
    def to_device(self, device: torch.device):
        for k, v in self.__dict__.items():
            setattr(self, k, to(v, device))
        return self

    def pin_memory(self):
        for k, v in self.__dict__.items():
            setattr(self, k, pin_memory(v))
        return self

    @staticmethod
    def select(experiences: List["Experience"], fields_: List[str]) -> List["Experience"]:
        out: List["Experience"] = []
        for exp in experiences:
            ne = Experience()
            for f in fields_:
                if hasattr(exp, f):
                    setattr(ne, f, getattr(exp, f))
            out.append(ne)
        return out

    @staticmethod
    def _merge_item(items: List, pad_value: int = 0) -> Union[torch.Tensor, list, dict, Any]:
        if not items:
            return None

        x0 = items[0]

        if isinstance(x0, torch.Tensor):
            if x0.dim() <= 1:
                return torch.cat([t.view(-1) for t in items], dim=0)
            return zero_pad_sequences(items, side="right", value=pad_value)

        if isinstance(x0, list):
            return sum(items, [])

        if isinstance(x0, dict):
            buckets: Dict[str, List[Any]] = {}
            for d in items:
                for k, v in d.items():
                    buckets.setdefault(k, []).append(v)
            return {k: Experience._merge_item(vs, pad_value) for k, vs in buckets.items()}

        if isinstance(x0, (str, int, float, bool)) or x0 is None:
            return items

        raise ValueError(f"Unsupported type in Experience merge: {type(x0)}")

    @staticmethod
    def concat_experiences(experiences_list: List["Experience"], pad_token_id: int) -> "Experience":
        if not experiences_list:
            return Experience()

        merged: Dict[str, Any] = {}
        for f in fields(Experience):
            name = f.name
            vals = [getattr(e, name) for e in experiences_list]
            merged[name] = Experience._merge_item(vals, pad_token_id if name == "sequences" else 0)

        out = Experience(**merged)

        if isinstance(out.sequences, torch.Tensor) and isinstance(out.info, dict):
            B = int(out.sequences.shape[0])
            for k in ("question_id", "k_id", "role_id", "traj_id", "is_leaf", "adv_group_id", "magrpo_group_id"):
                v = out.info.get(k, None)
                if isinstance(v, torch.Tensor) and v.dim() == 1 and int(v.numel()) != B:
                    raise RuntimeError(f"[Experience.concat] info['{k}'] numel={int(v.numel())} != B={B}")

        return out


# =============================================================================
# Reward post-processing (remote/local RM)
# =============================================================================
def update_samples_with_rewards(rewards_info, samples_list):
    if not rewards_info:
        return samples_list

    bs_list: List[int] = []
    for s in samples_list:
        if not isinstance(s.sequences, torch.Tensor):
            raise RuntimeError("update_samples_with_rewards expects samples.sequences to be a tensor.")
        bs_list.append(int(s.sequences.shape[0]))

    rewards_list = torch.cat([torch.as_tensor(info["rewards"]) for info in rewards_info], dim=0).split(bs_list)
    scores_list = (
        torch.cat([torch.as_tensor(info["scores"]) for info in rewards_info], dim=0).split(bs_list)
        if "scores" in rewards_info[0]
        else rewards_list
    )

    merged_logs = None
    if "extra_logs" in rewards_info[0]:
        merged_logs = {
            k: torch.cat([torch.as_tensor(info["extra_logs"][k]) for info in rewards_info], dim=0).split(bs_list)
            for k in rewards_info[0]["extra_logs"].keys()
        }

    for i, s in enumerate(samples_list):
        s.rewards = rewards_list[i]
        s.scores = scores_list[i]

        info = _ensure_info_dict(s)
        info["reward"] = rewards_list[i]
        info["score"] = scores_list[i]
        if merged_logs is not None:
            for k, v in merged_logs.items():
                info[k] = v[i]
    return samples_list


# =============================================================================
# Rollout sample generation (vLLM-only path)
# =============================================================================
class SamplesGenerator:
    def __init__(self, vllm_engines, strategy, tokenizer, prompt_max_len):
        self.strategy = strategy
        self.args = strategy.args
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Experience]:
        if self.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        rollout_samples = self._generate_vllm(all_prompts, all_labels, **generate_kwargs)

        if self.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "sleep")

        return rollout_samples

    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            return self.tokenizer(texts, add_special_tokens=False, max_length=max_length, truncation=True)
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Experience]:
        from vllm import SamplingParams

        args = self.args
        llms = self.vllm_engines

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
            logprobs=1 if args.enable_vllm_is_correction else None,
        )
        max_response_length = kwargs.get("max_new_tokens", 1024)
        truncate_length = self.prompt_max_len + max_response_length

        nsp = int(kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt))
        all_prompts = sum([[p] * nsp for p in all_prompts], [])
        all_labels = sum([[l] * nsp for l in all_labels], [])

        refs, used_llms = [], []
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            chunk = all_prompts[i * batch_size : (i + 1) * batch_size]
            if not chunk:
                continue
            used_llms.append(llm)
            refs.append(llm.add_requests.remote(sampling_params=sampling_params, prompts=chunk))

        if refs:
            ray.get(refs)
            all_outputs = sum(ray.get([llm.get_responses.remote() for llm in used_llms]), [])
        else:
            all_outputs = []

        samples_list: List[Experience] = []
        for i, output in enumerate(all_outputs):
            prompt = all_prompts[i]
            label = all_labels[i]

            input_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
            attn = [1] * len(input_ids)

            sequences = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attn, dtype=torch.long)

            action_mask = torch.zeros_like(attention_mask)
            response_len = len(output.outputs[0].token_ids)
            action_mask[len(output.prompt_token_ids) : len(output.prompt_token_ids) + response_len] = 1

            rollout_log_probs = None
            if args.enable_vllm_is_correction:
                response_ids = list(output.outputs[0].token_ids)
                lp = [logprob[response_ids[j]].logprob for j, logprob in enumerate(output.outputs[0].logprobs)]
                rollout_log_probs = torch.tensor([0.0] * len(list(output.prompt_token_ids)) + lp, dtype=torch.float32)
                rollout_log_probs = rollout_log_probs[1:truncate_length].to("cpu")

            sequences = sequences[:truncate_length].to("cpu")
            attention_mask = attention_mask[:truncate_length].to("cpu")
            action_mask = action_mask[1:truncate_length].to("cpu")

            total_length = attention_mask.float().sum()
            is_clipped = response_len >= max_response_length

            info = {
                "response_length": torch.tensor([response_len]),
                "total_length": torch.tensor([total_length]),
                "response_clip_ratio": torch.tensor([is_clipped]),
            }

            samples_list.append(
                Experience(
                    sequences=sequences.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    action_mask=action_mask.unsqueeze(0),
                    rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
                    prompts=[prompt],
                    labels=[label],
                    info=info,
                )
            )

        remote_reward_model = kwargs.get("remote_reward_model", None)
        if remote_reward_model:
            all_queries = sum(
                [
                    self.tokenizer.batch_decode(remove_pad_token(s.sequences, s.attention_mask), skip_special_tokens=False)
                    for s in samples_list
                ],
                [],
            )
            all_prompts_1d = sum([s.prompts for s in samples_list], [])
            all_labels_1d = sum([s.labels for s in samples_list], [])
            rewards_info = ray.get(remote_reward_model.get_rewards.remote(all_queries, all_prompts_1d, all_labels_1d))
            update_samples_with_rewards(rewards_info, samples_list)

        return samples_list


# =============================================================================
# Experience maker (main training path)
# =============================================================================
class RemoteExperienceMaker(ABC):
    def __init__(
        self,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        initial_model_group: RayActorGroup,
        kl_controller,
        strategy=None,
        tokenizer=None,
        remote_reward_model=None,
        **kwargs,
    ):
        super().__init__()

        for bad_key in ("rollout_generator", "generate_for_roles"):
            if bad_key in kwargs:
                raise RuntimeError(f"RemoteExperienceMaker: legacy kwarg {bad_key!r} has been removed.")

        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.initial_model_group = initial_model_group
        self.kl_ctl = kl_controller

        self.strategy = strategy
        self.args = strategy.args
        self.advantage_estimator = self.args.advantage_estimator

        self.q_critic_model_group = kwargs.get("q_critic_model_group", None)
        self.remote_rm_url = self.args.remote_rm_url
        self.remote_reward_model = remote_reward_model
        self.tokenizer = tokenizer

    # -------------------------------------------------------------------------
    # Ray helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _flatten_ray_batch_outputs(ref, duplicate_factor: int) -> List[Any]:
        return sum(ray.get(ref)[::duplicate_factor], [])

    @staticmethod
    def _group_by_question_id(samples: List[Experience]) -> Optional[List[List[Experience]]]:
        order: List[int] = []
        groups: Dict[int, List[Experience]] = {}
        for s in samples:
            info = s.info if isinstance(s.info, dict) else {}
            qid = _as_int_scalar(info.get("question_id", None))
            if qid is None:
                return None
            if qid not in groups:
                groups[qid] = []
                order.append(qid)
            groups[qid].append(s)
        return [groups[q] for q in order]

    # -------------------------------------------------------------------------
    # MAPPO: centralized state critic (prefers state_text; falls back to prompt_text)
    # -------------------------------------------------------------------------
    def _should_use_mappo_state_critic(self, samples_list: List[Experience]) -> bool:
        marl_alg = _canon_marl(getattr(self.args, "marl_algorithm", "auto"))
        if marl_alg != "mappo" or self.critic_model_group is None:
            return False

        info0 = getattr(samples_list[0], "info", None)
        if not isinstance(info0, dict):
            raise RuntimeError(
                "MAPPO requires Experience.info dict with 'state_text' (preferred) or 'prompt_text'. "
                "Ensure rollout generator writes these fields."
            )
        if ("state_text" not in info0) and ("prompt_text" not in info0):
            raise RuntimeError("MAPPO requires Experience.info['state_text'] or ['prompt_text'] (batch-aligned list[str]).")
        return True

    def _get_mappo_state_critic_tokenizer(self):
        tok = getattr(self, "_mappo_state_critic_tokenizer", None)
        if tok is not None:
            return tok

        from transformers import AutoTokenizer

        critic_pretrain = str(getattr(self.args, "critic_pretrain", "") or "")
        if not critic_pretrain:
            raise RuntimeError("MAPPO state critic requires args.critic_pretrain to build tokenizer.")

        tok = AutoTokenizer.from_pretrained(
            critic_pretrain,
            trust_remote_code=True,
            use_fast=not getattr(self.args, "disable_fast_tokenizer", False),
        )
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        if tok.pad_token is None:
            raise RuntimeError("critic tokenizer has no pad/eos/unk token; set pad_token for critic_pretrain.")

        self._mappo_state_critic_tokenizer = tok
        return tok

    # -------------------------------------------------------------------------
    # Memory safety: drop heavy rollout texts unless explicitly kept
    # -------------------------------------------------------------------------
    def _should_keep_rollout_texts(self) -> bool:
        try:
            return bool(getattr(self.args, "keep_rollout_texts", False))
        except Exception:
            return False

    def _prune_rollout_text_fields_inplace(self, exp: Experience):
        keep_rollout_texts = bool(getattr(self.args, "keep_rollout_texts", False))
        if keep_rollout_texts:
            return

        info = getattr(exp, "info", None)
        if not isinstance(info, dict) or not info:
            return

        # ------------------------------------------------------------------
        # C3(Q-critic / credit) requires per-node trajectory role maps.
        # If we prune them here, PPOTrainer._train_q_critic_if_needed will crash
        # in materialize_c3_batch_data/materialize_c3_tree_groups.
        #
        # NOTE: we still allow PPOTrainer to prune them later (after Q-critic).
        # ------------------------------------------------------------------
        preserve = None
        marl_alg = _canon_marl(getattr(self.args, "marl_algorithm", "auto"))
        if marl_alg == "c3":
            preserve = {"traj_role_outputs", "traj_role_prompts"}

        _prune_info_text_blobs_inplace(info, preserve=preserve)
        exp.info = info

    def _build_mappo_state_critic_inputs(
        self, samples_list: List[Experience]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        tok = self._get_mappo_state_critic_tokenizer()
        try:
            max_len = max(256, int(getattr(self.args, "mappo_state_max_len", 2560)))
        except Exception:
            max_len = 2560

        seqs_list: List[torch.Tensor] = []
        attn_list: List[torch.Tensor] = []
        cam_list: List[torch.Tensor] = []

        for s in samples_list:
            info = s.info if isinstance(getattr(s, "info", None), dict) else {}
            if not isinstance(s.sequences, torch.Tensor):
                raise RuntimeError("MAPPO state critic inputs require sequences to exist.")
            B = int(s.sequences.shape[0])

            raw_texts = info.get("state_text", None)
            if raw_texts is None:
                raw_texts = info.get("prompt_text", None)
            texts = _normalize_text_list(raw_texts, B=B)

            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            ids = enc["input_ids"].to("cpu")
            attn = enc["attention_mask"].to(torch.long).to("cpu")
            cam = torch.ones((B, 1), dtype=torch.float32).to("cpu")

            s.critic_input_ids = ids
            s.critic_attention_mask = attn
            s.critic_action_mask = cam

            seqs_list.append(ids)
            attn_list.append(attn)
            cam_list.append(cam)

            self._prune_rollout_text_fields_inplace(s)

        return seqs_list, attn_list, cam_list

    # -------------------------------------------------------------------------
    # Micro-batching (prompt-atomic when required)
    # -------------------------------------------------------------------------
    def split_rollout_samples(self, rollout_samples):
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]

        samples_list: List[Experience] = []
        nsp = int(getattr(self.args, "n_samples_per_prompt", 1) or 1)
        marl_alg = _canon_marl(getattr(self.args, "marl_algorithm", "auto"))

        want_qid_atomic = (marl_alg == "mappo") or ((nsp > 1) and (marl_alg in {"c3", "auto", "magrpo"}))
        grouped = self._group_by_question_id(rollout_samples) if want_qid_atomic else None
        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)

        def _total_len(s) -> int:
            info = getattr(s, "info", None)
            info = info if isinstance(info, dict) else {}
            for k in ("total_length", "tot_len", "total_len"):
                if k in info and info[k] is not None:
                    return max(1, int(_as_len_scalar(info[k])))
            seq = getattr(s, "sequences", None)
            if isinstance(seq, torch.Tensor) and seq.numel():
                return max(1, int(seq.shape[-1]))
            am = getattr(s, "attention_mask", None)
            if isinstance(am, torch.Tensor) and am.numel():
                return max(1, int(am.sum().item()))
            return 1

        def _concat(micro: List[Experience]) -> Experience:
            return Experience.concat_experiences(micro, pad_id)

        if self.args.use_dynamic_batch:
            len_map = {id(s): _total_len(s) for s in rollout_samples}
            effective_actor_num = (
                self.args.actor_num_nodes
                * self.args.actor_num_gpus_per_node
                // self.args.ring_attn_size
                // self.args.ds_tensor_parallel_size
            )

            def _make_num_batches(lengths: List[int]) -> int:
                nb = get_minimum_num_micro_batch_size(
                    lengths,
                    self.args.rollout_max_tokens_per_gpu,
                    self.args.ring_attn_size,
                    self.args.ds_tensor_parallel_size,
                )
                nb = nb // effective_actor_num * effective_actor_num
                return max(nb, effective_actor_num)

            if grouped is None:
                lens = [len_map[id(s)] for s in rollout_samples]
                num_batch = _make_num_batches(lens)
                batch_indexes = get_seqlen_balanced_partitions(lens, num_batch, False)
                for micro_index in batch_indexes:
                    micro = [rollout_samples[idx] for idx in micro_index]
                    samples_list.append(_concat(micro))
            else:
                group_lengths = [max(1, sum(len_map.get(id(s), _total_len(s)) for s in g)) for g in grouped]
                num_batch = _make_num_batches(group_lengths)
                group_batch_indexes = get_seqlen_balanced_partitions(group_lengths, num_batch, False)
                for gidxs in group_batch_indexes:
                    micro: List[Experience] = []
                    for gi in gidxs:
                        micro.extend(grouped[gi])
                    samples_list.append(_concat(micro))
        else:
            batch_size = int(self.args.micro_rollout_batch_size)

            if grouped is None:
                for i in range(0, len(rollout_samples), batch_size):
                    samples_list.append(_concat(rollout_samples[i : i + batch_size]))
            else:
                cur: List[Experience] = []
                cur_n = 0

                def _flush():
                    nonlocal cur, cur_n
                    if cur:
                        samples_list.append(_concat(cur))
                    cur, cur_n = [], 0

                for g in grouped:
                    g_n = len(g)
                    if g_n >= batch_size:
                        _flush()
                        samples_list.append(_concat(g))
                        continue
                    if cur and cur_n + g_n > batch_size:
                        _flush()
                    cur.extend(g)
                    cur_n += g_n

                _flush()

        return samples_list

    # -------------------------------------------------------------------------
    # Main entry
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def make_experience_batch(self, rollout_samples, *, compute_advantages: bool = True) -> List[Experience]:
        samples_list = self.split_rollout_samples(rollout_samples)
        experiences = self.make_experience(samples_list)
        if compute_advantages:
            experiences = self.compute_advantages_and_returns(experiences)
        for e in experiences:
            self._prune_rollout_text_fields_inplace(e)
        return experiences

    # -------------------------------------------------------------------------
    # Experience creation: rewards, logprobs, values, KL
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def make_experience(self, samples_list: List[Experience]) -> List[Experience]:
        t0 = time.time()

        for s in samples_list:
            if not isinstance(s.sequences, torch.Tensor):
                raise RuntimeError("make_experience expects sequences to be a torch.Tensor (warmup cache must be FULL).")
            if not isinstance(s.attention_mask, torch.Tensor):
                raise RuntimeError("make_experience expects attention_mask to be a torch.Tensor.")
            if not isinstance(s.action_mask, torch.Tensor):
                raise RuntimeError("make_experience expects action_mask to be a torch.Tensor.")
            _ensure_info_dict(s)

        total_samples = sum(int(s.sequences.shape[0]) for s in samples_list)
        logger.info(f"🚀 Starting experience making with {total_samples} samples")

        args = self.args
        device = torch.device("cpu")

        expected_len = len(samples_list)
        duplicate_factor = max(1, int(getattr(self.actor_model_group, "duplicate_actors", 1) or 1))

        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]

        # ---- Rewards ----
        r_refs = None
        if samples_list and samples_list[0].rewards is not None:
            for s in samples_list:
                if s.rewards is not None:
                    info = _ensure_info_dict(s)
                    info.setdefault("reward", s.rewards)
        elif self.remote_rm_url:
            if self.remote_reward_model is None:
                raise RuntimeError("remote_rm_url is set but remote_reward_model is None.")
            queries_list = sum(
                [
                    self.tokenizer.batch_decode(remove_pad_token(seq, am), skip_special_tokens=False)
                    for seq, am in zip(sequences_list, attention_mask_list)
                ],
                [],
            )
            prompts_list = sum([s.prompts for s in samples_list], [])
            labels_list = sum([s.labels for s in samples_list], [])
            r_refs = self.remote_reward_model.get_rewards.remote(queries_list, prompts_list, labels_list)
        else:
            if self.reward_model_group is None:
                raise RuntimeError(
                    "reward_model_group is None but rewards are not prefilled and remote_rm_url is not set. "
                    "Enable env rewards, set --remote_rm_url, or provide a reward model."
                )
            r_refs = self.reward_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                attention_mask=attention_mask_list,
                pad_sequence=[True] * len(samples_list),
            )

        if (
            args.colocate_all_models
            and (not self.remote_rm_url)
            and (r_refs is not None)
            and (self.reward_model_group is not None)
        ):
            ray.get(r_refs)
            ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

        # ---- Actor old-logprobs ----
        policy_loss_type = str(getattr(args, "policy_loss_type", "ppo") or "ppo").lower().strip()
        enable_is_corr = bool(getattr(args, "enable_vllm_is_correction", False))
        try:
            init_kl = float(getattr(args, "init_kl_coef", 0.0) or 0.0)
        except Exception:
            init_kl = 0.0

        need_old_for_kl_shaping = (self.initial_model_group is not None) and (not args.use_kl_loss) and (init_kl > 0.0)
        skip_old_logprobs = (policy_loss_type == "reinforce") and (not enable_is_corr) and (not need_old_for_kl_shaping)

        if skip_old_logprobs:
            action_log_probs_ref = ray.put([[None]] * (len(samples_list) * duplicate_factor))
        else:
            action_log_probs_ref = self.actor_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(action_log_probs_ref)
                ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))

        # ---- Critic values ----
        use_mappo_state_critic = self._should_use_mappo_state_critic(samples_list)
        critic_sequences_list = sequences_list
        critic_attention_mask_list = attention_mask_list
        critic_action_mask_list = action_mask_list
        if use_mappo_state_critic:
            critic_sequences_list, critic_attention_mask_list, critic_action_mask_list = self._build_mappo_state_critic_inputs(
                samples_list
            )

        if self.critic_model_group is not None:
            if (
                args.colocate_critic_reward
                and (not self.remote_rm_url)
                and (r_refs is not None)
                and (self.reward_model_group is not None)
            ):
                ray.get(r_refs)
                ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

            value_ref = self.critic_model_group.async_run_method_batch(
                method_name="forward",
                sequences=critic_sequences_list,
                action_mask=critic_action_mask_list,
                attention_mask=critic_attention_mask_list,
                pad_sequence=[True] * len(critic_sequences_list),
            )
            if args.colocate_all_models or args.colocate_critic_reward:
                ray.get(value_ref)
                ray.get(self.critic_model_group.async_run_method(method_name="empty_cache"))
        else:
            value_ref = ray.put([[None]] * (len(samples_list) * duplicate_factor))

        # ---- Base model logprobs ----
        if self.initial_model_group is not None:
            base_action_log_probs_ref = self.initial_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(base_action_log_probs_ref)
                ray.get(self.initial_model_group.async_run_method(method_name="empty_cache"))
        else:
            base_action_log_probs_ref = ray.put([[None]] * (len(samples_list) * duplicate_factor))

        # ---- Gather ----
        action_log_probs_list = self._flatten_ray_batch_outputs(action_log_probs_ref, duplicate_factor)[:expected_len]
        base_action_log_probs_list = self._flatten_ray_batch_outputs(base_action_log_probs_ref, duplicate_factor)[:expected_len]
        value_list = self._flatten_ray_batch_outputs(value_ref, duplicate_factor)[:expected_len]

        # ---- Materialize rewards ----
        if samples_list and samples_list[0].rewards is not None:
            pass
        elif self.remote_rm_url:
            if r_refs is None:
                raise RuntimeError("remote RM enabled but reward refs are None")
            rewards_info = ray.get(r_refs)
            update_samples_with_rewards(rewards_info, samples_list)
        else:
            if r_refs is None:
                raise RuntimeError("local reward model enabled but reward refs are None")
            rewards_list = self._flatten_ray_batch_outputs(r_refs, duplicate_factor)[:expected_len]
            for i, s in enumerate(samples_list):
                s.rewards = rewards_list[i]
                _ensure_info_dict(s)["reward"] = rewards_list[i]

        if not (len(samples_list) == len(action_log_probs_list) == len(base_action_log_probs_list) == len(value_list)):
            raise RuntimeError(
                f"Batch size mismatch: samples={len(samples_list)} "
                f"actor={len(action_log_probs_list)} base={len(base_action_log_probs_list)} critic={len(value_list)}"
            )

        # ---- Stitch ----
        for s, action_lp, base_lp, value in zip(samples_list, action_log_probs_list, base_action_log_probs_list, value_list):
            if action_lp is None:
                kl = torch.zeros_like(s.action_mask, dtype=torch.float32, device=device)
            else:
                if (self.initial_model_group is not None) and (not args.use_kl_loss) and (base_lp is not None):
                    kl = compute_approx_kl(action_lp, base_lp, kl_estimator=args.kl_estimator)
                else:
                    kl = torch.zeros_like(action_lp, dtype=action_lp.dtype, device=device)

            _ensure_info_dict(s)["kl"] = masked_mean(kl, s.action_mask, dim=-1)

            if not args.use_kl_loss:
                base_lp = None

            s.action_log_probs = action_lp
            s.base_action_log_probs = base_lp
            s.kl = kl

            if use_mappo_state_critic:
                if not isinstance(value, torch.Tensor) or value.dim() != 2 or value.shape[1] != 1:
                    raise RuntimeError(f"MAPPO expects critic value [B,1]; got {getattr(value, 'shape', None)}")
                s.critic_values = value
                v_scalar = value[:, 0].to(torch.float32)
                s.values = (v_scalar.unsqueeze(1) * s.action_mask.to(v_scalar.dtype)).contiguous()
            else:
                s.values = value

        logger.info(f"✨ Experience making completed in {str(timedelta(seconds=(time.time() - t0))).split('.')[0]}")
        return samples_list

    # -------------------------------------------------------------------------
    # MARL detection & info extraction
    # -------------------------------------------------------------------------
    def _is_marl_enabled_batch(self, experiences: List[Experience]) -> bool:
        marl_alg = _canon_marl(getattr(self.args, "marl_algorithm", "auto"))
        if marl_alg == "none":
            return False

        for exp in experiences:
            info = exp.info
            if not isinstance(info, dict):
                continue

            if "marl_enabled" in info:
                try:
                    v = info["marl_enabled"]
                    if isinstance(v, torch.Tensor):
                        vv = v.view(-1)
                        if vv.numel() and int(vv[0].item()) == 1:
                            return True
                    else:
                        if int(v[0] if isinstance(v, (list, tuple)) else v) == 1:
                            return True
                except Exception:
                    return True

            if "question_id" in info:
                return True

        return False

    def _extract_info_long_1d(self, exp: Experience, key: str, *, fallback: torch.Tensor) -> torch.Tensor:
        info = _ensure_info_dict(exp)
        if key not in info:
            if _looks_like_mas(info):
                raise RuntimeError(f"[MAS][FAIL-FAST] Missing info[{key!r}] during experience making.")
            return fallback

        x = info[key]
        try:
            t = x.to(torch.long).view(-1) if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.long).view(-1)
            if int(t.numel()) != int(fallback.numel()):
                if _looks_like_mas(info):
                    raise RuntimeError(f"[MAS][FAIL-FAST] Invalid {key} shape: batch={fallback.numel()} numel={t.numel()}.")
                return fallback
            return t
        except Exception:
            if _looks_like_mas(info):
                raise RuntimeError(f"[MAS][FAIL-FAST] Invalid {key} type/value: {type(x)}.")
            return fallback

    def _extract_info_long_1d_optional(self, exp: Experience, key: str, *, B: int, fallback: torch.Tensor) -> torch.Tensor:
        info = _ensure_info_dict(exp)
        if key not in info:
            return fallback
        return _as_long_1d(info.get(key, None), B=B, fallback=fallback)

    def _extract_role_ids_1d(self, exp: Experience, *, B: int, fallback: torch.Tensor) -> torch.Tensor:
        info = exp.info if isinstance(exp.info, dict) else {}
        if "role_id" in info:
            return _as_long_1d(info["role_id"], B=B, fallback=fallback)

        topo = _parse_roles_topo(info.get("roles_topo", None))
        if not topo:
            return fallback

        role_id_map = {name: i for i, name in enumerate(topo)}
        role_names = info.get("role", None)

        if isinstance(role_names, list) and len(role_names) == B:
            return torch.tensor([role_id_map.get(str(rn), 0) for rn in role_names], dtype=torch.long)
        if isinstance(role_names, list) and len(role_names) == 1 and B > 1:
            rid = role_id_map.get(str(role_names[0]), 0)
            return torch.full((B,), rid, dtype=torch.long)

        return fallback

    # -------------------------------------------------------------------------
    # MARL advantages/returns (C3/MAGRPO/MAPPO)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _compute_marl_advantages_and_returns(self, experiences: List[Experience]) -> List[Experience]:
        from c3.algorithms.magrpo import compute_magrpo
        from c3.algorithms.mappo import compute_mappo_step_gae
        from c3.algorithms.utils import group_mean_std

        args = self.args
        marl_alg = _canon_marl(getattr(args, "marl_algorithm", "auto"))

        if marl_alg in {"none", "auto"}:
            if int(getattr(args, "n_samples_per_prompt", 1) or 1) > 1:
                marl_alg = "magrpo"
            elif self.critic_model_group is not None:
                marl_alg = "mappo"
            else:
                marl_alg = "magrpo"

        normalize_adv = not getattr(args, "no_advantage_std_norm", False)
        nsp = int(getattr(args, "n_samples_per_prompt", 1) or 1)
        gamma = float(getattr(args, "gamma", 1.0))
        lambd = float(getattr(args, "lambd", 1.0))

        if marl_alg == "mappo" and nsp != 1:
            raise RuntimeError(
                "[MAPPO][FAIL-FAST] n_samples_per_prompt must be 1 for MAS/C3 tasks. "
                "MAPPO step-GAE requires per-episode step sequences."
            )

        total_rows = sum(int(e.rewards.shape[0]) for e in experiences if e.rewards is not None)
        fallback_qid_all = torch.arange(total_rows, dtype=torch.long) // max(1, nsp)
        cursor = 0

        exp_sizes: List[int] = []
        token_rewards_list: List[torch.Tensor] = []
        adv_group_ids_list: List[torch.Tensor] = []

        # MAGRPO caches
        magrpo_group_ids_list: List[torch.Tensor] = []
        magrpo_qids_list: List[torch.Tensor] = []
        magrpo_traj_ids_list: List[torch.Tensor] = []
        magrpo_leaf_list: List[torch.Tensor] = []
        magrpo_traj_from_fallback_any = False
        magrpo_has_kid_neg_any = False

        # MAPPO step arrays
        mappo_qids_list: List[torch.Tensor] = []
        mappo_episode_ids_list: List[torch.Tensor] = []
        mappo_step_ids_list: List[torch.Tensor] = []
        mappo_role_ids_list: List[torch.Tensor] = []
        mappo_terminals_list: List[torch.Tensor] = []
        mappo_values_list: List[torch.Tensor] = []
        mappo_rewards_list: List[torch.Tensor] = []

        magrpo_adv_unit = str(getattr(args, "magrpo_adv_unit", "joint_action") or "joint_action").lower().strip()
        if magrpo_adv_unit not in {"joint_action", "per_role"}:
            magrpo_adv_unit = "joint_action"

        def _mappo_episode_step(
            info: Dict[str, Any],
            *,
            B: int,
            kids: torch.Tensor,
            role_ids: torch.Tensor,
            traj_ids: torch.Tensor,
            raw_traj_present: bool,
            device: torch.device,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            role_ids_l = torch.clamp(role_ids.to(torch.long), min=0).to(device)

            # episode_id: per-trajectory
            if not raw_traj_present and torch.any(kids < 0):
                bad = (kids < 0).nonzero(as_tuple=False).view(-1)[:8].tolist()
                raise RuntimeError(
                    "[MAPPO][FAIL-FAST] traj_id is required when k_id<0 exists (cannot safely derive per-trajectory id). "
                    f"bad_rows={bad}"
                )
            episode_ids = traj_ids.to(torch.long).to(device)

            # step_id: step_id > c3_depth > role_id
            if "step_id" in info:
                step_ids = _as_long_1d(info.get("step_id", None), B=B, fallback=torch.zeros((B,), dtype=torch.long))
            elif "c3_depth" in info:
                step_ids = _as_long_1d(info.get("c3_depth", None), B=B, fallback=role_ids_l.to("cpu"))
            else:
                step_ids = role_ids_l.to("cpu")
            step_ids = torch.clamp(step_ids.to(torch.long), min=0).to(device)

            # terminals: last step per episode (in-batch)
            order = torch.argsort(step_ids, stable=True)
            order = order[torch.argsort(episode_ids[order], stable=True)]
            ep_s = episode_ids[order]
            _, counts = torch.unique_consecutive(ep_s, return_counts=True)

            terminals = torch.zeros((B,), dtype=torch.long, device=device)
            cur = 0
            for c in counts.tolist():
                if c > 0:
                    terminals[order[cur + c - 1]] = 1
                cur += max(0, c)

            info["episode_id"] = episode_ids.detach().to("cpu")
            info["step_id"] = step_ids.detach().to("cpu")
            info["mappo_terminals"] = terminals.detach().to("cpu")

            return episode_ids, step_ids, terminals, role_ids_l

        # ---- Pass 1: ids + token rewards ----
        for exp in experiences:
            info = _ensure_info_dict(exp)
            if exp.rewards is None:
                raise RuntimeError("MARL enabled but experience.rewards is None.")
            if exp.kl is None:
                raise RuntimeError("experience.kl is None before advantage computation.")

            B = int(exp.rewards.shape[0])
            exp_sizes.append(B)

            fallback_qid = fallback_qid_all[cursor : cursor + B]
            cursor += B

            qids = self._extract_info_long_1d(exp, "question_id", fallback=fallback_qid)
            kids = self._extract_info_long_1d(exp, "k_id", fallback=torch.zeros_like(fallback_qid))
            role_ids = self._extract_role_ids_1d(exp, B=B, fallback=torch.zeros_like(fallback_qid))
            is_leaf = self._extract_info_long_1d(exp, "is_leaf", fallback=(kids >= 0).to(torch.long))

            topo = _parse_roles_topo(info.get("roles_topo", None))
            if "num_roles" in info:
                num_roles_1d = _as_long_1d(info["num_roles"], B=B, fallback=torch.ones_like(fallback_qid))
            else:
                num_roles_1d = torch.full((B,), int(len(topo) if topo else 1), dtype=torch.long)

            nr_mod = max(1, int(num_roles_1d.max().item()) if num_roles_1d.numel() else 1)

            adv_gid = _as_long_1d(info.get("adv_group_id", None), B=B, fallback=(qids * nr_mod + role_ids))
            magrpo_gid_per_role = (qids * nr_mod + role_ids).to(torch.long)
            magrpo_gid = magrpo_gid_per_role if (marl_alg == "magrpo" and magrpo_adv_unit == "per_role") else qids.to(torch.long)

            raw_traj = info.get("traj_id", None)
            traj_fallback = (qids * max(1, nsp) + kids)
            traj_ids = _as_long_1d(raw_traj, B=B, fallback=traj_fallback)

            raw_traj_present = raw_traj is not None
            if not raw_traj_present:
                magrpo_traj_from_fallback_any = True
            if torch.any(kids < 0):
                magrpo_has_kid_neg_any = True

            info["question_id"] = qids.to("cpu")
            info["k_id"] = kids.to("cpu")
            info["role_id"] = role_ids.to("cpu")
            info["num_roles"] = num_roles_1d.to("cpu")
            info["traj_id"] = traj_ids.to("cpu")
            info["adv_group_id"] = adv_gid.to("cpu")
            info["magrpo_group_id"] = magrpo_gid.to("cpu")
            info["magrpo_group_id_per_role"] = magrpo_gid_per_role.to("cpu")
            info["is_leaf"] = is_leaf.to("cpu")

            # MAPPO: only leaf gets env reward; others get 0
            r_env = exp.rewards.to(torch.float32) * is_leaf.to(torch.float32) if marl_alg == "mappo" else exp.rewards
            tok_r = compute_reward(
                r_env,
                self.kl_ctl.value,
                exp.kl,
                action_mask=exp.action_mask,
                reward_clip_range=args.reward_clip_range,
            )

            token_rewards_list.append(tok_r)
            adv_group_ids_list.append(adv_gid.to(tok_r.device))

            if marl_alg == "magrpo":
                magrpo_group_ids_list.append(magrpo_gid.to(tok_r.device))
                magrpo_qids_list.append(qids.to(tok_r.device).to(torch.long))
                magrpo_traj_ids_list.append(traj_ids.to(tok_r.device).to(torch.long))
                magrpo_leaf_list.append(is_leaf.to(tok_r.device).to(torch.long))

            if marl_alg == "mappo":
                episode_ids, step_ids, terminals, role_ids_l = _mappo_episode_step(
                    info,
                    B=B,
                    kids=kids,
                    role_ids=role_ids,
                    traj_ids=traj_ids,
                    raw_traj_present=raw_traj_present,
                    device=tok_r.device,
                )

                if (
                    isinstance(exp.critic_values, torch.Tensor)
                    and exp.critic_values.dim() == 2
                    and exp.critic_values.shape[1] == 1
                ):
                    v_step = exp.critic_values[:, 0].to(tok_r.device).to(torch.float32)

                elif isinstance(exp.values, torch.Tensor):
                    v_tok = exp.values.to(tok_r.device)

                    if v_tok.dim() == 1:
                        v_step = v_tok.to(torch.float32)

                    elif v_tok.dim() == 2 and v_tok.shape[1] >= 1:
                        if v_tok.shape[1] == 1:
                            v_step = v_tok[:, 0].to(torch.float32)
                        else:
                            tail = v_tok[:, 1:]
                            pooled_mask = (tail.abs().amax(dim=1) < 1e-6)
                            v_mean = masked_mean(v_tok, exp.action_mask.to(tok_r.device), dim=-1).to(torch.float32)
                            v_step = torch.where(pooled_mask, v_tok[:, 0].to(torch.float32), v_mean)

                    else:
                        v_step = masked_mean(v_tok, exp.action_mask.to(tok_r.device), dim=-1).to(torch.float32)

                else:
                    v_step = None

                if v_step is None:
                    raise RuntimeError("[MAPPO][FAIL-FAST] Missing critic values for MAPPO step-GAE.")

                r_step = tok_r.sum(dim=1).to(torch.float32)

                mappo_qids_list.append(qids.to(tok_r.device).to(torch.long))
                mappo_episode_ids_list.append(episode_ids.to(torch.long))
                mappo_step_ids_list.append(step_ids.to(torch.long))
                mappo_role_ids_list.append(role_ids_l.to(tok_r.device).to(torch.long))
                mappo_terminals_list.append(terminals.to(tok_r.device).to(torch.long))
                mappo_values_list.append(v_step)
                mappo_rewards_list.append(r_step)

        # ---- Group reward std (logging) ----
        if nsp > 1:
            scalar_returns = torch.cat([tr.sum(dim=1) for tr in token_rewards_list], dim=0).to(torch.float32)
            gid_all = (
                torch.cat(magrpo_group_ids_list, dim=0).to(torch.long).to(scalar_returns.device)
                if marl_alg == "magrpo" and magrpo_group_ids_list
                else torch.cat(adv_group_ids_list, dim=0).to(torch.long).to(scalar_returns.device)
            )
            _, std_per_row = group_mean_std(scalar_returns, gid_all)
            std_slices = std_per_row.detach().to("cpu").split(exp_sizes)
            for exp, std_slice in zip(experiences, std_slices):
                _ensure_info_dict(exp)["group_reward_std"] = std_slice

        # ---- C3 credit assignment ----
        c3_credit_slices = None
        c3_diag = None
        c3_credit_tag = None
        if marl_alg == "c3":
            adv_gid_all = torch.cat(adv_group_ids_list, dim=0).to(torch.long)
            c3_credit_slices, c3_diag, c3_credit_tag = self._compute_c3_credit(
                experiences=experiences,
                adv_group_ids_all=adv_gid_all,
                token_rewards_list=token_rewards_list,
                normalize_adv=normalize_adv,
            )

        # ---- MAPPO step-GAE ----
        mappo_adv_all = None
        mappo_ret_all = None
        if marl_alg == "mappo":
            ep_all = torch.cat(mappo_episode_ids_list, dim=0)
            st_all = torch.cat(mappo_step_ids_list, dim=0)
            rl_all = torch.cat(mappo_role_ids_list, dim=0)
            tm_all = torch.cat(mappo_terminals_list, dim=0)
            v_all = torch.cat(mappo_values_list, dim=0)
            r_all = torch.cat(mappo_rewards_list, dim=0)

            adv_all, ret_all = compute_mappo_step_gae(
                rewards=r_all,
                values=v_all,
                terminals=tm_all,
                episode_ids=ep_all,
                step_ids=st_all,
                gamma=gamma,
                lambd=lambd,
            )

            if normalize_adv:
                scope = str(getattr(args, "mappo_normalize_scope", "global") or "global").lower().strip()
                if scope == "global":
                    mean = adv_all.mean()
                    std = adv_all.std(unbiased=False).clamp_min(1e-8)
                    adv_all = (adv_all - mean) / std
                else:
                    q_all = torch.cat(mappo_qids_list, dim=0)
                    nr_mod = int(
                        torch.max(
                            torch.cat(
                                [
                                    e.info["num_roles"].view(-1)
                                    for e in experiences
                                    if isinstance(e.info, dict) and "num_roles" in e.info
                                ],
                                dim=0,
                            )
                        ).item()
                    )
                    nr_mod = max(1, nr_mod)

                    if scope == "group":
                        norm_ids = q_all
                    elif scope == "episode":
                        norm_ids = ep_all
                    elif scope == "group_role":
                        norm_ids = (q_all * nr_mod + rl_all).to(torch.long)
                    elif scope == "episode_role":
                        norm_ids = (ep_all * nr_mod + rl_all).to(torch.long)
                    else:
                        norm_ids = None

                    if norm_ids is None:
                        mean = adv_all.mean()
                        std = adv_all.std(unbiased=False).clamp_min(1e-8)
                        adv_all = (adv_all - mean) / std
                    else:
                        m, s = group_mean_std(adv_all, norm_ids.to(torch.long).to(adv_all.device))
                        adv_all = (adv_all - m) / s.clamp_min(1e-8)

            mappo_adv_all, mappo_ret_all = adv_all, ret_all

        # ---- MAGRPO joint-action advantage ----
        magrpo_joint_adv_rows_all = None
        magrpo_joint_std_rows_all = None
        if marl_alg == "magrpo" and magrpo_adv_unit != "per_role":
            if magrpo_traj_from_fallback_any and magrpo_has_kid_neg_any:
                raise RuntimeError(
                    "[MAGRPO][FAIL-FAST] joint_action advantage requires traj_id for kid<0 rows. "
                    "Provide info['traj_id'] (unique per trajectory), or avoid non-leaf rows."
                )

            q_all = torch.cat(magrpo_qids_list, dim=0).to(torch.long)
            traj_all = torch.cat(magrpo_traj_ids_list, dim=0).to(torch.long)
            leaf_all = torch.cat(magrpo_leaf_list, dim=0).to(torch.long) > 0
            scalar_all = torch.cat([tr.sum(dim=1) for tr in token_rewards_list], dim=0).to(torch.float32)

            uniq_traj, inv_traj = torch.unique(traj_all, sorted=True, return_inverse=True)
            G = int(uniq_traj.numel())
            dev = scalar_all.device

            sums_all = torch.zeros((G,), dtype=torch.float32, device=dev)
            cnts_all = torch.zeros((G,), dtype=torch.float32, device=dev)
            sums_all.scatter_add_(0, inv_traj, scalar_all)
            cnts_all.scatter_add_(0, inv_traj, torch.ones_like(scalar_all, dtype=torch.float32))

            if leaf_all.any():
                inv_leaf = inv_traj[leaf_all]
                sums_leaf = torch.zeros((G,), dtype=torch.float32, device=dev)
                cnts_leaf = torch.zeros((G,), dtype=torch.float32, device=dev)
                sums_leaf.scatter_add_(0, inv_leaf, scalar_all[leaf_all])
                cnts_leaf.scatter_add_(0, inv_leaf, torch.ones_like(scalar_all[leaf_all], dtype=torch.float32))
                r_traj = torch.where(
                    cnts_leaf > 0,
                    sums_leaf / cnts_leaf.clamp_min(1.0),
                    sums_all / cnts_all.clamp_min(1.0),
                )

                q_sums_leaf = torch.zeros((G,), dtype=torch.float32, device=dev)
                q_cnts_leaf = torch.zeros((G,), dtype=torch.float32, device=dev)
                q_sums_leaf.scatter_add_(0, inv_leaf, q_all[leaf_all].to(torch.float32))
                q_cnts_leaf.scatter_add_(0, inv_leaf, torch.ones_like(q_all[leaf_all], dtype=torch.float32))

                q_sums_all = torch.zeros((G,), dtype=torch.float32, device=dev)
                q_sums_all.scatter_add_(0, inv_traj, q_all.to(torch.float32))
                q_traj = torch.where(
                    q_cnts_leaf > 0,
                    (q_sums_leaf / q_cnts_leaf.clamp_min(1.0)).round(),
                    (q_sums_all / cnts_all.clamp_min(1.0)).round(),
                ).to(torch.long)
            else:
                r_traj = sums_all / cnts_all.clamp_min(1.0)
                q_sums_all = torch.zeros((G,), dtype=torch.float32, device=dev)
                q_sums_all.scatter_add_(0, inv_traj, q_all.to(torch.float32))
                q_traj = (q_sums_all / cnts_all.clamp_min(1.0)).round().to(torch.long)

            uniq_q, inv_q = torch.unique(q_traj, sorted=True, return_inverse=True)
            Q = int(uniq_q.numel())
            q_sum = torch.zeros((Q,), dtype=torch.float32, device=dev)
            q_cnt = torch.zeros((Q,), dtype=torch.float32, device=dev)
            q_sum.scatter_add_(0, inv_q, r_traj)
            q_cnt.scatter_add_(0, inv_q, torch.ones_like(r_traj, dtype=torch.float32))

            baseline = str(getattr(args, "magrpo_baseline", "group_mean") or "group_mean").lower().strip()
            if baseline not in {"group_mean", "rloo"}:
                baseline = "group_mean"

            mean_q = q_sum[inv_q] / q_cnt[inv_q].clamp_min(1.0)
            if baseline == "rloo":
                denom = (q_cnt[inv_q] - 1.0).clamp_min(1.0)
                mean_q = (q_sum[inv_q] - r_traj) / denom

            diff = r_traj - (q_sum[inv_q] / q_cnt[inv_q].clamp_min(1.0))
            q_var = torch.zeros((Q,), dtype=torch.float32, device=dev)
            q_var.scatter_add_(0, inv_q, diff * diff)
            std_q = torch.sqrt(q_var[inv_q] / q_cnt[inv_q].clamp_min(1.0) + 1e-8)

            adv_traj = r_traj - mean_q
            if normalize_adv:
                adv_traj = adv_traj / std_q.clamp_min(1e-8)

            magrpo_joint_adv_rows_all = adv_traj[inv_traj]
            magrpo_joint_std_rows_all = std_q[inv_traj]

            std_rows = magrpo_joint_std_rows_all.detach().to("cpu").split(exp_sizes)
            for exp, std_slice in zip(experiences, std_rows):
                _ensure_info_dict(exp)["group_reward_std"] = std_slice

        # ---- Pass 2: write advantages/returns + logs ----
        m_cursor = 0
        magrpo_cursor = 0

        for i, (exp, tok_r) in enumerate(zip(experiences, token_rewards_list)):
            B = int(tok_r.shape[0])
            am = exp.action_mask.to(tok_r.dtype)
            info = _ensure_info_dict(exp)

            if marl_alg == "magrpo":
                if magrpo_adv_unit != "per_role":
                    if magrpo_joint_adv_rows_all is None:
                        raise RuntimeError("[MAGRPO][BUG] joint-action advantages are missing.")
                    adv_row = magrpo_joint_adv_rows_all[magrpo_cursor : magrpo_cursor + B].to(tok_r.device).to(tok_r.dtype)
                    exp.advantages = (adv_row.unsqueeze(1) * am).detach()
                    exp.returns = (tok_r * am).detach()
                    if magrpo_joint_std_rows_all is not None:
                        info["group_reward_std"] = magrpo_joint_std_rows_all[magrpo_cursor : magrpo_cursor + B].detach().to("cpu")
                    magrpo_cursor += B
                else:
                    gid_1d = _as_long_1d(
                        info.get("magrpo_group_id_per_role", None), B=B, fallback=torch.arange(B, dtype=torch.long)
                    )
                    baseline = str(getattr(args, "magrpo_baseline", "group_mean") or "group_mean").strip().lower()
                    token_norm = bool(getattr(args, "magrpo_token_normalize", False))

                    adv, ret = compute_magrpo(
                        action_mask=exp.action_mask,
                        rewards=tok_r,
                        values=None,
                        group_ids=gid_1d.to(tok_r.device),
                        baseline=baseline,
                        gamma=gamma,
                        lambd=lambd,
                        normalize_adv=normalize_adv,
                        require_group_k=(baseline == "rloo"),
                        token_normalize=token_norm,
                    )
                    exp.advantages = adv.detach()
                    exp.returns = ret.detach()

            elif marl_alg == "mappo":
                if mappo_adv_all is None or mappo_ret_all is None:
                    raise RuntimeError("[MAPPO][BUG] step-GAE outputs are missing.")

                adv_step = mappo_adv_all[m_cursor : m_cursor + B].to(tok_r.device).to(tok_r.dtype)
                ret_step = mappo_ret_all[m_cursor : m_cursor + B].to(tok_r.device).to(tok_r.dtype)
                m_cursor += B

                token_norm = bool(getattr(args, "mappo_token_normalize", True))
                if token_norm:
                    tok_cnt = exp.action_mask.to(tok_r.device).sum(dim=1)
                    if torch.any(tok_cnt <= 0):
                        bad = (tok_cnt <= 0).nonzero(as_tuple=False).view(-1)[:8].tolist()
                        raise RuntimeError(f"[MAPPO][FAIL-FAST] empty action_mask rows={bad}")
                    tok_cnt_f = tok_cnt.to(tok_r.dtype).clamp_min(1.0)
                    exp.advantages = ((adv_step / tok_cnt_f).unsqueeze(1) * am).detach()
                    info["mappo_action_tokens"] = tok_cnt.detach().to("cpu")
                else:
                    exp.advantages = (adv_step.unsqueeze(1) * am).detach()

                exp.returns = (ret_step.unsqueeze(1) * am).detach()

                if isinstance(exp.critic_action_mask, torch.Tensor):
                    cam = exp.critic_action_mask.to(tok_r.device).to(tok_r.dtype)
                    exp.critic_returns = (cam * ret_step.unsqueeze(1)).detach()
                else:
                    exp.critic_returns = ret_step.unsqueeze(1).detach()

                info["mappo_step_adv"] = adv_step.detach().to("cpu")
                info["mappo_step_return"] = ret_step.detach().to("cpu")

            elif marl_alg == "c3":
                if c3_credit_slices is not None and i < len(c3_credit_slices):
                    credit = c3_credit_slices[i].to(tok_r.device).view(-1).float()
                    exp.advantages = (credit.unsqueeze(1) * am).detach()
                    info["c3_credit_scalar"] = credit.detach().cpu()

                    if exp.values is not None and not (isinstance(exp.values, list) and exp.values and exp.values[0] is None):
                        _, ret = self.get_advantages_and_returns(exp.values, tok_r, exp.action_mask, gamma, lambd)
                        exp.returns = ret.detach()
                    else:
                        exp.returns = (tok_r * am).detach()

                    info["c3_credit"] = [c3_credit_tag or "c3"] * B
                    if isinstance(c3_diag, dict) and c3_diag:
                        for k, v in c3_diag.items():
                            try:
                                info[k] = torch.full((B,), float(v))
                            except Exception:
                                pass
                else:
                    scalar_ret = tok_r.sum(dim=1)
                    exp.advantages = (scalar_ret.unsqueeze(1) * 0.0 * am).detach()
                    exp.returns = (tok_r * am).detach()
                    info["c3_credit"] = ["fallback"] * B
                    info["c3_credit_scalar"] = scalar_ret.detach().cpu()
            else:
                raise RuntimeError(f"Unknown marl_algorithm={marl_alg!r}")

            token_return = tok_r.sum(dim=-1).detach().to("cpu")
            info["token_return"] = token_return
            if marl_alg == "mappo" and "mappo_step_return" in info:
                info["return"] = info["mappo_step_return"]
            else:
                info["return"] = token_return

            info["marl_algorithm"] = [marl_alg] * B
            exp.kl = None

        return experiences

    # -------------------------------------------------------------------------
    # C3 credit scoring
    # -------------------------------------------------------------------------
    def _compute_c3_credit(
        self,
        *,
        experiences: List[Experience],
        adv_group_ids_all: torch.Tensor,
        token_rewards_list: List[torch.Tensor],
        normalize_adv: bool,
    ):
        args = self.args
        k_rollouts = int(getattr(args, "n_samples_per_prompt", 1) or 1)
        if k_rollouts <= 1:
            raise RuntimeError("C3 requires n_samples_per_prompt (K) > 1.")

        q_group = getattr(self, "q_critic_model_group", None)

        from c3.credit.c3.materialize import materialize_c3_tree_groups
        from c3.credit.c3.registry import build_credit_cfg_from_args, build_credit_provider
        from c3.integration.marl_specs import load_task
        from c3.algorithms.utils import group_mean_std

        task_path = getattr(args, "c3_task", None)
        task_spec = getattr(self, "_c3_task_spec_cache", None)
        task_spec_path = getattr(self, "_c3_task_spec_cache_path", None)
        if task_path and (task_spec is None or task_spec_path != task_path):
            task_spec = load_task(task_path)
            self._c3_task_spec_cache = task_spec
            self._c3_task_spec_cache_path = task_path

        topo_names = None
        rt_arg = getattr(args, "c3_roles_topo", None)
        if isinstance(rt_arg, (list, tuple)) and rt_arg:
            topo_names = [str(x).strip() for x in rt_arg if str(x).strip()]
        elif isinstance(rt_arg, str) and rt_arg.strip():
            topo_names = _parse_roles_topo(rt_arg)

        if not topo_names and experiences:
            topo_names = _parse_roles_topo(experiences[0].info.get("roles_topo", None))

        if task_spec is None or not topo_names:
            raise RuntimeError("[C3][FAIL-FAST] Missing task spec or roles_topo for C3.")

        by_name = {r.name: r for r in task_spec.roles}
        by_name_lower = {r.name.lower(): r for r in task_spec.roles}

        remapped = []
        for n in topo_names:
            if n in by_name:
                remapped.append(n)
                continue
            key = str(n).lower()
            if key in by_name_lower:
                remapped.append(by_name_lower[key].name)
                continue
            raise RuntimeError(f"[C3][FAIL-FAST] roles_topo contains unknown role {n!r}.")
        topo_names = remapped

        roles = [by_name[n] for n in topo_names]
        role_names = [r.name for r in roles]

        cfg = build_credit_cfg_from_args(args)
        credit_variant = str(cfg.get("credit_variant", "value_assisted") or "value_assisted").lower().strip()
        need_q = credit_variant in ("value_assisted", "value_only")
        credit_tag = f"c3/{credit_variant}"

        if need_q and q_group is None:
            raise RuntimeError(f"C3 credit_variant={credit_variant!r} requires q_critic_model_group, but it is None.")

        class _RayCriticScorer:
            def __init__(self, critic_group):
                self.critic_group = critic_group

            def score_texts(self, texts, max_len: int, forward_bs: int):
                ah = getattr(self.critic_group, "_actor_handlers", None) or getattr(self.critic_group, "actor_handlers", None)
                if not isinstance(ah, list) or not ah:
                    refs = self.critic_group.async_run_method("score_texts", texts=texts, max_len=max_len, forward_bs=forward_bs)
                    return refs[0]

                world = len(ah)
                texts = texts or []
                if world <= 1 or len(texts) < world:
                    refs = self.critic_group.async_run_method("score_texts", texts=texts, max_len=max_len, forward_bs=forward_bs)
                    return refs[0]

                base, rem = len(texts) // world, len(texts) % world
                chunks, start = [], 0
                for i in range(world):
                    sz = base + (1 if i < rem else 0)
                    chunks.append(texts[start : start + sz])
                    start += sz

                outs = ray.get([ah[i].score_texts.remote(texts=chunks[i], max_len=max_len, forward_bs=forward_bs) for i in range(world)])
                outs = [o if isinstance(o, torch.Tensor) else torch.tensor(o) for o in outs]
                outs = [o.to(torch.float32).cpu() for o in outs]
                return torch.cat(outs, dim=0)

        q_scorer = _RayCriticScorer(q_group) if need_q else None
        provider = build_credit_provider(
            marl_algorithm="c3",
            args=args,
            roles=tuple(roles),
            critic=q_scorer,
            critic_preamble_path=getattr(args, "critic_preamble_path", "") or "",
        )
        if provider is None:
            raise RuntimeError("build_credit_provider returned None for c3.")

        tree_groups, _, tree_diag = materialize_c3_tree_groups(experiences, roles=role_names)
        per_exp_scalar, diag = provider.compute(tree_groups, experiences=experiences, cfg=cfg)

        if normalize_adv:
            all_scalar = torch.cat(per_exp_scalar, dim=0).float()
            m2, s2 = group_mean_std(all_scalar, adv_group_ids_all.to(all_scalar.device))
            all_scalar = (all_scalar - m2) / s2.clamp_min(1e-8)
            per_exp_scalar = list(all_scalar.split([tr.shape[0] for tr in token_rewards_list]))

        if isinstance(tree_diag, dict) and isinstance(diag, dict):
            tree_diag.update(diag)
            diag = tree_diag

        return per_exp_scalar, diag, credit_tag

    # -------------------------------------------------------------------------
    # Advantages/returns (MARL or vanilla PPO)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def compute_advantages_and_returns(self, experiences: List[Experience], **kwargs):
        args = self.args

        if args.overlong_buffer_len is not None:
            assert args.generate_max_len >= args.overlong_buffer_len
            overlong_buffer_len = args.overlong_buffer_len
            expected_len = args.generate_max_len - overlong_buffer_len
            overlong_penalty_factor = args.overlong_penalty_factor

            for exp in experiences:
                response_lengths = exp.info["response_length"]
                for j in range(len(response_lengths)):
                    valid_len = response_lengths[j].item()
                    exceed = min(valid_len - expected_len, overlong_buffer_len)
                    if exceed > 0:
                        exp.rewards[j] += -exceed / overlong_buffer_len * overlong_penalty_factor

        if self._is_marl_enabled_batch(experiences):
            return self._compute_marl_advantages_and_returns(experiences)

        # ---- Vanilla PPO ----
        exp_len = [len(exp.index) for exp in experiences]
        indices = torch.tensor(sum([exp.index for exp in experiences], []))
        raw_rewards = torch.cat([exp.rewards for exp in experiences], dim=0)

        rewards = torch.empty_like(raw_rewards)
        rewards[indices] = raw_rewards
        rewards = rewards.reshape(-1, args.n_samples_per_prompt)

        if args.n_samples_per_prompt > 1:
            group_reward_stds = (
                rewards.std(-1, keepdim=True).repeat(1, args.n_samples_per_prompt).reshape(-1)[indices].split(exp_len)
            )
            for exp, grs in zip(experiences, group_reward_stds):
                _ensure_info_dict(exp)["group_reward_std"] = grs

        if args.advantage_estimator == "rloo":
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            rewards = rewards - rewards.mean(-1, keepdim=True)
        elif args.advantage_estimator == "group_norm":
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)

        rewards = rewards.reshape(-1)[indices].split(exp_len)

        for exp, reward in zip(experiences, rewards):
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                exp.kl,
                action_mask=exp.action_mask,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                exp.advantages, exp.returns = self.get_advantages_and_returns(exp.values, reward, exp.action_mask, args.gamma, args.lambd)
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                if args.gamma != 1.0 and self.advantage_estimator in ["rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                    logger.warning("gamma is set to 1.0 for rloo, reinforce_baseline, group_norm, dr_grpo")
                    args.gamma = 1.0
                exp.returns = self.get_cumulative_returns(reward, exp.action_mask, args.gamma)
                exp.advantages = deepcopy(exp.returns)
            else:
                raise RuntimeError(f"Unknown advantage_estimator {self.advantage_estimator}")

            info = _ensure_info_dict(exp)
            token_return = reward.sum(dim=-1).detach().to("cpu")
            info["token_return"] = token_return
            info["return"] = token_return
            exp.kl = None

        if self.args.advantage_estimator in ["gae", "reinforce", "reinforce_baseline"]:
            all_adv, all_mask = [], []
            for exp in experiences:
                all_adv.append(exp.advantages.flatten())
                all_mask.append(exp.action_mask.flatten())

            adv_vec = torch.cat(all_adv, dim=0).float()
            mask_vec = torch.cat(all_mask, dim=0)
            num_actions = mask_vec.sum()

            mean = (adv_vec * mask_vec).sum() / num_actions
            if not self.args.no_advantage_std_norm:
                var = ((adv_vec - mean).pow(2) * mask_vec).sum() / num_actions
                rstd = var.clamp(min=1e-8).rsqrt()
            else:
                rstd = 1

            for exp in experiences:
                exp.advantages = (exp.advantages - mean) * rstd

        return experiences

    # -------------------------------------------------------------------------
    # Token-level GAE / returns
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(self, rewards: torch.Tensor, action_mask: torch.Tensor, gamma: float) -> torch.Tensor:
        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        if action_mask is not None:
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns
