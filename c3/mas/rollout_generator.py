"""
C3 MAS (multi-role) rollout generator.

Core behavior
- Builds a prefix tree over roles_topo with per-role fanout.
- Computes rewards on leaf episodes, then assigns each node the mean reward of its leaf subtree.
- Emits one PPO Experience per node.

Memory behavior
- Heavy texts (prompt/state/output) are gated unless explicitly needed (dump/analysis/alg deps).
- ctx_hash is always computed (stable 63-bit hash of state_text; fallback to prompt).

No-replay (C3 ablation)
- When args.c3_no_replay is true, advantage grouping for non-root roles must not condition on
  identical parent transcript. We implement this by collapsing parent_id to -1 for depth>0 when
  encoding adv_group_id, and we also log both parent_id (collapsed) and parent_id_true.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from c3.integration.marl_specs import RoleSpec, TaskSpec, load_task
from c3.mas.prompt_render import build_render_context, render_role_prompt
from c3.mas.role_graph import RoleGraph
from c3.utils.budget_ledger import append_ledger, make_budget_record
from c3.utils.collision_guard import CollisionGuard
from c3.utils.context_key import fingerprint, hash63

logger = logging.getLogger(__name__)

_RE_SYSTEM_PREFIX = re.compile(r"^\s*system\s*[:\n]\s*", flags=re.IGNORECASE)

StrMap = Dict[str, str]


# =============================================================================
# Small utilities (cheap, stable, defensive)
# =============================================================================
def _safe_json_dict(x: Optional[object]) -> Optional[dict]:
    """Parse dict from JSON string; pass-through dict; else None."""
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_env_cfg(environment: object) -> dict:
    """Small, stable env cfg subset for reward meta."""
    if not isinstance(environment, dict):
        return {}
    keep = {"episode_length", "math_backend", "reward_mode", "use_math_verify", "mix_seed", "code_backend"}
    return {k: environment[k] for k in keep if k in environment}


def _prod_int(xs: Sequence[int]) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return int(p)


def _is_flat_fanout(*, fanout: Sequence[int], k: int) -> bool:
    """True iff fanout is [K,1,1,...]."""
    try:
        return bool(fanout) and int(fanout[0]) == int(k) and all(int(x) == 1 for x in fanout[1:])
    except Exception:
        return False


def _lower(x: object) -> str:
    return ("" if x is None else str(x)).strip().lower()


def _truthy_str(x: object) -> bool:
    """Interpret legacy dump flags robustly."""
    if x is None:
        return False
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("", "0", "false", "no", "off", "none", "null"):
            return False
        if s in ("1", "true", "yes", "on"):
            return True
        return True  # non-empty path-like string
    return False


def _dump_jsonl_path(args) -> Optional[str]:
    for name in ("dump_rollouts_jsonl_path", "dump_rollouts_jsonl", "dump_rollouts_path"):
        v = getattr(args, name, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _dump_enabled(args) -> bool:
    """Dump enabled across new/legacy flags."""
    try:
        if int(getattr(args, "dump_rollouts_every", 0) or 0) > 0:
            return True
    except Exception:
        pass
    if _dump_jsonl_path(args) is not None:
        return True

    for name in ("dump_rollouts", "dump_rollouts_jsonl"):
        v = getattr(args, name, None)
        if v is None:
            continue
        if isinstance(v, bool) and v:
            return True
        try:
            if int(v or 0) > 0:
                return True
        except Exception:
            if _truthy_str(v):
                return True
    return False


def _is_c3_run(args, marl_algorithm_lower: str) -> bool:
    """Robust C3 detection across forks."""
    for k in ("alg", "algorithm", "rl_alg", "rl_algorithm", "trainer_alg", "train_alg", "ppo_alg"):
        if _lower(getattr(args, k, None)) == "c3":
            return True
    if marl_algorithm_lower == "c3":
        return True
    if _lower(getattr(args, "c3_value_mode", None)) in ("value_assisted", "value_only"):
        return True
    return False


def _infer_fallback_reward_source(
    *, provider_name: str, env_name: str, env_cfg: dict, remote_reward_model: object
) -> str:
    """Fallback label when rr.source is empty/unknown."""
    pn = (provider_name or "unknown").strip()
    pn_l = pn.lower()

    def _mathenv_label() -> str:
        mb = env_cfg.get("math_backend")
        return f"MathEnv:{mb}" if mb else "MathEnv"

    if pn_l in ("auto", "chain", "default"):
        if remote_reward_model is not None:
            return "remote_rm"
        if env_name == "MathEnv":
            return _mathenv_label()
        return str(env_name or "env")

    if pn_l in ("env", "mathenv", "codeenv"):
        if env_name == "MathEnv":
            return _mathenv_label()
        return str(env_name or "env")

    if pn_l in ("rm", "reward_model", "remote_rm", "remote-reward-model", "rewardmodel"):
        return "remote_rm"

    return pn or "unknown"


# =============================================================================
# Prompt / state composition
# =============================================================================
def _strip_chat_wrappers(s: str) -> str:
    ss = (s or "").strip()
    ss = ss.replace("<|im_start|>", "").replace("<|im_end|>", "")
    ss = _RE_SYSTEM_PREFIX.sub("", ss)
    return ss.strip()


def _compose_full_prompt_fallback(*, system_prompt: str, question: str, context: str) -> str:
    parts = [system_prompt.strip(), "", "Problem:", (question or "").strip()]
    if (context or "").strip():
        parts.extend(["", "Context:", context.strip()])
    return "\n".join(parts).strip() + "\n"


def _compose_full_prompt_chat(*, tokenizer, system_prompt: str, question: str, context: str) -> str:
    """Prefer tokenizer.apply_chat_template when available."""
    sys_content = _strip_chat_wrappers(system_prompt)
    user_content = "Problem:\n" + (question or "").strip()
    if (context or "").strip():
        user_content += "\n\nContext:\n" + context.strip()

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "system", "content": sys_content}, {"role": "user", "content": user_content}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception:
                pass

    return _compose_full_prompt_fallback(system_prompt=sys_content, question=question, context=context)


def _compose_mappo_state_text(
    *,
    question: str,
    topo_so_far: Sequence[str],
    role_outputs: StrMap,
    next_role: str,
    next_role_id: int,
    num_roles: int,
    depth: int,
) -> str:
    """Role-aware state for centralized critic / stable grouping."""
    q = (question or "").strip()
    parts: List[str] = ["Problem:", q]

    if topo_so_far:
        parts += ["", "Conversation so far:"]
        for rn in topo_so_far:
            out = (role_outputs.get(rn, "") or "").strip()
            if out:
                parts += [f"[{rn}]", out]

    parts += [
        "",
        f"Next role: {str(next_role)}",
        f"Next role id: {int(next_role_id)}",
        f"Num roles: {int(num_roles)}",
        f"Step: {int(depth)}",
        f"<mappo_next_role={str(next_role)}>",
        f"<mappo_next_role_id={int(next_role_id)}>",
        f"<mappo_step={int(depth)}>",
    ]
    return "\n".join(parts).strip() + "\n"


# =============================================================================
# Dry-run plan
# =============================================================================
@dataclass
class MASRolloutPlan:
    question_id: int
    k_id: int
    role_prompts: List[Tuple[str, str]]  # (role, rendered_prompt)


# =============================================================================
# Internal prefix-tree state / node records
# =============================================================================
@dataclass(slots=True)
class _PrefixState:
    qid: int
    question: str
    node_id: int
    depth: int
    path: Tuple[int, ...]
    role_outputs: StrMap


@dataclass(slots=True)
class _NodeRec:
    qid: int
    node_id: int
    parent_id: int  # true parent node_id (not collapsed)
    depth: int
    role: str
    role_id: int
    path: Tuple[int, ...]
    leaf_start: int
    leaf_size: int
    is_leaf: int
    k_id: int
    prompt_token_ids: List[int]
    output_token_ids: List[int]
    output_logprobs: Any

    reward: Optional[float] = None
    reward_source: Optional[str] = None
    reward_info_json: Optional[str] = None

    # Always-cheap compact meta
    ctx_hash: int = 0
    ctx_kind: str = "state_text"
    prompt_hash: int = 0
    state_hash: int = 0
    output_hash: int = 0
    prompt_nchars: int = 0
    state_nchars: int = 0
    output_nchars: int = 0

    # Optional heavy fields
    prompt_text: Optional[str] = None
    state_text: Optional[str] = None
    output_text: Optional[str] = None
    traj_role_outputs: Optional[StrMap] = None


@dataclass(slots=True)
class _LeafAux:
    qid: int
    k_id: int
    question: str
    role_outputs: StrMap


# =============================================================================
# MAS rollout generator
# =============================================================================
class MASRolloutGenerator:
    """
    Init modes:
      - MASRolloutGenerator(task_spec: TaskSpec): dry-run planner
      - MASRolloutGenerator(vllm_engines, strategy, tokenizer, prompt_max_len): training generator
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], TaskSpec):
            self.strategy = None
            self.args = None
            self.vllm_engines = None
            self.vllm_engines_by_role = None
            self.tokenizer = None
            self.prompt_max_len = None
            self.task_spec = args[0]
            self._init_from_task(self.task_spec)
            return

        if len(args) < 4:
            raise TypeError(
                "MASRolloutGenerator expects (task_spec) for dry-run or "
                "(vllm_engines, strategy, tokenizer, prompt_max_len) for training"
            )

        vllm_engines, strategy, tokenizer, prompt_max_len = args[:4]
        self.strategy = strategy
        self.args = strategy.args

        self.vllm_engines_by_role = vllm_engines if isinstance(vllm_engines, dict) else None
        self.vllm_engines = (
            [e for lst in vllm_engines.values() for e in lst] if isinstance(vllm_engines, dict) else vllm_engines
        )

        self.tokenizer = tokenizer
        self.prompt_max_len = int(prompt_max_len)

        task_path = getattr(self.args, "c3_task", None)
        if not task_path:
            raise ValueError("--c3_task must be set when using MASRolloutGenerator as rollout_generator_cls")
        self.task_spec = load_task(task_path)
        self._init_from_task(self.task_spec)

    def _init_from_task(self, task_spec: TaskSpec) -> None:
        self.roles: Tuple[RoleSpec, ...] = tuple(task_spec.roles)
        self._role_by_name: Dict[str, RoleSpec] = {r.name: r for r in self.roles}

        self.graph = RoleGraph(self.roles)
        self.topo = self.graph.topo_order()

        answer_roles = [r.name for r in self.roles if r.with_answer]
        self.answer_roles = tuple(answer_roles)

        ans = None
        if answer_roles:
            for name in reversed(self.topo):
                if name in answer_roles:
                    ans = name
                    break
        self.answer_role = ans if ans is not None else self.topo[-1]

    # -------------------------------------------------------------------------
    # Dry-run
    # -------------------------------------------------------------------------
    @classmethod
    def from_task_path(cls, task_path: str) -> "MASRolloutGenerator":
        return cls(load_task(task_path))

    def plan_rollouts(self, questions: Sequence[str], *, k: int) -> List[MASRolloutPlan]:
        if k <= 0:
            raise ValueError("k must be >= 1")

        plans: List[MASRolloutPlan] = []
        topo = list(self.topo)

        for qi, q in enumerate(questions):
            for kid in range(int(k)):
                role_outputs: StrMap = {}
                topo_so_far: List[str] = []
                prompts: List[Tuple[str, str]] = []

                for role_name in topo:
                    role = self._role_by_name[role_name]
                    ctx = build_render_context(question=q, role_outputs=role_outputs, topo_so_far=topo_so_far)
                    rendered = render_role_prompt(role.prompt, ctx=ctx)
                    prompts.append((role_name, rendered))
                    topo_so_far.append(role_name)

                plans.append(MASRolloutPlan(question_id=int(qi), k_id=int(kid), role_prompts=prompts))

        return plans

    # -------------------------------------------------------------------------
    # Training API (SamplesGenerator compatible)
    # -------------------------------------------------------------------------
    def tokenize_fn(self, texts: List[str], max_length: int, padding: bool = True, device=None):
        """padding=False returns python lists for vLLM."""
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

    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs):
        import torch

        if self.vllm_engines is None:
            raise RuntimeError("MASRolloutGenerator requires vLLM engines; set --vllm_num_engines > 0")

        with torch.no_grad():
            if self.args.vllm_enable_sleep:
                from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

                batch_vllm_engine_call(self.vllm_engines, "wake_up")

            samples = self._generate_mas_vllm(all_prompts, all_labels, **generate_kwargs)

            if self.args.vllm_enable_sleep:
                from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

                batch_vllm_engine_call(self.vllm_engines, "sleep")

            return samples

    # -------------------------------------------------------------------------
    # Core generation
    # -------------------------------------------------------------------------
    def _generate_mas_vllm(self, all_prompts: List[str], all_labels, **kwargs):
        """Prefix-tree rollout + leaf rewards; returns one Experience per node."""
        import ray
        import torch
        from vllm import SamplingParams

        from c3.rewards.registry import build_reward_provider
        from openrlhf.trainer.ppo_utils.experience_maker import Experience

        args = self.args
        phase = str(kwargs.pop("phase", "train")).strip().lower()
        all_metas = kwargs.pop("all_metas", None)

        if all_metas is not None:
            if not isinstance(all_metas, (list, tuple)):
                raise ValueError(f"all_metas must be a list/tuple, got {type(all_metas)}")
            if len(all_metas) != len(all_prompts):
                raise ValueError(f"all_metas length mismatch: {len(all_metas)} vs {len(all_prompts)}")

        env_name = str(self.task_spec.env_name)
        env_cfg = _extract_env_cfg(self.task_spec.environment)

        topo = list(self.topo)
        num_roles = len(topo)
        role_id_map = {name: i for i, name in enumerate(topo)}
        answer_roles = list(getattr(self, "answer_roles", ()))
        answer_role = self.answer_role

        n_samples_per_prompt = int(kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt))
        marl_alg0 = _lower(getattr(args, "marl_algorithm", "auto") or "auto")
        is_c3 = _is_c3_run(args, marl_alg0)
        is_mappo = marl_alg0 == "mappo"

        dump_enabled = _dump_enabled(args)
        dump_compact = bool(getattr(args, "dump_rollouts_compact", False))
        keep_rollout_texts = bool(getattr(args, "keep_rollout_texts", False))

        # Text retention gates (keep_rollout_texts overrides).
        dump_texts_ok = dump_enabled and (not dump_compact)
        keep_prompt_text = keep_rollout_texts or dump_texts_ok or is_mappo
        keep_state_text = keep_rollout_texts or dump_texts_ok or is_mappo or is_c3
        keep_output_text = keep_rollout_texts or dump_texts_ok

        # C3 credit/Q-critic needs traj_role_outputs (always for leaves; optionally for all nodes).
        keep_traj_role_outputs = keep_rollout_texts or dump_enabled or is_c3

        # Debug-only extras (largest footprint).
        keep_debug_blob = dump_texts_ok

        # No-replay knob (assumed finalized upstream; we only consume it here).
        no_replay = bool(getattr(args, "c3_no_replay", False))

        try:
            flag = f"_logged_rollout_gates_{phase}"
            if self.strategy is not None and self.strategy.is_rank_0() and not getattr(self, flag, False):
                logger.info(
                    "[MAS][%s] gates: marl_algorithm=%s is_c3=%s no_replay=%s keep_rollout_texts=%s dump=%s compact=%s "
                    "keep_prompt=%s keep_state=%s keep_output=%s keep_traj_out=%s",
                    phase,
                    marl_alg0,
                    is_c3,
                    no_replay,
                    keep_rollout_texts,
                    dump_enabled,
                    dump_compact,
                    keep_prompt_text,
                    keep_state_text,
                    keep_output_text,
                    keep_traj_role_outputs,
                )
                setattr(self, flag, True)
        except Exception:
            pass

        sanitize_fn = None
        if env_name == "MathEnv":
            from c3.text_sanitize import sanitize_math_solution_text as sanitize_fn  # noqa: N812

        # -------------------- fanout / mode --------------------
        train_k = int(getattr(args, "n_samples_per_prompt", n_samples_per_prompt) or n_samples_per_prompt)
        is_train_call = (phase == "train") and (int(n_samples_per_prompt) == int(train_k))
        use_user_fanout = is_c3 and is_train_call

        if use_user_fanout:
            fanout = getattr(args, "c3_fanout_list", None)
            if fanout is None:
                raw = getattr(args, "c3_fanout", None)
                if raw is None:
                    if num_roles == 2 and int(n_samples_per_prompt) == 8:
                        fanout = [2, 4]
                        logger.info("[MAS][c3_train_nested] --c3_fanout not set; using default fanout=%s", fanout)
                    else:
                        raise RuntimeError(
                            "[C3][FAIL-FAST] Missing c3_fanout for C3 training. "
                            "Set --c3_fanout and ensure product(fanout)==n_samples_per_prompt."
                        )
                else:
                    try:
                        fanout = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
                    except Exception as e:
                        raise RuntimeError(f"[C3][FAIL-FAST] Invalid c3_fanout={raw!r}: {e}")

            if len(fanout) != num_roles:
                raise RuntimeError(
                    f"[C3][FAIL-FAST] len(c3_fanout)={len(fanout)} must equal num_roles={num_roles}; "
                    f"roles_topo={topo}, fanout={fanout}"
                )
            if any(int(x) <= 0 for x in fanout):
                raise RuntimeError(f"[C3][FAIL-FAST] c3_fanout must be positive ints, got {fanout}")

            K_nested = _prod_int(fanout)
            if K_nested != int(n_samples_per_prompt):
                raise RuntimeError(
                    f"[C3][FAIL-FAST] product(c3_fanout)={K_nested} must equal n_samples_per_prompt={n_samples_per_prompt}; "
                    f"fanout={fanout}"
                )
            mode = "c3_train_nested"
        else:
            fanout = [int(n_samples_per_prompt)] + [1] * (num_roles - 1)
            mode = "flat_derived"

        is_flat = _is_flat_fanout(fanout=fanout, k=n_samples_per_prompt)

        if no_replay and (not is_flat) and self.strategy is not None and self.strategy.is_rank_0():
            # No-replay is defined for flat fanout; allow user override but make it visible.
            logger.warning("[MAS][C3] c3_no_replay=True but fanout is not flat: fanout=%s mode=%s", fanout, mode)

        # -------------------- sampling params --------------------
        temperature = float(kwargs.get("temperature", 1.0))
        top_p = float(kwargs.get("top_p", 1.0))
        top_k = int(kwargs.get("top_k", -1))
        max_new_tokens = int(kwargs.get("max_new_tokens", 1024))
        min_new_tokens = int(kwargs.get("min_new_tokens", 1))
        skip_special_tokens = bool(kwargs.get("skip_special_tokens", False))
        logprobs = 1 if args.enable_vllm_is_correction else None

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            min_tokens=min_new_tokens,
            skip_special_tokens=skip_special_tokens,
            include_stop_str_in_output=True,
            logprobs=logprobs,
        )

        truncate_length = int(self.prompt_max_len) + int(max_new_tokens)

        # total nodes per question: sum_{d=0..R-1} prod_{t<=d} fanout[t]
        tot_nodes_per_q = 0
        prod_so_far = 1
        for x in fanout:
            prod_so_far *= int(x)
            tot_nodes_per_q += prod_so_far

        def _leaf_size(depth: int) -> int:
            s = 1
            for t in range(depth + 1, num_roles):
                s *= int(fanout[t])
            return int(s)

        def _encode_prefix(path_digits: Tuple[int, ...]) -> int:
            v = 0
            for t, d in enumerate(path_digits):
                v = v * int(fanout[t]) + int(d)
            return int(v)

        # -------------------- vLLM helpers --------------------
        def _get_llms(role_name: str):
            if self.vllm_engines_by_role is not None:
                llms = self.vllm_engines_by_role.get(role_name)
                if not llms:
                    raise RuntimeError(f"per_role mode: missing vllm engines for role={role_name!r}")
            else:
                llms = self.vllm_engines
            if not llms:
                raise RuntimeError(f"Empty vLLM engines for role={role_name!r}")
            return llms

        def _vllm_generate(role_name: str, prompts: List[str]):
            if not prompts:
                return []

            llms = _get_llms(role_name)
            refs = []
            used_llms = []
            batch_size = (len(prompts) + len(llms) - 1) // len(llms)

            for i, llm in enumerate(llms):
                chunk = prompts[i * batch_size : (i + 1) * batch_size]
                if not chunk:
                    continue
                used_llms.append(llm)
                refs.append(llm.add_requests.remote(sampling_params=sampling_params, prompts=chunk))

            if not refs:
                return []

            ray.get(refs)
            parts = ray.get([llm.get_responses.remote() for llm in used_llms])

            outs = []
            for p in parts:
                if p:
                    outs.extend(p)

            if len(outs) != len(prompts):
                raise RuntimeError(
                    f"vLLM returned unexpected outputs for role={role_name}: expected={len(prompts)} got={len(outs)}"
                )
            return outs

        def _extract_logprobs(out_obj):
            if not args.enable_vllm_is_correction:
                return None
            lp_dicts = getattr(out_obj.outputs[0], "logprobs", None)
            if lp_dicts is None:
                return None

            resp_ids = list(out_obj.outputs[0].token_ids)
            if not resp_ids:
                return torch.empty((0,), dtype=torch.float32, device="cpu")

            lp: List[float] = []
            for t, logprob_dict in enumerate(lp_dicts):
                tok_id = resp_ids[t]
                lp.append(float(logprob_dict[tok_id].logprob))
            return torch.tensor(lp, dtype=torch.float32, device="cpu")

        def _decode_output_text(out_obj) -> Tuple[str, List[int], List[int]]:
            prompt_token_ids = list(out_obj.prompt_token_ids)
            out_token_ids = list(out_obj.outputs[0].token_ids)
            out_text = getattr(out_obj.outputs[0], "text", None)
            if not isinstance(out_text, str) or out_text == "":
                out_text = self.tokenizer.decode(out_token_ids, skip_special_tokens=skip_special_tokens)
            return out_text, prompt_token_ids, out_token_ids

        # -------------------- PPO-aligned tensors --------------------
        def _build_ppo_aligned_tensors(
            *,
            prompt_ids: List[int],
            out_ids: List[int],
            resp_log_probs: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, float, bool]:
            input_ids = prompt_ids + out_ids
            raw_len = len(input_ids)

            seq_len = raw_len if raw_len <= truncate_length else int(truncate_length)
            if seq_len <= 0:
                raise RuntimeError("[MAS][FAIL-FAST] Empty sequence after truncation.")

            seq_ids = input_ids[:seq_len]
            sequences = torch.tensor(seq_ids, dtype=torch.long, device="cpu")
            attention_mask_t = torch.ones((seq_len,), dtype=torch.long, device="cpu")

            prompt_len_eff = min(len(prompt_ids), seq_len)
            resp_len_eff = max(0, seq_len - prompt_len_eff)

            aligned_len = max(0, seq_len - 1)
            action_mask = torch.zeros((aligned_len,), dtype=torch.long, device="cpu")

            if aligned_len > 0 and resp_len_eff > 0 and prompt_len_eff > 0:
                start = prompt_len_eff - 1
                end = min(aligned_len, start + resp_len_eff)
                if start < end:
                    action_mask[start:end] = 1

            rollout_log_probs_aligned = None
            if resp_log_probs is not None:
                rollout_log_probs_aligned = torch.zeros((aligned_len,), dtype=torch.float32, device="cpu")
                if aligned_len > 0 and resp_len_eff > 0 and prompt_len_eff > 0:
                    start = prompt_len_eff - 1
                    end = min(aligned_len, start + resp_len_eff)
                    if start < end:
                        rollout_log_probs_aligned[start:end] = resp_log_probs[: (end - start)]

            raw_resp_len = len(out_ids)
            is_truncated = raw_len > seq_len
            is_clipped = (raw_resp_len >= max_new_tokens) or bool(is_truncated)

            return (
                sequences,
                attention_mask_t,
                action_mask,
                rollout_log_probs_aligned,
                int(resp_len_eff),
                float(seq_len),
                bool(is_clipped),
            )

        # -------------------- dataset meta alignment --------------------
        meta_jsons: List[Optional[str]] = [None] * len(all_prompts)
        meta_dicts: List[Optional[dict]] = [None] * len(all_prompts)
        if all_metas is not None:
            meta_jsons = [m for m in all_metas]
            meta_dicts = [_safe_json_dict(m) for m in all_metas]

        # Collision guard for this rollout-generation call.
        # We scope it to the call (not the whole process) to avoid unbounded
        # memory growth during long training runs, while still providing
        # fail-fast safety within a batch.
        ctx_guard = CollisionGuard()

        # -------------------- expand prefix tree role-by-role --------------------
        next_node_id = [0 for _ in range(len(all_prompts))]
        parents: List[_PrefixState] = [
            _PrefixState(qid=qi, question=q, node_id=-1, depth=-1, path=(), role_outputs={})
            for qi, q in enumerate(all_prompts)
        ]

        nodes: List[_NodeRec] = []
        leaves_aux: List[_LeafAux] = []

        for depth, role_name in enumerate(topo):
            role = self._role_by_name[role_name]
            topo_so_far = topo[:depth]
            fan = int(fanout[depth])

            prompts_batch: List[str] = []
            state_texts: Optional[List[str]] = [] if keep_state_text else None

            # compact meta aligned with prompts_batch
            ctx_hashes: List[int] = []
            ctx_kinds: List[str] = []
            prompt_hashes: List[int] = []
            state_hashes: List[int] = []
            prompt_lens: List[int] = []
            state_lens: List[int] = []

            parent_refs: List[_PrefixState] = []
            sibling_ids: List[int] = []

            for st in parents:
                role_outputs_for_ctx = st.role_outputs
                if sanitize_fn is not None and role_outputs_for_ctx:
                    role_outputs_for_ctx = {k: sanitize_fn(v) for k, v in role_outputs_for_ctx.items()}

                ctx = build_render_context(question=st.question, role_outputs=role_outputs_for_ctx, topo_so_far=topo_so_far)
                system_prompt = render_role_prompt(role.prompt, ctx=ctx)
                full_prompt = _compose_full_prompt_chat(
                    tokenizer=self.tokenizer,
                    system_prompt=system_prompt,
                    question=st.question,
                    context=ctx.get("context", ""),
                )

                stxt = _compose_mappo_state_text(
                    question=st.question,
                    topo_so_far=topo_so_far,
                    role_outputs=role_outputs_for_ctx,
                    next_role=role_name,
                    next_role_id=int(role_id_map[role_name]),
                    num_roles=int(num_roles),
                    depth=int(depth),
                )

                ph = hash63(full_prompt)
                sh = hash63(stxt)
                if stxt:
                    ch = sh
                    ck = "state_text"
                    ctx_guard.observe(ch, fingerprint(stxt), where=f"MASRolloutGenerator/ctx/{role_name}/{depth}")
                else:
                    ch = ph
                    ck = "prompt"
                    ctx_guard.observe(ch, fingerprint(full_prompt), where=f"MASRolloutGenerator/ctx/{role_name}/{depth}")

                # Also observe the component hashes to detect cross-type collisions.
                ctx_guard.observe(ph, fingerprint(full_prompt), where=f"MASRolloutGenerator/prompt/{role_name}/{depth}")
                ctx_guard.observe(sh, fingerprint(stxt), where=f"MASRolloutGenerator/state/{role_name}/{depth}")

                for s in range(fan):
                    prompts_batch.append(full_prompt)
                    if state_texts is not None:
                        state_texts.append(stxt)

                    parent_refs.append(st)
                    sibling_ids.append(s)

                    ctx_hashes.append(ch)
                    ctx_kinds.append(ck)
                    prompt_hashes.append(ph)
                    state_hashes.append(sh)
                    prompt_lens.append(len(full_prompt))
                    state_lens.append(len(stxt))

            outs = _vllm_generate(role_name, prompts_batch)

            children: List[_PrefixState] = []
            for idx, out in enumerate(outs):
                st = parent_refs[idx]
                s = int(sibling_ids[idx])

                qid = int(st.qid)
                nid = next_node_id[qid]
                next_node_id[qid] += 1

                path = st.path + (s,)
                prefix_id = _encode_prefix(path)
                leaf_sz = _leaf_size(depth)
                leaf_start = prefix_id * leaf_sz
                is_leaf = int(depth == num_roles - 1)
                k_id = int(prefix_id) if (is_leaf or is_flat) else -1

                out_text, prompt_token_ids, out_token_ids = _decode_output_text(out)

                role_outputs = dict(st.role_outputs)
                role_outputs[role_name] = out_text

                # Observe output hashes (within this batch) to detect any rare
                # cross-text collision of the 63-bit key.
                out_h = int(hash63(out_text))
                ctx_guard.observe(out_h, fingerprint(out_text), where=f"MASRolloutGenerator/output/{role_name}/{depth}")

                rec = _NodeRec(
                    qid=qid,
                    node_id=nid,
                    parent_id=int(st.node_id),
                    depth=int(depth),
                    role=role_name,
                    role_id=int(role_id_map[role_name]),
                    path=path,
                    leaf_start=int(leaf_start),
                    leaf_size=int(leaf_sz),
                    is_leaf=int(is_leaf),
                    k_id=int(k_id),
                    prompt_token_ids=prompt_token_ids,
                    output_token_ids=out_token_ids,
                    output_logprobs=_extract_logprobs(out),
                    ctx_hash=int(ctx_hashes[idx]),
                    ctx_kind=str(ctx_kinds[idx]),
                    prompt_hash=int(prompt_hashes[idx]),
                    state_hash=int(state_hashes[idx]),
                    output_hash=int(out_h),
                    prompt_nchars=int(prompt_lens[idx]),
                    state_nchars=int(state_lens[idx]),
                    output_nchars=int(len(out_text)),
                )

                if keep_prompt_text:
                    rec.prompt_text = prompts_batch[idx]
                if keep_state_text and state_texts is not None:
                    rec.state_text = state_texts[idx]
                if keep_output_text:
                    rec.output_text = out_text

                # Always keep traj_role_outputs for leaves; optionally for all nodes.
                if keep_traj_role_outputs or bool(is_leaf):
                    rec.traj_role_outputs = role_outputs

                nodes.append(rec)

                if is_leaf:
                    leaves_aux.append(_LeafAux(qid=qid, k_id=int(k_id), question=st.question, role_outputs=role_outputs))
                else:
                    children.append(
                        _PrefixState(
                            qid=qid,
                            question=st.question,
                            node_id=nid,
                            depth=int(depth),
                            path=path,
                            role_outputs=role_outputs,
                        )
                    )

            parents = children

        # -------------------- reconstruct prompts for reward/meta (leaves) --------------------
        def _rebuild_role_prompts_for_leaf(question: str, role_outputs_full: StrMap) -> StrMap:
            """Re-render full prompts per role for an episode (needed for reward meta)."""
            out: StrMap = {}
            topo_so_far_local: List[str] = []
            rolling_outputs: StrMap = {}

            for rn in topo:
                role_spec = self._role_by_name[rn]
                role_outputs_for_ctx = rolling_outputs
                if sanitize_fn is not None and role_outputs_for_ctx:
                    role_outputs_for_ctx = {k: sanitize_fn(v) for k, v in role_outputs_for_ctx.items()}

                ctx = build_render_context(question=question, role_outputs=role_outputs_for_ctx, topo_so_far=topo_so_far_local)
                system_prompt = render_role_prompt(role_spec.prompt, ctx=ctx)
                full_prompt = _compose_full_prompt_chat(
                    tokenizer=self.tokenizer,
                    system_prompt=system_prompt,
                    question=question,
                    context=ctx.get("context", ""),
                )
                out[rn] = full_prompt

                rolling_outputs = dict(rolling_outputs)
                rolling_outputs[rn] = role_outputs_full.get(rn, "")
                topo_so_far_local.append(rn)

            return out

        # -------------------- reward computation (leaves only) --------------------
        remote_reward_model = kwargs.get("remote_reward_model", None)
        provider_name = getattr(args, "reward_provider_cls", "auto")
        reward_provider = build_reward_provider(provider_name, env_name=env_name, remote_reward_model=remote_reward_model)

        from c3.rewards.base import RewardRequest

        reward_reqs: List[RewardRequest] = []
        leaf_traj_prompts: List[StrMap] = []
        leaf_prompts_by_qk: Dict[Tuple[int, int], StrMap] = {}  # for optional logging / info

        for lf in leaves_aux:
            traj_prompts = _rebuild_role_prompts_for_leaf(lf.question, lf.role_outputs)
            leaf_traj_prompts.append(traj_prompts)
            leaf_prompts_by_qk[(int(lf.qid), int(lf.k_id))] = traj_prompts

            atxt = lf.role_outputs.get(answer_role, "")
            ptxt = traj_prompts.get(answer_role, "")
            query_text = str(ptxt) + str(atxt)

            qid = int(lf.qid)
            kid = int(lf.k_id)

            label = all_labels[qid]
            label_text = None if label is None else str(label)

            meta = {
                "question": all_prompts[qid],
                "question_id": int(qid),
                "k_id": int(kid),
                "task_name": self.task_spec.experiment_name,
                "task_path": self.task_spec.task_path,
                "env_name": env_name,
                "task_env_cfg": env_cfg,
                "roles_topo": list(topo),
                "answer_role": answer_role,
                "answer_roles": answer_roles,
                "answer_text": atxt,
                "traj_role_outputs": {rn: lf.role_outputs.get(rn, "") for rn in topo},
                "traj_role_prompts": {rn: traj_prompts.get(rn, "") for rn in topo},
            }
            if meta_jsons[qid] is not None:
                meta["dataset_meta_json"] = meta_jsons[qid]
            if meta_dicts[qid] is not None:
                meta["dataset_meta"] = meta_dicts[qid]

            reward_reqs.append(
                RewardRequest(
                    query_text=query_text,
                    prompt_text=all_prompts[qid],
                    label_text=label_text,
                    meta=meta,
                )
            )

        reward_results = reward_provider.compute(reward_reqs)
        fallback_src = _infer_fallback_reward_source(
            provider_name=str(provider_name),
            env_name=str(env_name),
            env_cfg=env_cfg,
            remote_reward_model=remote_reward_model,
        )

        # -------------------- leaf rewards per question --------------------
        K = int(n_samples_per_prompt)
        leaf_rewards_by_q: List[List[float]] = [[float("nan")] * K for _ in range(len(all_prompts))]
        leaf_src_by_q: List[List[Optional[str]]] = [[None] * K for _ in range(len(all_prompts))]
        leaf_info_by_q: List[List[Optional[str]]] = [[None] * K for _ in range(len(all_prompts))]

        for lf, rr in zip(leaves_aux, reward_results):
            if rr is None:
                raise RuntimeError(
                    f"RewardProvider({provider_name!r}) returned None for some leaves; check labels/RM availability."
                )

            qid = int(lf.qid)
            kid = int(lf.k_id)
            if not (0 <= kid < K):
                raise RuntimeError(
                    f"[C3][FAIL-FAST] Leaf k_id out of range: question_id={qid} k_id={kid} expected [0..{K-1}]"
                )

            leaf_rewards_by_q[qid][kid] = float(rr.reward)

            src = (getattr(rr, "source", None) or "").strip()
            if (not src) or (src.lower() == "unknown"):
                info = rr.info if isinstance(rr.info, dict) else {}
                env = (info.get("env") or info.get("env_name") or "").strip()
                backend = (info.get("backend_used") or info.get("backend") or "").strip()

                if env and backend:
                    src = f"{env}:{backend}"
                elif backend:
                    src = f"{env_name}:{backend}" if env_name else backend
                elif env:
                    src = env
                else:
                    src = fallback_src

            leaf_src_by_q[qid][kid] = src
            leaf_info_by_q[qid][kid] = json.dumps(rr.info, ensure_ascii=False) if rr.info is not None else None

        for qi in range(len(all_prompts)):
            if any((not isinstance(x, float)) or (x != x) for x in leaf_rewards_by_q[qi]):  # NaN: x!=x
                raise RuntimeError(
                    f"[C3][FAIL-FAST] Missing leaf rewards for question_id={qi}. "
                    "This indicates a bug in leaf indexing or reward computation."
                )

        # -------------------- node rewards: subtree mean --------------------
        for node in nodes:
            qid = int(node.qid)
            s0 = int(node.leaf_start)
            sz = int(node.leaf_size)
            seg = leaf_rewards_by_q[qid][s0 : s0 + sz]
            node.reward = float(sum(seg)) / float(len(seg))

            if int(node.is_leaf) == 1:
                kid = int(node.k_id)
                node.reward_source = str(leaf_src_by_q[qid][kid] or fallback_src or "unknown")
                node.reward_info_json = leaf_info_by_q[qid][kid]
            else:
                node.reward_source = "subtree_mean"
                node.reward_info_json = None

        # -------------------- build Experiences (one per node) --------------------
        samples_list: List[Experience] = []

        parent_base = int(tot_nodes_per_q) + 1
        q_base = int(num_roles) * parent_base
        roles_topo_str = "->".join(topo)

        for node in nodes:
            qid = int(node.qid)
            role_name = str(node.role)
            rid = int(node.role_id)
            depth = int(node.depth)

            path = node.path
            sibling_id = int(path[-1]) if path else -1
            path_str = "-".join(str(x) for x in path) if path else ""

            (
                sequences,
                attention_mask_t,
                action_mask,
                rollout_log_probs,
                response_length,
                total_length,
                is_clipped,
            ) = _build_ppo_aligned_tensors(
                prompt_ids=list(node.prompt_token_ids),
                out_ids=list(node.output_token_ids),
                resp_log_probs=node.output_logprobs,
            )

            node_id = int(node.node_id)
            parent_id_true = int(node.parent_id)
            parent_id = -1 if (no_replay and depth > 0) else parent_id_true

            if (marl_alg0 in ("magrpo", "mappo")) and bool(is_flat):
                kid = int(node.k_id)
                if kid < 0 or kid >= int(n_samples_per_prompt):
                    raise RuntimeError(
                        f"[{marl_alg0.upper()}][FAIL-FAST] invalid k_id for flat-derived traj_id: question_id={qid} k_id={kid} "
                        f"expected [0..{int(n_samples_per_prompt)-1}] (is_flat={is_flat})."
                    )
                traj_id = int(qid) * int(n_samples_per_prompt) + int(kid)
            else:
                traj_id = int(qid) * int(tot_nodes_per_q) + int(node_id)

            # Encode (qid, rid, parent_id) for stable grouping. parent_id may be collapsed in no-replay mode.
            adv_group_id = int(qid) * int(q_base) + int(rid) * int(parent_base) + int(parent_id + 1)
            magrpo_group_id = int(qid) * int(num_roles) + int(rid)

            reward_t = torch.tensor([float(node.reward)], dtype=torch.float32)

            rj = node.reward_info_json
            if rj is None:
                rj_s = "null"
            elif isinstance(rj, str):
                rj_s = rj
            else:
                try:
                    rj_s = json.dumps(rj, ensure_ascii=False)
                except Exception:
                    rj_s = str(rj)

            info: Dict[str, Any] = {
                "response_length": torch.tensor([int(response_length)]),
                "total_length": torch.tensor([float(total_length)]),
                "response_clip_ratio": torch.tensor([bool(is_clipped)]),
                "marl_enabled": torch.tensor([1]),
                "question_id": torch.tensor([qid]),
                "k_id": torch.tensor([int(node.k_id)]),
                "role_id": torch.tensor([rid], dtype=torch.long),
                "num_roles": torch.tensor([num_roles]),
                "traj_id": torch.tensor([traj_id], dtype=torch.long),
                "adv_group_id": torch.tensor([adv_group_id], dtype=torch.long),
                "magrpo_group_id": torch.tensor([magrpo_group_id], dtype=torch.long),
                "is_leaf": torch.tensor([int(node.is_leaf)], dtype=torch.long),
                "c3_node_id": torch.tensor([node_id], dtype=torch.long),
                "c3_depth": torch.tensor([depth], dtype=torch.long),
                "c3_sibling_id": torch.tensor([sibling_id], dtype=torch.long),
                "c3_leaf_start": torch.tensor([int(node.leaf_start)], dtype=torch.long),
                "c3_leaf_size": torch.tensor([int(node.leaf_size)], dtype=torch.long),
                "c3_path": [path_str],
                # No-replay debug (collapsed vs true parent)
                "c3_parent_id": torch.tensor([int(parent_id)], dtype=torch.long),
                "c3_parent_id_true": torch.tensor([int(parent_id_true)], dtype=torch.long),
                "c3_no_replay": torch.tensor([1 if no_replay else 0], dtype=torch.long),
                "reward": reward_t,
                "score": reward_t,
                "reward_source": [str(node.reward_source or "unknown")],
                "reward_info_json": [rj_s],
                "role": [role_name],
                "task_name": [self.task_spec.experiment_name],
                "env_name": [env_name],
                "answer_role": [answer_role],
                "roles_topo": [roles_topo_str],
                "ctx_hash": torch.tensor([int(node.ctx_hash)], dtype=torch.long),
                "ctx_kind": [str(node.ctx_kind)],
            }

            # Compact dump metadata: cheap hashes/lengths even when texts are gated off.
            if dump_enabled and dump_compact and (not keep_rollout_texts):
                info["prompt_hash64"] = torch.tensor([int(node.prompt_hash)], dtype=torch.long)
                info["state_hash64"] = torch.tensor([int(node.state_hash)], dtype=torch.long)
                info["output_hash64"] = torch.tensor([int(node.output_hash)], dtype=torch.long)
                info["prompt_nchars"] = torch.tensor([int(node.prompt_nchars)], dtype=torch.long)
                info["state_nchars"] = torch.tensor([int(node.state_nchars)], dtype=torch.long)
                info["output_nchars"] = torch.tensor([int(node.output_nchars)], dtype=torch.long)

            if keep_prompt_text:
                info["prompt_text"] = [str(node.prompt_text or "")]
            if keep_state_text:
                info["state_text"] = [str(node.state_text or "")]
            if keep_output_text:
                info["output_text"] = [str(node.output_text or "")]

            if keep_traj_role_outputs:
                info["c3_keep_traj_role_outputs"] = torch.tensor([1], dtype=torch.long)

            # C3 materialization depends on traj_role_outputs; always present for leaves.
            if keep_traj_role_outputs or bool(getattr(node, "is_leaf", 0)):
                info["traj_role_outputs"] = node.traj_role_outputs or {}

            # If dumping texts, include per-leaf prompts (expensive to store for all nodes).
            if dump_texts_ok and bool(getattr(node, "is_leaf", 0)):
                tp = leaf_prompts_by_qk.get((qid, int(node.k_id)))
                if isinstance(tp, dict):
                    info["traj_role_prompts"] = tp

            if keep_debug_blob:
                info["question"] = [all_prompts[qid]]
                info["dataset_meta_json"] = [str(meta_jsons[qid]) if meta_jsons[qid] is not None else None]
                info["label"] = [str(all_labels[qid]) if all_labels[qid] is not None else None]
                md = meta_dicts[qid]
                if isinstance(md, dict):
                    info["meta_dict"] = md

            samples_list.append(
                Experience(
                    sequences=sequences.unsqueeze(0),
                    attention_mask=attention_mask_t.unsqueeze(0),
                    action_mask=action_mask.unsqueeze(0),
                    rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
                    rewards=reward_t,
                    scores=reward_t,
                    prompts=[all_prompts[qid]],
                    labels=[all_labels[qid]],
                    info=info,
                )
            )

        # -------------------- Budget ledger (Appendix B) --------------------
        # Record terminal evaluator call accounting for each training rollout-generation call.
        #
        # We only write when training loop indices are provided (global_step). This keeps the
        # ledger aligned to the paper's "per update" budget and avoids counting optional
        # warmup stages that may also generate rollouts.
        try:
            if str(phase) == "train":
                gs = kwargs.get("global_step", None)
                if gs is not None:
                    rec = make_budget_record(
                        global_step=gs,
                        epoch_idx=kwargs.get("epoch_idx", None),
                        iter_in_epoch=kwargs.get("iter_in_epoch", None),
                        marl_algorithm=marl_alg0,
                        n_questions_in_batch=len(all_prompts),
                        n_samples_per_prompt=int(n_samples_per_prompt),
                        roles_topo=topo,
                        fanout=fanout if isinstance(fanout, (list, tuple)) else None,
                    )
                    append_ledger(str(getattr(args, "run_dir", "") or ""), rec)
        except Exception:
            # Best-effort: never break training due to ledger IO.
            pass

        return samples_list


def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Dry-run C3 MAS rollout planning")
    p.add_argument("--task", type=str, required=True, help="Path to C3 task yaml")
    p.add_argument("--k", type=int, default=2, help="Number of rollouts per question (K)")
    p.add_argument("--question", type=str, default="2+2=?", help="A single question string to plan prompts for")
    args = p.parse_args()

    gen = MASRolloutGenerator.from_task_path(args.task)
    plans = gen.plan_rollouts([args.question], k=args.k)

    print("task:", gen.task_spec.experiment_name, "env=", gen.task_spec.env_name)
    print("roles_topo:", " -> ".join(gen.topo), "num_roles=", len(gen.roles))
    print("k:", args.k)
    for pl in plans:
        print(f"\n[question_id={pl.question_id} k_id={pl.k_id}]")
        for role, rp in pl.role_prompts:
            print(f"--- role={role} prompt ---")
            print(rp.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
