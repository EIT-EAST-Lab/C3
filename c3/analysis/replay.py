"""c3.analysis.replay

Counterfactual replay for C3 analysis (Credit / Influence).

A bucket = (restart context) + (candidate actions for target role) + (rollout returns).
Designed to be JSONL-friendly and schema-stable across experiments.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

from c3.utils.collision_guard import global_guard
from c3.utils.context_key import fingerprint, hash63


# -----------------------------------------------------------------------------
# Type/format helpers
# -----------------------------------------------------------------------------


def _role_to_name(role_obj: Any) -> str:
    """Extract canonical role name from either str or RoleSpec-like objects."""
    if isinstance(role_obj, str):
        return role_obj
    name = getattr(role_obj, "name", None)
    if isinstance(name, str) and name:
        return name
    name2 = getattr(role_obj, "role", None)
    if isinstance(name2, str) and name2:
        return name2
    raise TypeError(f"Unsupported role object in roles_topo: {type(role_obj)}: {role_obj!r}")


def _reward_to_float(x: Any) -> float:
    """Coerce reward output to float (supports scalar/tensor/tuple/dict)."""
    if x is None:
        raise TypeError("reward_fn returned None")
    if isinstance(x, (float, int)):
        return float(x)

    item = getattr(x, "item", None)
    if callable(item):
        try:
            return float(item())
        except Exception:
            pass

    if isinstance(x, (tuple, list)):
        if not x:
            raise TypeError("reward_fn returned empty tuple/list")
        return _reward_to_float(x[0])

    if isinstance(x, dict):
        for k in ("reward", "score", "value", "r"):
            if k in x:
                return _reward_to_float(x[k])
        raise TypeError(f"reward_fn returned dict without reward keys: {list(x.keys())}")

    raise TypeError(f"Unsupported reward type from reward_fn: {type(x)} ({x!r})")


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _strip_stop(text: str, stop: Sequence[str] | None) -> str:
    if not stop:
        return text.strip()
    cut: int | None = None
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx >= 0:
            cut = idx if cut is None else min(cut, idx)
    return (text[:cut] if cut is not None else text).strip()


def _as_policy_callable(policy: Any) -> Callable[[str, int, Mapping[str, Any]], list[str]]:
    """Normalize policy into: (prompt, n, decoding) -> list[str]."""
    if hasattr(policy, "sample") and callable(getattr(policy, "sample")):

        def _call(prompt: str, n: int, decoding: Mapping[str, Any]) -> list[str]:
            return list(policy.sample(prompt, n=n, **dict(decoding)))

        return _call

    if callable(policy):

        def _call(prompt: str, n: int, decoding: Mapping[str, Any]) -> list[str]:
            try:
                return list(policy(prompt, n, **dict(decoding)))
            except TypeError:
                try:
                    return list(policy(prompt=prompt, n=n, **dict(decoding)))
                except TypeError:
                    return list(policy(prompt=prompt, n=n, decoding=dict(decoding)))

        return _call

    raise TypeError("policy must be callable or expose .sample(prompt, n, **decoding)")


def _resolve_task_yaml(task: str) -> str:
    """Map 'math' -> .../configs/tasks/math.yaml; keep existing path if provided."""
    p = Path(task)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return str(p)

    here = Path(__file__).resolve()
    tasks_dir = here.parents[1] / "configs" / "tasks"
    for ext in ("yaml", "yml"):
        cand = tasks_dir / f"{task}.{ext}"
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(f"Cannot resolve task YAML for task={task!r}")


def _match_key_case_insensitive(items: Mapping[str, Any], key: str) -> tuple[str, Any]:
    lk = key.lower()
    for k, v in items.items():
        if k.lower() == lk:
            return k, v
    raise KeyError(f"{key!r} not found. Available: {list(items.keys())!r}")


def _ensure_tokenizer_padding(tokenizer: Any) -> Any:
    """Make tokenizer usable for left-padded batching."""
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    try:
        if getattr(tokenizer, "pad_token", None) is None:
            eos = getattr(tokenizer, "eos_token", None)
            if eos is not None:
                tokenizer.pad_token = eos
    except Exception:
        pass
    return tokenizer


def _sha1_u32(text: str) -> int:
    """Deterministic small salt derived from text (first 32 bits of sha1)."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def _unique_extend(dst: list[str], seen: set[str], items: Sequence[str], limit: int) -> None:
    """Append unique items into dst until limit."""
    for s in items:
        if s in seen:
            continue
        seen.add(s)
        dst.append(s)
        if len(dst) >= limit:
            break


# -----------------------------------------------------------------------------
# Public schema (JSONL-friendly)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RestartState:
    """A restartable prefix of a multi-agent episode."""
    question_id: str | int
    question: str
    roles_topo: list[str]
    role_outputs_prefix: dict[str, str]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Replay configuration for a single bucket (one context)."""
    target_role: str
    num_candidates: int
    num_completions_per_candidate: int
    decoding: dict[str, Any] = field(default_factory=dict)

    # Influence
    record_next_teammate: bool = False
    next_role: str | None = None

    # Fidelity alignment
    include_real_as_j0: bool = False
    num_extra_v_samples: int = 0

    # Prefix sampling for restart states (previous roles)
    prefix_decoding: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Results for one candidate action under a fixed context."""
    action_text: str
    returns: list[float]
    next_actions: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.returns)


@dataclass(frozen=True, slots=True)
class Bucket:
    """A bucket: fixed context + multiple candidate actions + rollout returns."""
    ctx_hash: int
    restart: RestartState
    candidates: list[CandidateResult]
    meta: dict[str, Any] = field(default_factory=dict)

    # Back-compat
    target_role: str | None = None

    @property
    def question_id(self) -> str | int:
        return self.restart.question_id

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("target_role") is None and "target_role" in d:
            d.pop("target_role")
        return d


# -----------------------------------------------------------------------------
# Minimal integration contracts
# -----------------------------------------------------------------------------


class PromptRenderer(Protocol):
    def render_role_prompt(
        self,
        *,
        question: str,
        roles_topo: Sequence[str],
        role_outputs: Mapping[str, str],
        target_role: str,
        meta: Mapping[str, Any] | None = None,
    ) -> str:
        ...


class SamplingPolicy(Protocol):
    def sample(self, prompt: str, n: int = 1, **decoding: Any) -> Sequence[str]:
        ...


class Evaluator(Protocol):
    def evaluate(
        self,
        *,
        restart: RestartState,
        role_outputs: Mapping[str, str],
        meta: Mapping[str, Any] | None = None,
    ) -> float:
        ...


# -----------------------------------------------------------------------------
# Core runner
# -----------------------------------------------------------------------------


class ReplayRunner:
    """Counterfactual replay runner."""

    _SEED_MOD: int = 2_147_483_647  # prime-ish, fits int32

    def __init__(
        self,
        *,
        task_spec: Any,
        policy: SamplingPolicy | Callable[..., Any],
        evaluator: Evaluator,
        prompt_renderer: PromptRenderer,
        runner_meta: Mapping[str, Any] | None = None,
        context_string_fn: Callable[[RestartState, str], str] | None = None,
        dataset: Any | None = None,
        eval_suite_name: str | None = None,
        roles_topo: Sequence[str] | None = None,
    ) -> None:
        self.task_spec = task_spec
        self._policy_obj = policy
        self._policy_call = _as_policy_callable(policy)
        self.evaluator = evaluator
        self.prompt_renderer = prompt_renderer
        self.runner_meta = dict(runner_meta or {})
        self._context_string_fn = context_string_fn

        self._dataset = dataset
        self._eval_suite_name = eval_suite_name
        self._roles_topo = list(roles_topo) if roles_topo is not None else None

    def close(self) -> None:
        # Best-effort cleanup: especially important for vLLM EngineCore shutdown.
        obj = getattr(self, "_policy_obj", None)
        if obj is None:
            return
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    # -------------------------
    # OpenRLHF integrated build
    # -------------------------

    @classmethod
    def from_cli(
        cls,
        *,
        task: str,
        split: str,
        policy_ckpt: str,
        # accepted for forward-compat with higher-level CLIs
        target_role: str | None = None,
        next_role: str | None = None,
        method: str | None = None,
        seed: int = 0,
        device: str | None = None,
        engine: str = "auto",
        tensor_parallel_size: int | None = None,
        trust_remote_code: bool = True,
        cache_dir: str | None = None,
        decode_defaults: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> "ReplayRunner":
        """Build a runner from OpenRLHF C3 task configs (used by build-buckets)."""
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from transformers import AutoTokenizer

        from c3.integration.marl_specs import load_task
        from c3.envs.registry import get_env_reward_fn
        from c3.integration.marl_specs import topo_sort_roles
        from c3.integration.task_datasets import load_task_datasets

        task_yaml = _resolve_task_yaml(task)
        task_spec = load_task(task_yaml)

        td = load_task_datasets(task_spec, cache_dir=cache_dir)
        suite_name, ds = _match_key_case_insensitive(td.evals, split)

        roles_topo_specs = topo_sort_roles(task_spec.roles)
        roles_topo = [_role_to_name(r) for r in roles_topo_specs]

        tokenizer = AutoTokenizer.from_pretrained(
            policy_ckpt,
            trust_remote_code=bool(trust_remote_code),
            use_fast=True,
        )
        tokenizer = _ensure_tokenizer_padding(tokenizer)

        policy = _build_sampling_policy(
            policy_ckpt,
            tokenizer=tokenizer,
            device=device,
            seed=seed,
            engine=engine,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
        )

        prompt_renderer = _OpenRLHFPromptRenderer(task_spec=task_spec, tokenizer=tokenizer)

        reward_fn = get_env_reward_fn(task_spec.env_name)
        evaluator = _OpenRLHFEvaluator(task_spec=task_spec, reward_fn=reward_fn, roles_topo=roles_topo)

        runner_meta = {
            "task_yaml": task_yaml,
            "task": getattr(task_spec, "experiment_name", None)
            or getattr(task_spec, "env_name", None)
            or getattr(task_spec, "name", None)
            or "task",
            "split": suite_name,
            "method": method,
            "seed": int(seed),
            "requested_target_role": (target_role.lower() if isinstance(target_role, str) else None),
            "requested_next_role": (next_role.lower() if isinstance(next_role, str) else None),
            "engine_requested": engine,
            "engine_resolved": getattr(policy, "engine", None),
            "tensor_parallel_size": tensor_parallel_size,
            "decode_defaults": dict(decode_defaults or {}),
        }

        def _ctx_string(restart: RestartState, role: str) -> str:
            return prompt_renderer.render_role_prompt(
                question=restart.question,
                roles_topo=restart.roles_topo,
                role_outputs=restart.role_outputs_prefix,
                target_role=role,
                meta=restart.meta,
            )

        return cls(
            task_spec=task_spec,
            policy=policy,
            evaluator=evaluator,
            prompt_renderer=prompt_renderer,
            runner_meta=runner_meta,
            context_string_fn=_ctx_string,
            dataset=ds,
            eval_suite_name=suite_name,
            roles_topo=roles_topo,
        )

    # -------------------------
    # Restart states
    # -------------------------

    @staticmethod
    def _canonical_role(role: str, roles_topo: Sequence[Any]) -> str:
        role_s = _role_to_name(role)
        for r in roles_topo:
            r_s = _role_to_name(r)
            if r_s.lower() == role_s.lower():
                return r_s
        avail = [_role_to_name(r) for r in roles_topo]
        raise ValueError(f"Unknown role={role!r}. Available roles: {avail!r}")

    def _iter_dataset(self, limit: int | None) -> Iterator[tuple[str | int, Mapping[str, Any]]]:
        if self._dataset is None:
            raise RuntimeError("Runner has no dataset. Build via ReplayRunner.from_cli or pass dataset=...")

        ds = self._dataset
        if hasattr(ds, "__len__") and hasattr(ds, "__getitem__"):
            n = len(ds)
            upto = n if limit is None else min(limit, n)
            for i in range(upto):
                item = ds[i]
                qid = item.get("id", i) if isinstance(item, Mapping) else i
                yield qid, item
            return

        k = 0
        for item in ds:  # type: ignore[assignment]
            qid = item.get("id", k) if isinstance(item, Mapping) else k
            yield qid, item
            k += 1
            if limit is not None and k >= limit:
                break

    def iter_restart_states(
        self,
        *,
        task: str,
        split: str,
        target_role: str,
        limit: int | None,
        seed: int | None = None,
        prefix_decoding: Mapping[str, Any] | None = None,
    ) -> Iterable[RestartState]:
        """Yield RestartState objects ready for `target_role`."""
        _ = task, split  # bound via from_cli()
        roles_topo = list(self._roles_topo or [])
        if not roles_topo:
            raise RuntimeError("roles_topo unavailable; cannot build restart states")

        target_role = self._canonical_role(target_role, roles_topo)

        dec = dict(self.runner_meta.get("decode_defaults") or {})
        if prefix_decoding:
            dec.update(dict(prefix_decoding))
        if seed is not None:
            dec.setdefault("seed", int(seed))

        for qid, item in self._iter_dataset(limit):
            question = str(item.get("input") or item.get("question") or "")
            label = item.get("answer")
            dataset_meta = item.get("meta") if isinstance(item, Mapping) else None

            role_outputs: dict[str, str] = {}
            for role in roles_topo:
                if role == target_role:
                    break
                prompt = self.prompt_renderer.render_role_prompt(
                    question=question,
                    roles_topo=roles_topo,
                    role_outputs=role_outputs,
                    target_role=role,
                    meta={"dataset_meta": dataset_meta},
                )
                outs = self.sample_action(prompt, dec, n=1)
                role_outputs[role] = outs[0] if outs else ""

            meta = {
                "label": label,
                "dataset_meta": dataset_meta,
                "task_env_cfg": getattr(self.task_spec, "environment", None) or {},
                "env_name": getattr(self.task_spec, "env_name", None),
                "eval_suite": self._eval_suite_name,
            }
            yield RestartState(
                question_id=qid,
                question=question,
                roles_topo=roles_topo,
                role_outputs_prefix=role_outputs,
                meta=meta,
            )

    # -------------------------
    # Bucket execution
    # -------------------------

    def build_context_hash(self, restart_state: RestartState, target_role: str) -> int:
        if self._context_string_fn is not None:
            ctx_text = str(self._context_string_fn(restart_state, target_role))
        else:
            hook = getattr(self.prompt_renderer, "build_context_string", None)
            if callable(hook):
                ctx_text = str(hook(restart_state=restart_state, target_role=target_role))
            else:
                payload = {
                    "question": restart_state.question,
                    "roles_topo": list(restart_state.roles_topo),
                    "target_role": target_role,
                    "role_outputs_prefix": dict(sorted(restart_state.role_outputs_prefix.items())),
                }
                ctx_text = _stable_json(payload)

        key = int(hash63(ctx_text))
        global_guard().observe(key, fingerprint(ctx_text), where="ReplayRunner/build_context_hash")
        return key

    def sample_action(self, prompt: str, decoding: Mapping[str, Any], n: int = 1) -> list[str]:
        return self._policy_call(prompt, n, decoding)

    def _decoding_with_seed(self, base: Mapping[str, Any], seed: int) -> dict[str, Any]:
        d = dict(base)
        d["seed"] = int(seed) % self._SEED_MOD
        return d

    def run_bucket(
        self,
        restart_state: RestartState,
        cfg: ReplayConfig,
        forced_actions: list[str] | None = None,
    ) -> Bucket:
        """Build one bucket for a fixed restart state."""
        self._validate_cfg(restart_state, cfg)
        ctx_hash = self.build_context_hash(restart_state, cfg.target_role)

        target_prompt = self.prompt_renderer.render_role_prompt(
            question=restart_state.question,
            roles_topo=restart_state.roles_topo,
            role_outputs=restart_state.role_outputs_prefix,
            target_role=cfg.target_role,
            meta=restart_state.meta,
        )

        # Base decoding (reproducible); per-attempt jitter avoids pathological resampling loops.
        decoding_base = dict(cfg.decoding)
        base_seed = decoding_base.get("seed", None)
        if base_seed is None:
            base_seed = int(self.runner_meta.get("seed") or 0)
            decoding_base["seed"] = base_seed
        else:
            base_seed = int(base_seed)

        bucket_salt = _sha1_u32(f"{ctx_hash}:{cfg.target_role}")

        def dec_at(k: int) -> dict[str, Any]:
            return self._decoding_with_seed(decoding_base, base_seed + bucket_salt + int(k))

        credit_n_req = max(0, int(cfg.num_candidates))
        extra_n_req = max(0, int(cfg.num_extra_v_samples))
        total_req = max(0, credit_n_req + extra_n_req)

        forced = list(forced_actions or [])
        candidates: list[str] = []
        seen: set[str] = set()

        _unique_extend(candidates, seen, forced, total_req if total_req > 0 else (len(forced) + 1))

        if cfg.include_real_as_j0 and total_req > 0 and not candidates:
            _unique_extend(candidates, seen, self.sample_action(target_prompt, dec_at(0), n=1), total_req)

        if total_req > len(candidates):
            attempts = 0
            max_attempts = min(30, max(5, 2 * total_req))  # hard bound; don't burn GPU
            while len(candidates) < total_req and attempts < max_attempts:
                need = total_req - len(candidates)
                batch = self.sample_action(target_prompt, dec_at(1 + attempts), n=need)
                _unique_extend(candidates, seen, batch, total_req)
                attempts += 1

        candidates = candidates[:total_req] if total_req > 0 else []
        credit_n = min(credit_n_req, len(candidates))
        extra_n = max(0, len(candidates) - credit_n)

        role_index = {r: i for i, r in enumerate(restart_state.roles_topo)}
        t_idx = role_index[cfg.target_role]
        roles_after = list(restart_state.roles_topo[t_idx + 1 :])

        cand_results: list[CandidateResult] = []
        batched_next = (
            bool(cfg.record_next_teammate)
            and cfg.next_role is not None
            and roles_after == [cfg.next_role]
        )

        for a_text in candidates:
            returns: list[float] = []
            next_actions: list[str] = []

            base_out = dict(restart_state.role_outputs_prefix)
            base_out[cfg.target_role] = a_text

            if batched_next:
                next_prompt = self.prompt_renderer.render_role_prompt(
                    question=restart_state.question,
                    roles_topo=restart_state.roles_topo,
                    role_outputs=base_out,
                    target_role=cfg.next_role,  # type: ignore[arg-type]
                    meta=restart_state.meta,
                )
                ys = self.sample_action(next_prompt, decoding_base, n=cfg.num_completions_per_candidate)
                for y in ys:
                    role_outputs = dict(base_out)
                    role_outputs[cfg.next_role] = y  # type: ignore[index]
                    r = float(self.evaluator.evaluate(restart=restart_state, role_outputs=role_outputs, meta=self.runner_meta))
                    returns.append(r)
                    next_actions.append(y)
            else:
                for _ in range(cfg.num_completions_per_candidate):
                    role_outputs = dict(base_out)
                    captured_next: str | None = None

                    for role in roles_after:
                        p = self.prompt_renderer.render_role_prompt(
                            question=restart_state.question,
                            roles_topo=restart_state.roles_topo,
                            role_outputs=role_outputs,
                            target_role=role,
                            meta=restart_state.meta,
                        )
                        y_list = self.sample_action(p, decoding_base, n=1)
                        y = y_list[0] if y_list else ""
                        role_outputs[role] = y
                        if cfg.record_next_teammate and cfg.next_role == role and captured_next is None:
                            captured_next = y

                    r = float(self.evaluator.evaluate(restart=restart_state, role_outputs=role_outputs, meta=self.runner_meta))
                    returns.append(r)
                    if captured_next is not None:
                        next_actions.append(captured_next)

            cand_results.append(CandidateResult(action_text=a_text, returns=returns, next_actions=next_actions))

        meta = dict(self.runner_meta)
        meta.update(
            {
                "task_spec": self._safe_task_id(),
                "decoding": dict(decoding_base),
                "seed": decoding_base.get("seed"),
                "record_next_teammate": cfg.record_next_teammate,
                "next_role": cfg.next_role,
                "include_real_as_j0": cfg.include_real_as_j0,
                "real_j": 0 if (cfg.include_real_as_j0 and candidates) else None,
                "credit_n": credit_n,
                "v_extra_start": credit_n,
                "v_extra_n": extra_n,
                "candidate_total_req": total_req,
                "candidate_total_got": len(candidates),
            }
        )
        if meta.get("real_j") is None:
            meta.pop("real_j", None)

        return Bucket(
            ctx_hash=ctx_hash,
            restart=restart_state,
            candidates=cand_results,
            meta=meta,
            target_role=cfg.target_role,
        )

    # -------------------------
    # Validation / bookkeeping
    # -------------------------

    def _validate_cfg(self, restart_state: RestartState, cfg: ReplayConfig) -> None:
        if cfg.num_candidates <= 0:
            raise ValueError("ReplayConfig.num_candidates must be > 0")
        if cfg.num_completions_per_candidate <= 0:
            raise ValueError("ReplayConfig.num_completions_per_candidate must be > 0")
        if cfg.target_role not in restart_state.roles_topo:
            raise ValueError(f"target_role={cfg.target_role!r} not in roles_topo={restart_state.roles_topo!r}")

        if cfg.record_next_teammate:
            if not cfg.next_role:
                raise ValueError("record_next_teammate=True requires next_role")
            role_index = {r: i for i, r in enumerate(restart_state.roles_topo)}
            if cfg.next_role not in role_index:
                raise ValueError(f"next_role={cfg.next_role!r} not in roles_topo={restart_state.roles_topo!r}")
            if role_index[cfg.next_role] <= role_index[cfg.target_role]:
                raise ValueError("next_role must be strictly after target_role in roles_topo")

        role_index = {r: i for i, r in enumerate(restart_state.roles_topo)}
        t_idx = role_index[cfg.target_role]
        for prev_role in restart_state.roles_topo[:t_idx]:
            if prev_role not in restart_state.role_outputs_prefix:
                raise ValueError(
                    f"restart_state.role_outputs_prefix missing previous role {prev_role!r} "
                    f"before target_role={cfg.target_role!r}"
                )

    def _safe_task_id(self) -> str:
        ts = self.task_spec
        try:
            if isinstance(ts, str):
                return ts
            if isinstance(ts, Mapping):
                return str(ts.get("experiment_name") or ts.get("name") or ts.get("env_name") or ts.get("id") or "task")
            return (
                getattr(ts, "experiment_name", None)
                or getattr(ts, "name", None)
                or getattr(ts, "env_name", None)
                or getattr(ts, "id", None)
                or "task"
            )
        except Exception:
            return "task"


# -----------------------------------------------------------------------------
# OpenRLHF prompt/eval adapters (lazy-imported by from_cli)
# -----------------------------------------------------------------------------


class _OpenRLHFPromptRenderer:
    def __init__(self, *, task_spec: Any, tokenizer: Any) -> None:
        self.task_spec = task_spec
        self.tokenizer = tokenizer

        from c3.mas.prompt_render import build_render_context, render_role_prompt
        from c3.mas.rollout_generator import _compose_full_prompt_chat

        self._build_render_context = build_render_context
        self._render_role_prompt = render_role_prompt
        self._compose = _compose_full_prompt_chat

    def render_role_prompt(
        self,
        *,
        question: str,
        roles_topo: Sequence[str],
        role_outputs: Mapping[str, str],
        target_role: str,
        meta: Mapping[str, Any] | None = None,
    ) -> str:
        role_spec = next((r for r in self.task_spec.roles if r.name == target_role), None)
        if role_spec is None:
            role_spec = next((r for r in self.task_spec.roles if r.name.lower() == target_role.lower()), None)
        if role_spec is None:
            raise ValueError(f"Unknown target_role={target_role!r}")

        roles_topo_names = [_role_to_name(r) for r in list(roles_topo)]
        role_index = {r: i for i, r in enumerate(roles_topo_names)}
        if target_role not in role_index:
            for r in roles_topo_names:
                if r.lower() == target_role.lower():
                    target_role = r
                    break

        t_idx = role_index.get(target_role, 0)
        topo_so_far = list(roles_topo_names[:t_idx])

        ctx = self._build_render_context(
            question=question,
            role_outputs=dict(role_outputs),
            topo_so_far=topo_so_far,
        )

        for k, v in dict(meta or {}).items():
            if v is None:
                continue
            try:
                ctx[str(k)] = str(v)
            except Exception:
                pass

        system_prompt = self._render_role_prompt(role_spec.prompt, ctx=ctx)
        return self._compose(
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            question=question,
            context=ctx.get("context", ""),
        )


class _OpenRLHFEvaluator:
    def __init__(self, *, task_spec: Any, reward_fn: Any, roles_topo: Sequence[str]) -> None:
        self.task_spec = task_spec
        self.reward_fn = reward_fn
        self.roles_topo = [_role_to_name(r) for r in list(roles_topo)]
        env = getattr(task_spec, "environment", None) or {}
        self.answer_role = env.get("answer_role") or self.roles_topo[-1]

    def evaluate(
        self, *, restart: RestartState, role_outputs: Mapping[str, str], meta: Mapping[str, Any] | None = None
    ) -> float:
        m = dict(meta or {})
        m.update(dict(restart.meta or {}))
        m.setdefault("task_env_cfg", getattr(self.task_spec, "environment", None) or {})

        prediction = str(role_outputs.get(self.answer_role, "")).strip()
        label = m.get("label")
        raw = self.reward_fn(prediction=prediction, label=label, meta=m)

        if isinstance(raw, (tuple, list)) and len(raw) >= 2:
            try:
                if isinstance(m, dict):
                    m.setdefault("reward_extra", raw[1])
            except Exception:
                pass

        return _reward_to_float(raw)


# -----------------------------------------------------------------------------
# Sampling policy adapters (HF / vLLM)
# -----------------------------------------------------------------------------


def _build_sampling_policy(
    model_path: str,
    *,
    tokenizer: Any,
    device: str | None,
    seed: int,
    engine: str,
    tensor_parallel_size: int | None = None,
    trust_remote_code: bool = True,
) -> SamplingPolicy:
    """
    engine:
      - vllm : require vLLM (fail fast)
      - hf   : require HF
      - auto : prefer vLLM; by default fail fast if vLLM init fails
              (set C3_ALLOW_HF_FALLBACK=1 to allow fallback)
    """
    engine = (engine or "auto").lower()
    if engine not in {"auto", "hf", "vllm"}:
        raise ValueError("engine must be one of: auto, hf, vllm")

    allow_fallback = os.environ.get("C3_ALLOW_HF_FALLBACK", "0").strip() in {"1", "true", "True"}

    if engine in {"auto", "vllm"}:
        try:
            import vllm  # noqa: F401

            return _VLLMPolicy(
                model_path,
                tokenizer=tokenizer,
                seed=seed,
                trust_remote_code=trust_remote_code,
                tensor_parallel_size=tensor_parallel_size,
            )
        except Exception as e:
            if engine == "vllm":
                raise
            if not allow_fallback:
                raise RuntimeError(
                    "[C3][FAIL-FAST] vLLM initialization failed under engine=auto. "
                    "Pass --engine hf to allow HF, or set C3_ALLOW_HF_FALLBACK=1.\n"
                    f"Underlying error: {type(e).__name__}: {e}"
                ) from e

    return _HFPolicy(model_path, tokenizer=tokenizer, device=device, seed=seed, trust_remote_code=trust_remote_code)


class _HFPolicy:
    engine: str = "hf"

    def __init__(
        self, model_path: str, *, tokenizer: Any, device: str | None, seed: int, trust_remote_code: bool
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(seed)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=bool(trust_remote_code),
            torch_dtype=(torch.bfloat16 if self.device.startswith("cuda") else None),
        )
        self.model.to(self.device)
        self.model.eval()

        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def sample(self, prompt: str, n: int = 1, **decoding: Any) -> Sequence[str]:
        import torch

        temperature = float(decoding.get("temperature", 0.7))
        top_p = float(decoding.get("top_p", 0.95))
        top_k = int(decoding.get("top_k", 0))
        max_new_tokens = int(decoding.get("max_new_tokens", 512))
        stop = decoding.get("stop")
        if isinstance(stop, str):
            stop = [stop]

        seed = decoding.get("seed")
        seed = int(seed) if seed is not None else self.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attn = inputs.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)

        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            num_return_sequences=int(n),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )

        start = input_ids.shape[1]
        outs: list[str] = []
        for seq in gen:
            text = self.tokenizer.decode(seq[start:], skip_special_tokens=True)
            outs.append(_strip_stop(text, stop))
        return outs


def _filter_kwargs_by_signature(fn: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Best-effort filter for cross-version compatibility."""
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_varkw:
            return dict(kwargs)
        allowed = set(params.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)


def _env_int(name: str, default: int) -> int:
    try:
        v = str(os.environ.get(name, "")).strip()
        return default if v == "" else int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        v = str(os.environ.get(name, "")).strip()
        return default if v == "" else float(v)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    try:
        v = str(os.environ.get(name, "")).strip().lower()
        if v in ("1", "true", "t", "yes", "y", "on"):
            return True
        if v in ("0", "false", "f", "no", "n", "off"):
            return False
        return default
    except Exception:
        return default


class _VLLMPolicy:
    engine: str = "vllm"

    def __init__(
        self,
        model_path: str,
        *,
        tokenizer: Any,
        seed: int,
        trust_remote_code: bool,
        tensor_parallel_size: int | None = None,
    ) -> None:
        from vllm import LLM, SamplingParams  # local import for optional dep

        self.SamplingParams = SamplingParams
        self.tokenizer = tokenizer
        self.seed = int(seed)

        # Resolve TP size.
        tp: int | None
        try:
            tp = int(tensor_parallel_size) if tensor_parallel_size is not None else None
        except Exception:
            tp = None

        if tp is None or tp <= 0:
            try:
                import torch

                tp = int(torch.cuda.device_count() or 1)
            except Exception:
                tp = 1

        try:
            import torch

            tp = min(int(tp), int(torch.cuda.device_count() or 1))
        except Exception:
            tp = int(tp) if tp is not None else 1

        try:
            import torch

            dtype: Any = torch.bfloat16
        except Exception:
            dtype = "bfloat16"

        # IMPORTANT: vLLM defaults can over-allocate KV cache based on the model's
        # max context length (often 32k+), which is a common cause of EngineCore
        # crashes / OOM when running many parallel jobs. We set safer defaults.
        # You can override these via environment variables.
        max_model_len = _env_int("C3_VLLM_MAX_MODEL_LEN", 8192)
        gpu_mem_util = _env_float("C3_VLLM_GPU_MEMORY_UTILIZATION", 0.75)
        max_num_seqs = _env_int("C3_VLLM_MAX_NUM_SEQS", 32)
        swap_space_gb = _env_int("C3_VLLM_SWAP_SPACE_GB", 2)
        enforce_eager = _env_bool("C3_VLLM_ENFORCE_EAGER", False)
        disable_log_stats = _env_bool("C3_VLLM_DISABLE_LOG_STATS", True)

        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": bool(trust_remote_code),
            "tensor_parallel_size": int(tp),
            "dtype": dtype,
        }
        llm_kwargs.update(
            {
                "max_model_len": int(max_model_len),
                "gpu_memory_utilization": float(gpu_mem_util),
                "max_num_seqs": int(max_num_seqs),
                "swap_space": int(swap_space_gb),
                "enforce_eager": bool(enforce_eager),
                "disable_log_stats": bool(disable_log_stats),
            }
        )

        if int(tp) > 1:
            llm_kwargs["distributed_executor_backend"] = "mp"

        llm_kwargs = _filter_kwargs_by_signature(LLM.__init__, llm_kwargs)
        self.llm = LLM(**llm_kwargs)

    def sample(self, prompt: str, n: int = 1, **decoding: Any) -> Sequence[str]:
        SamplingParams = self.SamplingParams

        temperature = float(decoding.get("temperature", 0.7))
        top_p = float(decoding.get("top_p", 0.95))
        top_k = int(decoding.get("top_k", -1))
        max_tokens = int(decoding.get("max_new_tokens", 512))
        stop = decoding.get("stop")
        if isinstance(stop, str):
            stop = [stop]

        seed = decoding.get("seed")
        seed = int(seed) if seed is not None else self.seed

        params = SamplingParams(
            n=int(n),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=list(stop) if stop else None,
            seed=seed,
        )
        out = self.llm.generate([prompt], params)
        return [_strip_stop(o.text, stop) for o in out[0].outputs]

    def close(self) -> None:
        # Best-effort shutdown for vLLM to avoid EngineCore errors at process exit.
        llm = getattr(self, "llm", None)
        if llm is None:
            return
        for name in ("shutdown", "close"):
            fn = getattr(llm, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        self.llm = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
