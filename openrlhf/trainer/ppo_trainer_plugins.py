# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

import atexit
import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ray
import torch

from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.run_metadata import build_wandb_config, build_wandb_tags

logger = init_logger(__name__)

# =============================================================================
# JSONL buffered dumping (best-effort, low overhead)
# =============================================================================

# Per-process dedupe: avoid accidental duplicate rows (e.g., retries).
_JSONL_DEDUPE: "OrderedDict[str, None]" = OrderedDict()
_JSONL_DEDUPE_MAX = int(os.environ.get("OPENRLHF_JSONL_DEDUPE_MAX", "100000"))

# Per-process append buffers: reduce syscall overhead.
_JSONL_BUFFERS: Dict[str, Dict[str, object]] = {}
_JSONL_BUFFER_LOCK = threading.Lock()


def _jsonl_buffer_cfg() -> tuple[int, int, float]:
    """(max_lines, max_bytes, flush_sec)."""
    max_lines = int(os.environ.get("OPENRLHF_JSONL_BUFFER_LINES", "64"))
    max_bytes = int(os.environ.get("OPENRLHF_JSONL_BUFFER_BYTES", str(1 << 20)))
    flush_sec = float(os.environ.get("OPENRLHF_JSONL_FLUSH_SEC", "2.0"))
    return max_lines, max_bytes, flush_sec


def flush_jsonl_buffers(path: str | None = None) -> None:
    """Flush buffered JSONL rows to disk (best-effort)."""
    try:
        now = time.time()
        with _JSONL_BUFFER_LOCK:
            items = list(_JSONL_BUFFERS.items()) if path is None else [(path, _JSONL_BUFFERS.get(path))]
            for p, buf in items:
                if not buf:
                    continue
                lines = buf.get("lines") or []
                if not lines:
                    continue
                Path(p).parent.mkdir(parents=True, exist_ok=True)
                with Path(p).open("a", encoding="utf-8") as f:
                    f.writelines(lines)
                buf["lines"] = []
                buf["bytes"] = 0
                buf["t"] = float(now)
    except Exception:
        pass


# Flush on normal interpreter exit (Ctrl-C may bypass).
atexit.register(flush_jsonl_buffers)


def _append_jsonl(path: str, obj: dict, *, mode: str = "append", dedupe_key=None) -> None:
    """Append (or overwrite) one JSONL row. Buffered by default."""
    if not path:
        return

    # Dedupe (best-effort).
    if dedupe_key is not None:
        try:
            if dedupe_key in _JSONL_DEDUPE:
                return
            _JSONL_DEDUPE[dedupe_key] = None
            if len(_JSONL_DEDUPE) > _JSONL_DEDUPE_MAX:
                _JSONL_DEDUPE.popitem(last=False)
        except Exception:
            pass

    p = str(path)
    m = str(mode).lower().strip()
    line = json.dumps(obj, ensure_ascii=False) + "\n"

    # Overwrite forces a flush (so we don't lose buffered lines).
    if m == "overwrite":
        flush_jsonl_buffers(p)
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text(line, encoding="utf-8")
        return

    max_lines, max_bytes, flush_sec = _jsonl_buffer_cfg()
    now = time.time()

    should_flush = False
    with _JSONL_BUFFER_LOCK:
        buf = _JSONL_BUFFERS.get(p)
        if buf is None:
            buf = {"lines": [], "bytes": 0, "t": float(now)}
            _JSONL_BUFFERS[p] = buf

        buf["lines"].append(line)
        buf["bytes"] = int(buf.get("bytes", 0)) + len(line.encode("utf-8", errors="ignore"))
        last_flush_t = float(buf.get("t", now))

        if len(buf["lines"]) >= max_lines:
            should_flush = True
        elif int(buf["bytes"]) >= max_bytes:
            should_flush = True
        elif (now - last_flush_t) >= flush_sec:
            should_flush = True

        if should_flush:
            # Mark flush time eagerly to reduce stampede.
            buf["t"] = float(now)

    if should_flush:
        flush_jsonl_buffers(p)


def _maybe_reset_file(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")


def _with_suffix(path: str, suffix: str) -> str:
    """Append suffix to filename (not extension)."""
    try:
        p = Path(path)
        return str(p.with_name(p.name + str(suffix)))
    except Exception:
        return str(path) + str(suffix)


# =============================================================================
# Small helpers (keep behavior stable across strategy variants)
# =============================================================================


def _is_rank0(strategy) -> bool:
    """Best-effort rank0 detection across strategy variants."""
    try:
        v = getattr(strategy, "is_rank_0", None)
        if callable(v):
            return bool(v())
        if isinstance(v, bool):
            return v
    except Exception:
        pass
    r = getattr(strategy, "global_rank", getattr(strategy, "rank", 0))
    try:
        return int(r) == 0
    except Exception:
        return True


def _safe_print(strategy, msg: str) -> None:
    try:
        if strategy is not None and hasattr(strategy, "print"):
            strategy.print(msg)
            return
    except Exception:
        pass
    print(msg)


def _first(v, default=None):
    if isinstance(v, list):
        return v[0] if v else default
    return v if v is not None else default


def _tensor_first_scalar(x, default=None):
    """Extract first scalar from tensor/list/scalar."""
    try:
        if isinstance(x, torch.Tensor):
            t = x.detach().flatten()
            return t[0].item() if t.numel() else default
    except Exception:
        pass
    try:
        if isinstance(x, (list, tuple)) and x:
            return _tensor_first_scalar(x[0], default=default)
    except Exception:
        pass
    return x if x is not None else default


def _tensor1_int(x, default: int = -1) -> int:
    try:
        return int(_tensor_first_scalar(x, default=default))
    except Exception:
        return int(default)


def _tensor1_float(x, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        v = _tensor_first_scalar(x, default=default)
        return float(v) if v is not None else default
    except Exception:
        return default


def _sample_role_name(sample) -> Optional[str]:
    info = getattr(sample, "info", None) or {}
    r = _first(info.get("role", None), None)
    return r if isinstance(r, str) else None


# =============================================================================
# JSON-safe dumping (summarize large structures)
# =============================================================================


def _to_jsonable(
    x,
    *,
    max_tensor_elems: int | None = None,
    max_list_elems: int | None = None,
    max_str_len: int | None = None,
    _depth: int = 0,
) -> object:
    """Convert python/torch objects into compact JSONable forms (for dumps only)."""
    if max_tensor_elems is None:
        max_tensor_elems = int(os.environ.get("OPENRLHF_DUMP_MAX_TENSOR_ELEMS", "64"))
    if max_list_elems is None:
        max_list_elems = int(os.environ.get("OPENRLHF_DUMP_MAX_LIST_ELEMS", "64"))
    if max_str_len is None:
        max_str_len = int(os.environ.get("OPENRLHF_DUMP_MAX_STR_LEN", "4096"))

    if _depth > 8:
        return {"__truncated__": True, "reason": "max_depth"}

    if x is None or isinstance(x, (bool, int, float)):
        return x

    if isinstance(x, str):
        if len(x) <= max_str_len:
            return x
        return {"__str__": True, "len": len(x), "head": x[:max_str_len]}

    if isinstance(x, (bytes, bytearray)):
        return {"__bytes__": True, "len": len(x)}

    if torch.is_tensor(x):
        try:
            t = x.detach().cpu()
            numel = int(t.numel())
            if numel == 0:
                return {"__tensor__": True, "dtype": str(t.dtype), "shape": list(t.shape), "numel": 0}
            if numel == 1:
                return t.flatten()[0].item()
            if numel <= max_tensor_elems:
                return t.flatten().tolist()
            head_n = min(16, numel)
            return {
                "__tensor__": True,
                "dtype": str(t.dtype),
                "shape": list(t.shape),
                "numel": numel,
                "head": t.flatten()[:head_n].tolist(),
            }
        except Exception:
            return {"__tensor__": True, "dtype": str(getattr(x, "dtype", "?")), "shape": list(getattr(x, "shape", []))}

    if isinstance(x, dict):
        out = {}
        if len(x) > 256:
            items = list(x.items())[:256]
            out["__dict_truncated__"] = True
            out["__dict_len__"] = len(x)
        else:
            items = x.items()
        for k, v in items:
            out[str(k)] = _to_jsonable(
                v,
                max_tensor_elems=max_tensor_elems,
                max_list_elems=max_list_elems,
                max_str_len=max_str_len,
                _depth=_depth + 1,
            )
        return out

    if isinstance(x, (list, tuple)):
        n = len(x)

        # Numeric sequences: summarize if huge.
        if n > max_list_elems and n > 0 and all(isinstance(v, (bool, int, float)) for v in x):
            xs = [float(v) for v in x]
            mn, mx = min(xs), max(xs)
            mean = sum(xs) / max(n, 1)
            if all(isinstance(v, bool) for v in x):
                return {"__seq__": True, "len": n, "true_frac": mean}
            return {"__seq__": True, "len": n, "min": mn, "max": mx, "mean": mean}

        # Otherwise truncate.
        if n > max_list_elems:
            head_n = min(16, max_list_elems)
            tail_n = min(4, max_list_elems - head_n)
            head = [
                _to_jsonable(
                    v,
                    max_tensor_elems=max_tensor_elems,
                    max_list_elems=max_list_elems,
                    max_str_len=max_str_len,
                    _depth=_depth + 1,
                )
                for v in x[:head_n]
            ]
            tail = (
                [
                    _to_jsonable(
                        v,
                        max_tensor_elems=max_tensor_elems,
                        max_list_elems=max_list_elems,
                        max_str_len=max_str_len,
                        _depth=_depth + 1,
                    )
                    for v in x[-tail_n:]
                ]
                if tail_n
                else []
            )
            return {"__list__": True, "len": n, "head": head, "tail": tail}

        return [
            _to_jsonable(
                v,
                max_tensor_elems=max_tensor_elems,
                max_list_elems=max_list_elems,
                max_str_len=max_str_len,
                _depth=_depth + 1,
            )
            for v in x
        ]

    return {"__repr__": repr(x)}


# =============================================================================
# Arg / batch helpers
# =============================================================================


def _normalize_use_wandb_arg(use_wandb_arg) -> Tuple[bool, str, Optional[str]]:
    """Returns (enable_wandb, mode, key). mode: auto|require|key."""
    if isinstance(use_wandb_arg, bool):
        return bool(use_wandb_arg), ("auto" if use_wandb_arg else "key"), None

    if isinstance(use_wandb_arg, str):
        s = use_wandb_arg.strip()
        if s == "" or s.lower() in ("false", "0", "off", "no", "none", "null"):
            return False, "key", None
        if s.lower() == "auto":
            return True, "auto", None
        if s.lower() in ("true", "1", "yes", "on"):
            return True, "require", None
        return True, "key", s

    enable = bool(use_wandb_arg)
    return enable, ("auto" if enable else "key"), None


def _unpack_prompt_batch(batch, where: str):
    """
    Expected:
      (datasources, prompts, labels) or (datasources, prompts, labels, meta_jsons)
    """
    if len(batch) == 3:
        datasources, prompts, labels = batch
        return datasources, prompts, labels, None
    if len(batch) == 4:
        datasources, prompts, labels, meta_jsons = batch
        return datasources, prompts, labels, meta_jsons
    raise ValueError(
        f"Unexpected batch size={len(batch)} during {where}. Expected 3 or 4 items: "
        "(datasources, prompts, labels[, meta_jsons])"
    )


def _as_list_bool(v, B: int) -> List[bool]:
    if isinstance(v, list):
        if len(v) == B:
            return [bool(x) for x in v]
        return ([bool(v[0])] * B) if v else ([False] * B)
    if torch.is_tensor(v):
        t = v.detach().cpu().flatten()
        if t.numel() == 0:
            return [False] * B
        if t.numel() == 1:
            return [bool(t.item())] * B
        arr = [bool(x) for x in t.tolist()]
        if len(arr) < B:
            arr = (arr + [arr[-1]] * B)[:B]
        return arr[:B]
    if v is None:
        return [False] * B
    return [bool(v)] * B


def _as_list_int(v, B: int, default: int = 0) -> List[int]:
    if isinstance(v, list):
        if len(v) == B:
            out = []
            for x in v:
                try:
                    out.append(int(x))
                except Exception:
                    out.append(int(default))
            return out
        return ([int(v[0])] * B) if v else ([int(default)] * B)
    if torch.is_tensor(v):
        t = v.detach().cpu().flatten()
        if t.numel() == 0:
            return [int(default)] * B
        if t.numel() == 1:
            return [int(t.item())] * B
        arr = []
        for x in t.tolist():
            try:
                arr.append(int(x))
            except Exception:
                arr.append(int(default))
        if len(arr) < B:
            arr = (arr + [arr[-1]] * B)[:B]
        return arr[:B]
    if v is None:
        return [int(default)] * B
    try:
        return [int(v)] * B
    except Exception:
        return [int(default)] * B


def _as_list_str(v, B: int, fill: str = "") -> List[str]:
    if isinstance(v, list):
        if len(v) == B:
            return [str(x) if x is not None else fill for x in v]
        return ([str(v[0])] * B) if v else ([fill] * B)
    if v is None:
        return [fill] * B
    return [str(v)] * B


# =============================================================================
# Plugins / mixin: optional trainer parts
# =============================================================================


class PPOTrainerPluginsMixin:
    """Tooling-style mixin used by PPOTrainer (logging, dumping, syncing, eval)."""

    # -------------------------------------------------------------------------
    # Run metadata
    # -------------------------------------------------------------------------

    def _with_run_context(self, obj: dict) -> dict:
        out = dict(obj)
        try:
            args = self.strategy.args
            for k in ("run_id", "run_dir", "wandb_run_name", "wandb_project", "c3_task"):
                v = getattr(args, k, None)
                if v is not None and k not in out:
                    out[k] = str(v)

            eval_only = bool(getattr(args, "eval_only", False))
            out.setdefault("eval_only", eval_only)

            egs = getattr(args, "eval_global_step", None)
            if egs is not None and "eval_global_step" not in out:
                try:
                    out["eval_global_step"] = int(egs)
                except Exception:
                    out["eval_global_step"] = str(egs)

            if eval_only and "global_step" not in out and egs is not None:
                try:
                    out["global_step"] = int(egs)
                except Exception:
                    out["global_step"] = str(egs)
        except Exception:
            pass
        return out

    # -------------------------------------------------------------------------
    # vLLM helpers
    # -------------------------------------------------------------------------

    def _vllm_batch_call(self, engines, method_name: str) -> None:
        if not engines:
            return
        from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

        batch_vllm_engine_call(engines, method_name)

    def _maybe_vllm_wake(self, engines) -> None:
        if bool(getattr(self.strategy.args, "vllm_enable_sleep", False)) and engines:
            self._vllm_batch_call(engines, "wake_up")

    def _maybe_vllm_sleep(self, engines) -> None:
        if bool(getattr(self.strategy.args, "vllm_enable_sleep", False)) and engines:
            self._vllm_batch_call(engines, "sleep")

    def _maybe_reload_states(self, actor_group) -> None:
        if bool(getattr(self.strategy.args, "deepspeed_enable_sleep", False)) and actor_group is not None:
            ray.get(actor_group.async_run_method(method_name="reload_states"))

    def _maybe_offload_states(self, actor_group) -> None:
        if bool(getattr(self.strategy.args, "deepspeed_enable_sleep", False)) and actor_group is not None:
            ray.get(actor_group.async_run_method(method_name="offload_states"))

    # -------------------------------------------------------------------------
    # Actor train + vLLM sync
    # -------------------------------------------------------------------------

    def ppo_train(self, global_steps: int) -> Dict[str, object]:
        status: Dict[str, object] = {}

        # per_role
        if getattr(self, "actor_model_groups", None) is not None:
            role_stats = {}
            for rn in self.role_names:
                self._maybe_reload_states(self.actor_model_groups[rn])
                ref = self.actor_model_groups[rn].async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value)
                role_stats[rn] = ray.get(ref)[0]
                self._maybe_offload_states(self.actor_model_groups[rn])

                if getattr(self, "vllm_engines_by_role", None) is not None:
                    self._broadcast_to_vllm_one_role(rn)

            all_kls = []
            for rn, st in role_stats.items():
                for k, v in st.items():
                    status[f"{rn}/{k}"] = v
                if "kl" in st:
                    try:
                        all_kls.append(float(st["kl"]))
                    except Exception:
                        pass
            if all_kls:
                status["kl"] = sum(all_kls) / max(len(all_kls), 1)
            return status

        # shared
        self._maybe_reload_states(self.actor_model_group)
        ref = self.actor_model_group.async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value)
        status.update(ray.get(ref)[0])
        self._maybe_offload_states(self.actor_model_group)

        if getattr(self, "vllm_engines", None) is not None:
            self._broadcast_to_vllm()
        return status

    def _broadcast_to_vllm(self):
        self._maybe_vllm_wake(self.vllm_engines)
        self.policy_version += 1
        ray.get(
            self.actor_model_group.async_run_method(method_name="broadcast_to_vllm", weights_version=self.policy_version)
        )
        self._maybe_vllm_sleep(self.vllm_engines)

    def _broadcast_to_vllm_one_role(self, role_name: str):
        if self.vllm_engines_by_role is None:
            return
        engines = self.vllm_engines_by_role.get(role_name)
        if not engines:
            raise RuntimeError(f"per_role mode: missing vllm engines for role={role_name!r}")

        self._maybe_vllm_wake(engines)
        self.policy_versions_by_role[role_name] += 1
        ray.get(
            self.actor_model_groups[role_name].async_run_method(
                method_name="broadcast_to_vllm",
                weights_version=int(self.policy_versions_by_role[role_name]),
            )
        )
        self._maybe_vllm_sleep(engines)

    def _sync_actor_weights_to_vllm(self):
        """Force-sync current actor weights into vLLM engines before rollouts/eval."""
        if getattr(self, "actor_model_groups", None) is not None:
            if getattr(self, "vllm_engines_by_role", None) is not None:
                for rn in self.role_names:
                    self._broadcast_to_vllm_one_role(rn)
        else:
            if getattr(self, "vllm_engines", None) is not None:
                self._broadcast_to_vllm()

    def _strict_check_vllm_before_rollout(self):
        if not bool(getattr(self.args, "strict_weights_version_check", False)):
            return

        # shared
        if getattr(self, "actor_model_groups", None) is None:
            if not self.vllm_engines:
                return
            engines = self.vllm_engines
            expected = int(self.policy_version)

            self._maybe_vllm_wake(engines)
            try:
                got = ray.get([e.get_weights_version.remote() for e in engines])
                if any(int(v) != expected for v in got):
                    raise RuntimeError(
                        f"[shared] vLLM weights_version mismatch before rollout: expected={expected}, got={got}"
                    )
            finally:
                self._maybe_vllm_sleep(engines)
            return

        # per_role
        if not self.vllm_engines_by_role:
            return
        for rn in self.role_names:
            engines = self.vllm_engines_by_role.get(rn) or []
            if not engines:
                continue
            expected = int(self.policy_versions_by_role.get(rn, 0))

            self._maybe_vllm_wake(engines)
            try:
                got = ray.get([e.get_weights_version.remote() for e in engines])
                if any(int(v) != expected for v in got):
                    raise RuntimeError(
                        f"[role={rn}] vLLM weights_version mismatch before rollout: expected={expected}, got={got}"
                    )
            finally:
                self._maybe_vllm_sleep(engines)

    # -------------------------------------------------------------------------
    # Metrics JSONL (rank0)
    # -------------------------------------------------------------------------

    def _maybe_dump_train_metrics_jsonl(self, global_step: int, logs_dict: dict, client_states: dict) -> None:
        """Append buffered train metrics JSONL (rank0 only)."""
        if not getattr(self, "strategy", None) or not _is_rank0(self.strategy):
            return

        path = getattr(self.args, "train_metrics_jsonl_path", None)
        if not path or (isinstance(path, str) and path.strip() == ""):
            return

        try:
            every = int(getattr(self.args, "train_metrics_every", 0) or 0)
        except Exception:
            every = 0
        if every <= 0 or (int(global_step) % every) != 0:
            return

        dump_client_states = bool(
            getattr(self.args, "dump_client_states", False)
            or str(os.environ.get("OPENRLHF_DUMP_CLIENT_STATES", "0")).strip() == "1"
        )
        dump_q_critic_status = bool(
            getattr(self.args, "dump_q_critic_status", False)
            or str(os.environ.get("OPENRLHF_DUMP_Q_CRITIC_STATUS", "0")).strip() == "1"
        )

        def _compact_metrics(d: dict) -> dict:
            out = {}
            for k, v in (d or {}).items():
                if isinstance(v, (int, float, bool)):
                    out[k] = v
                elif isinstance(v, str):
                    out[k] = v.strip()[:256]
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    out[k] = float(v.detach().cpu().item())
            return out

        payload = self._with_run_context(
            {
                "kind": "train_metrics",
                "ts": time.time(),
                "global_step": int(global_step),
                "marl_algorithm": getattr(self.args, "marl_algorithm", None),
                "policy_sharing_mode": getattr(self.args, "policy_sharing_mode", None),
                "q_critic_status": _to_jsonable(getattr(self, "_last_q_critic_status", None)) if dump_q_critic_status else None,
                "client_states": _to_jsonable(client_states or {}) if dump_client_states else None,
                "metrics": _compact_metrics(logs_dict),
            }
        )

        try:
            _append_jsonl(str(path), payload, mode="append", dedupe_key=("train_metrics", int(global_step)))
        except Exception as e:
            logger.warning(f"Failed to append train metrics JSONL to {path}: {e}")

    def _maybe_dump_eval_metrics_jsonl(
        self,
        *,
        global_step: int,
        logs: dict,
        duration_sec: float,
        temperature: float,
        n_samples_per_prompt: int,
        eval_dump_path: Optional[str] = None,
    ) -> None:
        """Append buffered eval metrics JSONL (rank0 only)."""
        if not getattr(self, "strategy", None) or not _is_rank0(self.strategy):
            return

        path = getattr(self.args, "eval_metrics_jsonl_path", None)
        if not path or (isinstance(path, str) and path.strip() == ""):
            if eval_dump_path:
                path = _with_suffix(str(eval_dump_path), ".metrics.jsonl")
            else:
                return

        try:
            if (
                int(global_step) == 0
                and str(getattr(self.args, "eval_metrics_jsonl_mode", "append") or "append").lower().strip() == "overwrite"
            ):
                _maybe_reset_file(str(path))
        except Exception:
            pass

        payload = self._with_run_context(
            {
                "kind": "eval_metrics",
                "ts": time.time(),
                "global_step": int(global_step),
                "duration_sec": float(duration_sec),
                "temperature": float(temperature),
                "n_samples_per_prompt": int(n_samples_per_prompt),
                "marl_algorithm": getattr(self.args, "marl_algorithm", None),
                "policy_sharing_mode": getattr(self.args, "policy_sharing_mode", None),
                "q_critic_status": _to_jsonable(getattr(self, "_last_q_critic_status", None)),
                "metrics": _to_jsonable(dict(logs or {})),
            }
        )

        try:
            _append_jsonl(str(path), payload, mode="append", dedupe_key=("eval_metrics", int(global_step)))
            # Ray teardown / eval_only often terminates workers without running atexit.
            # eval metrics is usually a single line and can stay in buffer forever.
            # Force flush so downstream scripts can see *.metrics.jsonl immediately.
            flush_jsonl_buffers(str(path))
        except Exception as e:
            logger.warning(f"Failed to append eval metrics JSONL to {path}: {e}")

    # -------------------------------------------------------------------------
    # WandB / Tensorboard
    # -------------------------------------------------------------------------

    def _init_wandb(self):
        self._wandb = None
        self._tensorboard = None
        self.generated_samples_table = None

        enable, mode, key = _normalize_use_wandb_arg(getattr(self.strategy.args, "use_wandb", None))
        if enable:
            import wandb

            self._wandb = wandb

            def _wandb_has_key() -> bool:
                try:
                    return bool(getattr(getattr(wandb, "api", None), "api_key", None))
                except Exception:
                    return False

            if not _wandb_has_key():
                try:
                    if mode == "key" and key:
                        wandb.login(key=key)
                    else:
                        wandb.login()
                except Exception as e:
                    if mode == "auto":
                        if _is_rank0(self.strategy):
                            logger.warning(f"[wandb] auto login failed; disable wandb. err={e}")
                        self._wandb = None
                    else:
                        raise

            if self._wandb is not None and (not _wandb_has_key()) and mode != "auto":
                raise RuntimeError("--use_wandb requires wandb login/WANDB_API_KEY (or pass a key string).")

            if self._wandb is not None:
                wandb.init(
                    entity=self.strategy.args.wandb_org,
                    project=self.strategy.args.wandb_project,
                    group=self.strategy.args.wandb_group,
                    name=self.strategy.args.wandb_run_name,
                    tags=build_wandb_tags(self.strategy.args),
                    config=build_wandb_config(self.strategy.args),
                    reinit=True,
                )

                try:
                    wandb.config.update(
                        {
                            "run_id": getattr(self.strategy.args, "run_id", None),
                            "run_dir": getattr(self.strategy.args, "run_dir", None),
                            "eval_only": bool(getattr(self.strategy.args, "eval_only", False)),
                            "eval_global_step": getattr(self.strategy.args, "eval_global_step", None),
                        },
                        allow_val_change=True,
                    )
                except Exception:
                    pass

                wandb.define_metric("train/global_step")
                wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
                wandb.define_metric("eval/global_step")
                wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

                self.generated_samples_table = wandb.Table(
                    columns=[
                        "global_step",
                        "question_id",
                        "k_id",
                        "role",
                        "marl_algorithm",
                        "c3_credit",
                        "c3_credit_scalar",
                        "c3_diag_json",
                        "reward_source",
                        "reward",
                        "text",
                        "env_info_json",
                    ]
                )

        if self.strategy.args.use_tensorboard and self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, self.strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def _wandb_log_generated_sample(self, global_step: int, gs: dict):
        if not isinstance(gs, dict):
            try:
                text, reward = gs
            except Exception:
                text, reward = "", 0.0
            gs = {"text": text, "reward": reward}

        if self.generated_samples_table is None:
            return

        row = [global_step if c == "global_step" else gs.get(c, None) for c in self.generated_samples_table.columns]
        for col, cast, default in [
            ("question_id", int, -1),
            ("k_id", int, -1),
            ("reward", float, 0.0),
            ("c3_credit_scalar", float, 0.0),
        ]:
            try:
                idx = self.generated_samples_table.columns.index(col)
                row[idx] = cast(row[idx]) if row[idx] is not None else default
            except Exception:
                pass

        self.generated_samples_table.add_data(*row)
        self._wandb.log({"train/generated_samples": self.generated_samples_table})

    def _format_tensorboard_generated_sample(self, gs) -> str:
        if isinstance(gs, dict):
            return (
                f"QID: {gs.get('question_id',-1)}  KID: {gs.get('k_id',-1)}\n"
                f"Role: {gs.get('role','')}\n"
                f"MARL: {gs.get('marl_algorithm','')}\n"
                f"C3_Credit: {gs.get('c3_credit','')}  C3_Scalar: {gs.get('c3_credit_scalar',None)}\n"
                f"RewardSource: {gs.get('reward_source','')}\n"
                f"Reward: {float(gs.get('reward',0.0)):.4f}\n\n"
                f"Sample:\n{gs.get('text','')}\n\n"
                f"EnvInfoJSON:\n{gs.get('env_info_json','')}\n\n"
                f"C3_DiagJSON:\n{gs.get('c3_diag_json','')}"
            )
        try:
            text, reward = gs
        except Exception:
            text, reward = "", 0.0
        return f"Sample:\n{text}\n\nReward: {float(reward):.4f}"

    # -------------------------------------------------------------------------
    # Checkpoint management
    # -------------------------------------------------------------------------

    def _materialize_ckpt_client_states(self, client_states: Optional[dict]) -> dict:
        """Materialize lightweight ckpt client states (dataloader state only at ckpt time)."""
        # Hard kill-switch: disabling means resume isn't bit-exact.
        save_dl_state = str(os.environ.get("OPENRLHF_SAVE_DATALOADER_STATE", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
        }

        if client_states is None:
            cs = {}
        elif isinstance(client_states, dict):
            cs = dict(client_states)
        else:
            cs = dict(client_states or {})

        if not save_dl_state:
            cs["data_loader_state_dict"] = None
            return cs

        if cs.get("data_loader_state_dict", None) is None:
            dl = getattr(self, "prompts_dataloader", None)
            if dl is not None and hasattr(dl, "state_dict"):
                try:
                    cs["data_loader_state_dict"] = dl.state_dict()
                except Exception:
                    cs["data_loader_state_dict"] = None
        return cs

    def _save_checkpoints(self, tag: str, client_states: Optional[dict] = None) -> None:
        # IMPORTANT: do not call dataloader.state_dict() per step; do it only here.
        cs = self._materialize_ckpt_client_states(client_states)

        refs = []
        if getattr(self, "actor_model_groups", None) is not None:
            for rn in self.role_names:
                refs.extend(
                    self.actor_model_groups[rn].async_run_method(
                        method_name="save_checkpoint", tag=tag, client_states=cs
                    )
                )
        else:
            refs.extend(self.actor_model_group.async_run_method(method_name="save_checkpoint", tag=tag, client_states=cs))

        if getattr(self, "critic_model_group", None) is not None:
            refs.extend(self.critic_model_group.async_run_method(method_name="save_checkpoint", tag=tag))
        if getattr(self, "q_critic_model_group", None) is not None:
            refs.extend(self.q_critic_model_group.async_run_method(method_name="save_checkpoint", tag=tag))

        ray.get(refs)

    def _load_checkpoint_states(self) -> dict:
        args = self.args
        checkpoint_states = {"global_step": 0, "episode": 0, "data_loader_state_dict": {}}

        if getattr(self, "actor_model_groups", None) is not None:
            if not args.load_checkpoint:
                return checkpoint_states

            present = []
            for rn in self.role_names:
                ckpt_path = os.path.join(args.ckpt_path, f"_actor_{rn}")
                if os.path.exists(ckpt_path):
                    present.append(rn)

            if present and len(present) != len(self.role_names):
                raise RuntimeError(
                    "per_role resume requires all roles checkpoints. "
                    f"found={present}, expected_all={self.role_names}. "
                    "Please ensure _actor_<role> exists for every role."
                )

            if len(present) == len(self.role_names):
                st = ray.get(
                    self.actor_model_groups[self.role_names[0]].async_run_method(method_name="get_checkpoint_states")
                )[0]
                if st:
                    checkpoint_states = st
            return checkpoint_states

        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            st = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[0]
            if st:
                checkpoint_states = st
            logger.info(f"checkpoint_states: {checkpoint_states}")
        return checkpoint_states

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict=None, client_states=None, last_step=0):
        """
        Save logs/eval/checkpoints.

        Returns updated last_step (last eval step).
        """
        logs_dict = dict(logs_dict or {})

        # Keep caller's dict identity (resume bookkeeping).
        if client_states is None:
            client_states = {}
        elif not isinstance(client_states, dict):
            client_states = dict(client_states or {})

        try:
            _last_eval_step = int(last_step or 0)
        except Exception:
            _last_eval_step = 0

        # logs
        if global_step % args.logging_steps == 0:
            if self._wandb is not None:
                if "generated_samples" in logs_dict:
                    self._wandb_log_generated_sample(global_step, logs_dict.pop("generated_samples"))
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

            elif self._tensorboard is not None:
                for k, v in logs_dict.items():
                    if k == "generated_samples":
                        self._tensorboard.add_text(
                            "train/generated_samples", self._format_tensorboard_generated_sample(v), global_step
                        )
                    else:
                        self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # Auto-enable save_on_eval for progress-based eval schedules.
        try:
            progress_ratio = float(getattr(args, "eval_every_ratio", 0.0) or 0.0)
            progress_percent = float(getattr(args, "eval_every_percent", 0.0) or 0.0)
            progress_eval_enabled = (progress_ratio > 0.0) or (progress_percent > 0.0)
        except Exception:
            progress_eval_enabled = False

        if progress_eval_enabled and not bool(getattr(args, "save_on_eval", False)):
            if not bool(getattr(self, "_auto_save_on_eval_enabled", False)):
                for tgt in (args, getattr(self.strategy, "args", None)):
                    if tgt is None:
                        continue
                    try:
                        tgt.save_on_eval = True
                    except Exception:
                        pass
                self._auto_save_on_eval_enabled = True
                logger.info("[INFO] auto-enabled save_on_eval because eval_every_percent/ratio is set.")

        did_eval = False

        # eval
        try:
            eval_steps = getattr(args, "eval_steps", None)
            eval_offset = int(getattr(args, "eval_steps_offset", 0) or 0)
        except Exception:
            eval_steps = None
            eval_offset = 0
        if eval_offset < 0:
            eval_offset = 0

        def _is_finite_int_step(x) -> bool:
            try:
                if x is None:
                    return False
                if isinstance(x, float):
                    if x != x or x == float("inf"):
                        return False
                return int(x) > 0
            except Exception:
                return False

        if _is_finite_int_step(eval_steps) and self.eval_dataloader and len(self.eval_dataloader) > 0:
            es = int(eval_steps)
            if int(global_step) >= int(eval_offset) and ((int(global_step) - int(eval_offset)) % es == 0):
                try:
                    _cur = int(global_step or 0)
                except Exception:
                    _cur = 0

                if _cur != int(_last_eval_step):
                    _ = self.evaluate(
                        self.eval_dataloader,
                        global_step,
                        args.eval_temperature,
                        args.eval_n_samples_per_prompt,
                    )
                    did_eval = True
                    _last_eval_step = int(_cur)
                    try:
                        client_states["last_eval_step"] = int(_last_eval_step)
                    except Exception:
                        pass

        # checkpoint after eval
        if did_eval and bool(getattr(args, "save_on_eval", False)):
            self._barrier_pending_q_critic(where="before_save_on_eval")
            self._save_checkpoints(tag=f"eval_step{global_step}", client_states=client_states)

        # checkpoints
        if global_step % args.save_steps == 0:
            self._barrier_pending_q_critic(where="before_save_steps")
            self._save_checkpoints(tag=f"global_step{global_step}", client_states=client_states)

        return int(_last_eval_step)

    # -------------------------------------------------------------------------
    # Eval JSONL dump
    # -------------------------------------------------------------------------

    @classmethod
    def _summarize_experience(cls, exp) -> dict:
        out = {"info": {}}
        info = getattr(exp, "info", None)
        if isinstance(info, dict):
            out["info"] = _to_jsonable(info)
        for name in ("reward", "rewards", "returns", "advantages", "values"):
            if hasattr(exp, name):
                out[name] = _to_jsonable(getattr(exp, name))
        for name in ("action_mask",):
            if hasattr(exp, name):
                out[name] = _to_jsonable(getattr(exp, name))
        return out

    def _dump_eval_jsonl(
        self,
        samples_list_full,
        base_prompts,
        base_labels,
        base_datasources,
        base_meta_jsons,
        global_step: int,
        n_samples_per_prompt: int,
        dump_path: str,
    ) -> int:
        if not dump_path:
            return 0

        # detect MARL batches
        marl_enabled = False
        try:
            if samples_list_full and isinstance(getattr(samples_list_full[0], "info", None), dict):
                info0 = samples_list_full[0].info
                if "marl_enabled" in info0:
                    marl_enabled = bool(int(_tensor_first_scalar(info0.get("marl_enabled"), 0)))
        except Exception:
            marl_enabled = False

        if marl_enabled:
            grouped = {}
            for exp in samples_list_full:
                info = getattr(exp, "info", None) or {}
                qid = _tensor1_int(info.get("question_id", -1), default=-1)
                kid = _tensor1_int(info.get("k_id", 0), default=0)
                key = (int(qid), int(kid))

                if key not in grouped:
                    ans_role = _first(info.get("answer_role", None), None)
                    grouped[key] = {
                        "ts": float(time.time()),
                        "global_step": int(global_step),
                        "question_id": int(qid),
                        "k_id": int(kid),
                        "datasource": base_datasources[qid] if 0 <= qid < len(base_datasources) else None,
                        "prompt": base_prompts[qid] if 0 <= qid < len(base_prompts) else None,
                        "label": base_labels[qid] if 0 <= qid < len(base_labels) else None,
                        "dataset_meta_json": base_meta_jsons[qid]
                        if base_meta_jsons is not None and 0 <= qid < len(base_meta_jsons)
                        else None,
                        "task_name": _first(info.get("task_name", None), None),
                        "env_name": _first(info.get("env_name", None), None),
                        "roles_topo": _to_jsonable(info.get("roles_topo", None)),
                        "answer_role": ans_role,
                        "traj_role_outputs": _to_jsonable(info.get("traj_role_outputs", {})),
                        "answer_text": None,
                        "answer_reward": None,
                        "reward_source": None,
                        "reward_info_json": None,
                    }

                role = _sample_role_name(exp)
                ans = grouped[key].get("answer_role")
                if ans is None:
                    grouped[key]["answer_role"] = _first(info.get("answer_role", None), None)
                    ans = grouped[key]["answer_role"]

                if ans is not None and role == ans:
                    grouped[key]["answer_reward"] = _tensor1_float(getattr(exp, "rewards", None), default=None)
                    tro = info.get("traj_role_outputs", None)
                    if isinstance(tro, dict):
                        grouped[key]["traj_role_outputs"] = _to_jsonable(tro)
                        if ans in tro:
                            grouped[key]["answer_text"] = tro.get(ans)

                    grouped[key]["reward_source"] = _first(info.get("reward_source", None), None)
                    grouped[key]["reward_info_json"] = _first(info.get("reward_info_json", None), None)

            n = 0
            for key in sorted(grouped.keys()):
                _append_jsonl(str(dump_path), self._with_run_context(grouped[key]))
                n += 1
            return int(n)

        # non-MARL
        n = 0
        k = max(int(n_samples_per_prompt), 1)
        for idx, exp in enumerate(samples_list_full or []):
            qid = idx // k
            kid = idx % k
            info = getattr(exp, "info", None) or {}

            text = None
            try:
                seq = exp.sequences[0]
                text = self.tokenizer.decode(seq, skip_special_tokens=False)
            except Exception:
                text = None

            row = {
                "ts": float(time.time()),
                "global_step": int(global_step),
                "question_id": int(qid),
                "k_id": int(kid),
                "datasource": base_datasources[qid] if 0 <= qid < len(base_datasources) else None,
                "prompt": base_prompts[qid] if 0 <= qid < len(base_prompts) else None,
                "label": base_labels[qid] if 0 <= qid < len(base_labels) else None,
                "dataset_meta_json": base_meta_jsons[qid]
                if base_meta_jsons is not None and 0 <= qid < len(base_meta_jsons)
                else None,
                "text": text,
                "reward": _tensor1_float(getattr(exp, "rewards", None), default=None),
                "info": _to_jsonable(info),
            }
            _append_jsonl(str(dump_path), self._with_run_context(row))
            n += 1
        return int(n)

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def run_eval_only(self, global_step: int = 0):
        if self.eval_dataloader is None:
            raise RuntimeError("eval_only requested but eval_dataloader is None. Provide --eval_dataset or --c3_task.")
        self._sync_actor_weights_to_vllm()
        return self.evaluate(
            self.eval_dataloader,
            global_step=int(global_step),
            temperature=float(
                0.7 if getattr(self.args, "eval_temperature", None) is None
                else getattr(self.args, "eval_temperature")
            ),
            n_samples_per_prompt=int(getattr(self.args, "eval_n_samples_per_prompt", 1) or 1),
        )

    def evaluate(self, eval_dataloader, global_step, temperature=0.6, n_samples_per_prompt=1):
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        args = self.args
        self._maybe_vllm_wake(self.vllm_engines)

        dump_path = getattr(args, "eval_dump_path", None)
        dump_mode = str(getattr(args, "eval_dump_mode", "append") or "append").lower().strip()
        do_dump = bool(dump_path) and _is_rank0(self.strategy)

        if do_dump and dump_mode == "overwrite":
            try:
                _maybe_reset_file(str(dump_path))
            except Exception as e:
                _safe_print(self.strategy, f"[WARN] eval_dump overwrite reset failed: {type(e).__name__}: {e}")

        logs = {}
        with torch.no_grad():
            base_prompts, base_labels, base_datasources, base_meta_jsons = [], [], [], []
            for batch in eval_dataloader:
                datasources, prompts, labels, meta_jsons = _unpack_prompt_batch(batch, where="evaluate()")
                base_prompts.extend(list(prompts))
                base_labels.extend(list(labels))
                base_datasources.extend(list(datasources))
                if meta_jsons is not None:
                    base_meta_jsons.extend(list(meta_jsons))

            gen_kwargs = dict(self.generate_kwargs)
            gen_kwargs.update({"temperature": temperature, "n_samples_per_prompt": n_samples_per_prompt})
            gen_kwargs["phase"] = "eval"
            if base_meta_jsons and len(base_meta_jsons) == len(base_prompts):
                gen_kwargs["all_metas"] = base_meta_jsons

            samples_list_full = self.samples_generator.generate_samples(
                base_prompts, base_labels, remote_reward_model=self.remote_reward_model, **gen_kwargs
            )

            if do_dump:
                try:
                    self._dump_eval_jsonl(
                        samples_list_full,
                        base_prompts=base_prompts,
                        base_labels=base_labels,
                        base_datasources=base_datasources,
                        base_meta_jsons=(
                            base_meta_jsons if base_meta_jsons and len(base_meta_jsons) == len(base_prompts) else None
                        ),
                        global_step=int(global_step),
                        n_samples_per_prompt=int(n_samples_per_prompt),
                        dump_path=str(dump_path),
                    )
                except Exception as e:
                    _safe_print(self.strategy, f"[WARN] eval_dump_jsonl failed: {type(e).__name__}: {e}")

            # MAS/MARL: keep answer_role only for metrics
            samples_list = samples_list_full
            if samples_list and isinstance(getattr(samples_list[0], "info", None), dict):
                info0 = samples_list[0].info
                marl_enabled = False
                try:
                    if "marl_enabled" in info0:
                        marl_enabled = bool(int(_tensor_first_scalar(info0.get("marl_enabled", torch.tensor([0])), 0)))
                except Exception:
                    marl_enabled = False

                if marl_enabled and "answer_role" in info0 and "role" in info0:
                    ans_role = _first(info0.get("answer_role"), None)
                    samples_list = [s for s in samples_list if _sample_role_name(s) == ans_role]

            # rewards -> [num_prompts, K]
            K = int(n_samples_per_prompt)
            rewards_flat: List[float] = []
            for s in samples_list:
                rewards_flat.append(_tensor1_float(getattr(s, "rewards", None), default=0.0) or 0.0)
            rewards = torch.tensor(rewards_flat, dtype=torch.float32).reshape(-1, K)

            global_metrics = {}
            num_prompts = int(rewards.shape[0])
            for i in range(num_prompts):
                datasource = base_datasources[i] if i < len(base_datasources) else "unknown"
                global_metrics.setdefault(datasource, {f"pass{K}": 0.0, "pass1": 0.0, "count": 0})

                chunk_rewards = rewards[i]
                if K > 1:
                    global_metrics[datasource][f"pass{K}"] += float(chunk_rewards.max().item())
                global_metrics[datasource]["pass1"] += float(chunk_rewards.mean().item())
                global_metrics[datasource]["count"] += 1

            logs = {}
            for datasource, metrics in global_metrics.items():
                cnt = max(int(metrics["count"]), 1)
                logs[f"eval_{datasource}_pass{K}"] = float(metrics[f"pass{K}"]) / cnt
                logs[f"eval_{datasource}_pass1"] = float(metrics["pass1"]) / cnt

            if self._wandb is not None:
                self._wandb.log({"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()})
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        self._maybe_vllm_sleep(self.vllm_engines)

        duration = time.time() - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]

        try:
            self._maybe_dump_eval_metrics_jsonl(
                global_step=int(global_step),
                logs=dict(logs or {}),
                duration_sec=float(duration),
                temperature=float(temperature),
                n_samples_per_prompt=int(n_samples_per_prompt),
                eval_dump_path=(str(dump_path) if dump_path else None),
            )
        except Exception:
            pass

        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")
        return logs

    # -------------------------------------------------------------------------
    # Timing / async Q-critic
    # -------------------------------------------------------------------------

    def _q_critic_async_overlap_enabled(self) -> bool:
        if bool(getattr(self, "_in_critic_warmup_stage", False)):
            return False
        if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
            return False
        return bool(int(getattr(self.args, "q_critic_async_overlap", 0) or 0))

    def _timing_enabled(self) -> bool:
        return bool(int(getattr(self.args, "log_time_breakdown", 0) or 0))

    def _barrier_pending_q_critic(self, where: str = "") -> Optional[dict]:
        pending = getattr(self, "_pending_q_critic_ref", None)
        if pending is None:
            return None
        try:
            stats_any = ray.get(pending)
            stats_any = _first(stats_any)
            stats_any = _first(stats_any)
            self._last_q_critic_status = dict(stats_any) if isinstance(stats_any, Mapping) else None
            return self._last_q_critic_status
        finally:
            self._pending_q_critic_ref = None

    # -------------------------------------------------------------------------
    # Rollout-derived scalar stats
    # -------------------------------------------------------------------------

    def _rewardprovider_rollout_metrics(self, experiences_all):
        """Scalar metrics from Experience.info (reward_source/env_info_json/reward)."""
        from collections import Counter

        rows = []
        for exp in experiences_all or []:
            info = getattr(exp, "info", None) or {}
            B = int(exp.sequences.shape[0]) if getattr(exp, "sequences", None) is not None else 1

            roles = _as_list_str(info.get("role", None), B, fill="")
            sources = _as_list_str(info.get("reward_source", None), B, fill="unknown")
            env_infos = _as_list_str(info.get("env_info_json", None), B, fill="")

            r = info.get("reward", None)
            if r is None:
                r = getattr(exp, "rewards", None)

            r_list = []
            try:
                if isinstance(r, torch.Tensor):
                    r_list = [float(x) for x in r.detach().cpu().flatten().tolist()]
            except Exception:
                r_list = []

            if not r_list:
                if isinstance(r, list):
                    r_list = [float(x) for x in r]
                elif r is not None:
                    try:
                        r_list = [float(r)]
                    except Exception:
                        r_list = []
                else:
                    r_list = [0.0] * B

            if len(r_list) == 1 and B > 1:
                r_list = r_list * B
            if len(r_list) != B:
                r_list = (r_list + [r_list[-1]] * B)[:B]

            for i in range(B):
                rows.append(
                    {
                        "role": str(roles[i] or ""),
                        "source": str(sources[i] or "unknown"),
                        "reward": float(r_list[i]),
                        "env_info_json": str(env_infos[i] or ""),
                    }
                )

        if not rows:
            return {}

        rewards = [x["reward"] for x in rows]
        srcs = [x["source"] for x in rows]
        roles = [x["role"] for x in rows]

        n = len(rewards)
        mean = sum(rewards) / max(n, 1)
        var = sum((x - mean) ** 2 for x in rewards) / max(n, 1)
        std = var**0.5

        metrics = {
            "reward/mean": mean,
            "reward/std": std,
            "reward/min": min(rewards),
            "reward/max": max(rewards),
            "reward/count": n,
        }

        c = Counter(srcs)
        for src, cnt in c.items():
            src_key = str(src).replace("/", "_")
            metrics[f"reward/source/{src_key}/count"] = int(cnt)
            metrics[f"reward/source/{src_key}/frac"] = float(cnt) / max(n, 1)

        by_role_sum: Dict[str, float] = {}
        by_role_cnt: Dict[str, int] = {}
        for rname, rv in zip(roles, rewards):
            key = (rname or "unknown").replace("/", "_")
            by_role_sum[key] = by_role_sum.get(key, 0.0) + float(rv)
            by_role_cnt[key] = by_role_cnt.get(key, 0) + 1
        for rname, s in by_role_sum.items():
            metrics[f"reward/by_role/{rname}/mean"] = s / max(by_role_cnt.get(rname, 1), 1)

        err = 0
        for x in rows:
            s = x["env_info_json"]
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "error" in obj:
                    err += 1
            except Exception:
                if "error" in s.lower():
                    err += 1
        metrics["env/error_frac"] = float(err) / max(n, 1)
        return metrics

    def _marl_response_clip_metrics(self, experiences_all) -> Dict[str, object]:
        by_role_sum: Dict[str, float] = {}
        by_role_cnt: Dict[str, int] = {}
        topo = list(getattr(self, "role_names", []) or [])

        for exp in experiences_all or []:
            info = getattr(exp, "info", None) or {}
            B = int(exp.sequences.shape[0]) if getattr(exp, "sequences", None) is not None else 1

            marl_enabled = _as_list_bool(info.get("marl_enabled", None), B)
            clips = _as_list_bool(info.get("response_clip_ratio", None), B)

            if "role" in info:
                roles = _as_list_str(info.get("role", None), B, fill="")
            else:
                rids = _as_list_int(info.get("role_id", None), B, default=-1)
                roles = []
                for rid in rids:
                    if topo and 0 <= int(rid) < len(topo):
                        roles.append(str(topo[int(rid)]))
                    else:
                        roles.append(f"id{int(rid)}")

            for rname, en, cl in zip(roles, marl_enabled, clips):
                if not en:
                    continue
                key = (rname or "unknown").replace("/", "_")
                by_role_sum[key] = by_role_sum.get(key, 0.0) + (1.0 if cl else 0.0)
                by_role_cnt[key] = by_role_cnt.get(key, 0) + 1

        metrics: Dict[str, object] = {}
        for rname, s in by_role_sum.items():
            cnt = int(by_role_cnt.get(rname, 0) or 0)
            if cnt <= 0:
                continue
            metrics[f"response_clip_ratio/by_role/{rname}/mean"] = float(s) / float(cnt)
            metrics[f"response_clip_ratio/by_role/{rname}/count"] = int(cnt)
        return metrics

    def _c3_credit_rollout_metrics(self, experiences_all):
        from collections import Counter, defaultdict

        scalars: List[float] = []
        tags: List[str] = []
        qids: List[int] = []
        kids: List[int] = []
        K_expected = int(getattr(self.args, "n_samples_per_prompt", 1) or 1)

        def _as_int_list(v, B: int, default: int = -1) -> List[int]:
            arr = []
            if isinstance(v, torch.Tensor):
                arr = [int(x) for x in v.detach().cpu().view(-1).tolist()]
            elif isinstance(v, list):
                for x in v:
                    try:
                        arr.append(int(x))
                    except Exception:
                        arr.append(default)
            elif v is None:
                arr = []
            else:
                try:
                    arr = [int(v)]
                except Exception:
                    arr = [default]

            if len(arr) == B:
                return arr
            if len(arr) == 1 and B > 1:
                return arr * B
            if 1 < len(arr) < B:
                return arr + [arr[0]] * (B - len(arr))
            if len(arr) > B:
                return arr[:B]
            return [default] * B

        for exp in experiences_all or []:
            info = getattr(exp, "info", None) or {}
            if not isinstance(info, dict):
                continue
            cs = info.get("c3_credit_scalar", None)
            if cs is None:
                continue

            if isinstance(cs, torch.Tensor):
                cs_list = [float(x) for x in cs.detach().cpu().view(-1).tolist()]
            elif isinstance(cs, list):
                try:
                    cs_list = [float(x) for x in cs]
                except Exception:
                    cs_list = []
            else:
                try:
                    cs_list = [float(cs)]
                except Exception:
                    cs_list = []

            if not cs_list:
                continue
            B = len(cs_list)

            tg = info.get("c3_credit", None)
            if isinstance(tg, list):
                tg_list = [str(tg[0])] * B if tg else [""] * B
                if len(tg) == B:
                    tg_list = [str(x) for x in tg]
            elif isinstance(tg, str):
                tg_list = [tg] * B
            else:
                tg_list = [""] * B

            q_list = _as_int_list(info.get("question_id", None), B, default=-1)
            k_list = _as_int_list(info.get("k_id", None), B, default=-1)

            scalars.extend(cs_list)
            tags.extend(tg_list)
            qids.extend(q_list)
            kids.extend(k_list)

        if not scalars:
            return {}

        s = torch.tensor(scalars, dtype=torch.float32)
        n = int(s.numel())
        mean = float(s.mean().item())
        std = float(s.std(unbiased=False).item()) if n > 1 else 0.0
        abs_p90 = float(torch.quantile(s.abs(), 0.9).item()) if n > 1 else float(s.abs().view(-1)[0].item())

        c = Counter(tags)
        out = {
            "c3/credit_scalar/mean": mean,
            "c3/credit_scalar/std": std,
            "c3/credit_scalar/abs_p90": abs_p90,
            "c3/credit_tag/q_critic_frac": float(c.get("q_critic", 0)) / max(n, 1),
            "c3/credit_tag/fallback_frac": float(c.get("fallback", 0)) / max(n, 1),
            "c3/credit_scalar/count": n,
        }

        if K_expected > 1:
            qid_to_kset = defaultdict(set)
            for q, k in zip(qids, kids):
                if q is None or k is None:
                    continue
                if int(q) < 0 or int(k) < 0:
                    continue
                qid_to_kset[int(q)].add(int(k))

            if qid_to_kset:
                unique_k_counts = [len(v) for v in qid_to_kset.values()]
                mean_unique_k = float(sum(unique_k_counts) / max(len(unique_k_counts), 1))
                incomplete_qids = {q for q, ks in qid_to_kset.items() if len(ks) < K_expected}
                incomplete_qid_frac = float(len(incomplete_qids)) / max(len(qid_to_kset), 1)

                incomplete_row = 0
                fallback_incomplete_row = 0
                fallback_total = 0
                for q, t in zip(qids, tags):
                    if int(q) in incomplete_qids:
                        incomplete_row += 1
                        if str(t) == "fallback":
                            fallback_incomplete_row += 1
                    if str(t) == "fallback":
                        fallback_total += 1

                out.update(
                    {
                        "c3/group/unique_k_count/mean": mean_unique_k,
                        "c3/group/incomplete_qid_frac": incomplete_qid_frac,
                        "c3/group/incomplete_row_frac": float(incomplete_row) / max(n, 1),
                        "c3/credit_tag/fallback_incomplete_k_row_frac": float(fallback_incomplete_row) / max(n, 1),
                        "c3/credit_tag/fallback_incomplete_k_of_fallback_frac": float(fallback_incomplete_row)
                        / max(fallback_total, 1),
                    }
                )
        return out

    def _make_generated_sample_meta(self, exp, decoded_text: str):
        info = getattr(exp, "info", None) or {}

        role = _first(info.get("role", None), "")
        role = role if isinstance(role, str) else ""

        rs = _first(info.get("reward_source", None), "")
        rs = rs if isinstance(rs, str) else ""

        envj = _first(info.get("env_info_json", None), "")
        envj = envj if isinstance(envj, str) else ""

        qid = _tensor1_int(info.get("question_id", None), default=-1)
        kid = _tensor1_int(info.get("k_id", None), default=-1)

        c3_credit = _first(info.get("c3_credit", None), "")
        c3_credit = c3_credit if isinstance(c3_credit, str) else ""

        ss = info.get("c3_credit_scalar", None)
        c3_credit_scalar = None
        try:
            if isinstance(ss, torch.Tensor) and ss.numel() >= 1:
                c3_credit_scalar = float(ss.detach().cpu().view(-1)[0].item())
            elif isinstance(ss, list) and ss:
                c3_credit_scalar = float(ss[0])
            elif ss is not None:
                c3_credit_scalar = float(ss)
        except Exception:
            c3_credit_scalar = None

        c3_diag = {}
        try:
            for k, v in info.items():
                if str(k).startswith("c3/"):
                    vv = _tensor1_float(v, default=None)
                    if vv is not None:
                        c3_diag[str(k)] = vv
        except Exception:
            c3_diag = {}

        c3_diag_json = ""
        if c3_diag:
            try:
                c3_diag_json = json.dumps(c3_diag, ensure_ascii=False)
            except Exception:
                c3_diag_json = ""

        reward = 0.0
        try:
            rr = info.get("reward", None) if isinstance(info, dict) else None
            if rr is None:
                rr = getattr(exp, "rewards", None)
            reward = float(_tensor_first_scalar(rr, 0.0) or 0.0)
        except Exception:
            reward = 0.0

        marl_alg = str(getattr(self.args, "marl_algorithm", "") or "")
        return {
            "text": decoded_text,
            "reward": float(reward),
            "question_id": int(qid),
            "k_id": int(kid),
            "role": role,
            "marl_algorithm": marl_alg,
            "c3_credit": c3_credit or "",
            "c3_credit_scalar": c3_credit_scalar,
            "c3_diag_json": c3_diag_json,
            "reward_source": rs or "unknown",
            "env_info_json": envj,
        }

    # -------------------------------------------------------------------------
    # Dataloader epoch helper
    # -------------------------------------------------------------------------

    def _maybe_set_dataloader_epoch(self, dl, epoch: int, resume_has_state: bool, start_epoch: int):
        if not bool(getattr(self, "_c3_reshuffle_each_epoch", False)) or dl is None:
            return
        if (epoch == start_epoch) and resume_has_state:
            if _is_rank0(self.strategy):
                logger.info(
                    "[FS-114] Resume detected (data_loader_state_dict!=empty). "
                    f"Skip set_epoch for epoch={epoch} to preserve resume position."
                )
            return

        try:
            if hasattr(dl, "set_epoch"):
                dl.set_epoch(int(epoch))
                return
        except Exception:
            pass

        sampler = getattr(dl, "sampler", None)
        if sampler is None:
            bs = getattr(dl, "batch_sampler", None)
            sampler = getattr(bs, "sampler", None) if bs is not None else None
        if sampler is not None and hasattr(sampler, "set_epoch"):
            try:
                sampler.set_epoch(int(epoch))
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Critic target resolution / eval schedule setters
    # -------------------------------------------------------------------------

    def set_eval_steps(self, eval_steps: int):
        try:
            v = int(eval_steps)
        except Exception:
            v = eval_steps
        for tgt in (self.args, getattr(self.strategy, "args", None)):
            if tgt is None:
                continue
            try:
                tgt.eval_steps = v
            except Exception:
                pass
        return v

    def set_eval_steps_offset(self, eval_steps_offset: int):
        try:
            v = int(eval_steps_offset)
        except Exception:
            v = eval_steps_offset
        try:
            if isinstance(v, int) and v < 0:
                v = 0
        except Exception:
            pass
        for tgt in (self.args, getattr(self.strategy, "args", None)):
            if tgt is None:
                continue
            try:
                tgt.eval_steps_offset = v
            except Exception:
                pass
        return v

    def _resolve_critic_target_eff(self, marl_alg: str) -> str:
        raw = str(getattr(self, "critic_target_raw", "auto") or "auto").lower().strip()
        marl_alg = str(marl_alg or "auto").lower().strip()
        if raw != "auto":
            if marl_alg == "c3":
                cv = str(getattr(self.args, "c3_credit_variant", "value_assisted") or "value_assisted").lower().strip()
                if cv == "reward_only" and raw in {"q", "all"}:
                    raise RuntimeError(
                        "[C3][FAIL-FAST] critic_target is explicitly set to 'q/all' but c3_credit_variant=reward_only. "
                        "Reward-only C3 prunes Q-critic; please set --critic_target auto/none or use value_assisted/value_only."
                    )
            return raw

        if marl_alg == "c3":
            cv = str(getattr(self.args, "c3_credit_variant", "value_assisted") or "value_assisted").lower().strip()
            return "none" if cv == "reward_only" else "q"
        if marl_alg == "mappo":
            return "v"
        if marl_alg == "magrpo":
            return "none"
        return "v"


__all__ = [
    "PPOTrainerPluginsMixin",
    "_is_rank0",
    "_safe_print",
    "_first",
    "_sample_role_name",
    "_tensor1_int",
    "_tensor1_float",
    "_to_jsonable",
    "_append_jsonl",
    "_maybe_reset_file",
    "_with_suffix",
    "_normalize_use_wandb_arg",
    "_unpack_prompt_batch",
]
