"""
C3/C3 paper analysis CLI (thin orchestrator).

Subcommands:
  - build-buckets : replay + bucket construction
  - credit        : compute fidelity/variance from buckets
  - influence     : compute conditional MI influence from buckets
  - latex         : format JSON outputs into a LaTeX row

Stability contract:
  This CLI is treated as a paper-facing interface. To avoid brittle bash scripts,
  we intentionally support common argument aliases across refactors, e.g.:
    build-buckets:
      --task / --task_yaml
      --split / --suite
      --policy_ckpt / --policy
      --target_role / --target
      --next_role / --next
      --analysis_yaml / --analysis-yaml
      --engine / --inference_engine
      --out / --out_jsonl
    credit/influence:
      --bucket / --buckets_jsonl
      --out / --out_json
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, NoReturn, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# IO / small utils (no external deps)
# -----------------------------------------------------------------------------

def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _die(msg: str, *, code: int = 2) -> NoReturn:
    _eprint(f"[c3_analysis] ERROR: {msg}")
    raise SystemExit(code)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, s: str) -> None:
    _ensure_parent_dir(path)
    path.write_text(s, encoding="utf-8")


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize objects before json dumping:
      * float NaN/Inf -> 0.0
      * torch/numpy scalar -> python scalar (best-effort)
    This prevents downstream validators from treating artifacts as invalid.
    """
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None  # type: ignore

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else 0.0
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj

    if _np is not None and isinstance(obj, (_np.floating, _np.integer)):
        v = obj.item()
        if isinstance(v, float) and not math.isfinite(v):
            return 0.0
        return v

    # torch scalar (best effort)
    try:
        import torch as _torch  # type: ignore
        if isinstance(obj, _torch.Tensor) and obj.ndim == 0:
            v = obj.item()
            if isinstance(v, float) and not math.isfinite(v):
                return 0.0
            return v
    except Exception:
        pass

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]

    return obj


def _write_json(path: Path, obj: Any) -> None:
    _ensure_parent_dir(path)
    obj = _sanitize_for_json(obj)
    # allow_nan=False ensures we never write bare NaN/Infinity into JSON.
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _import_symbol(dotted: str) -> Any:
    """Import `pkg.mod:attr` or `pkg.mod.attr`."""
    if ":" in dotted:
        mod, sym = dotted.split(":", 1)
    else:
        parts = dotted.split(".")
        if len(parts) < 2:
            _die(f"Invalid symbol path: {dotted!r} (expected 'pkg.mod:attr' or 'pkg.mod.attr').")
        mod, sym = ".".join(parts[:-1]), parts[-1]

    try:
        m = importlib.import_module(mod)
    except Exception as e:
        _die(f"Failed to import module {mod!r}: {e}")
    try:
        return getattr(m, sym)
    except Exception as e:
        _die(f"Module {mod!r} has no attribute {sym!r}: {e}")


def _call_by_signature(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call `fn` with only the kwargs it accepts."""
    sig = inspect.signature(fn)
    accepted: Dict[str, Any] = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**accepted)


def _try_load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML to dict (requires PyYAML)."""
    try:
        import yaml  # type: ignore
    except Exception:
        _die("PyYAML is required for --analysis_yaml. Install pyyaml or omit --analysis_yaml.")
    try:
        data = yaml.safe_load(_read_text(path))
    except Exception as e:
        _die(f"Failed to parse YAML {str(path)!r}: {e}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        _die(f"YAML root must be a mapping/dict. Got: {type(data).__name__}")
    return data


def _cfg_get(d: Mapping[str, Any], path: Sequence[str], default: Any) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_bool_tri(v: str) -> Optional[bool]:
    """Parse tri-state bool: auto -> None; true/false -> bool."""
    s = v.strip().lower()
    if s == "auto":
        return None
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    _die(f"Invalid tri-state boolean: {v!r} (expected auto,true,false).")
    return None  # unreachable


def _as_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if dataclasses.is_dataclass(x):
        return dataclasses.asdict(x)
    if hasattr(x, "dict") and callable(getattr(x, "dict")):
        return x.dict()  # type: ignore[no-any-return]
    if hasattr(x, "to_dict") and callable(getattr(x, "to_dict")):
        return x.to_dict()  # type: ignore[no-any-return]
    _die(f"Unsupported bucket object type: {type(x).__name__} (expected dict/dataclass or .to_dict()).")


# -----------------------------------------------------------------------------
# Imports from analysis package (fail-fast diagnostics)
# -----------------------------------------------------------------------------

def _import_analysis_modules() -> Tuple[Any, Any, Any]:
    """Return (replay_mod, buckets_mod, metrics_mod)."""
    mods: Dict[str, Any] = {}
    for name in (
        "c3.analysis.replay",
        "c3.analysis.buckets",
        "c3.analysis.metrics",
    ):
        try:
            mods[name] = importlib.import_module(name)
        except Exception as e:
            _die(f"Cannot import {name}. Import error: {e}")
    return (
        mods["c3.analysis.replay"],
        mods["c3.analysis.buckets"],
        mods["c3.analysis.metrics"],
    )


def _make_runner(args: argparse.Namespace, replay_mod: Any, *, decode_defaults: Mapping[str, Any] | None = None) -> Any:
    """
    Runner construction protocol:
      - Optional: --runner_factory pkg.mod:make_runner(args)->ReplayRunner
      - Default : ReplayRunner.from_cli(...) (should ignore unknown kwargs)
    """
    if getattr(args, "runner_factory", None):
        factory = _import_symbol(args.runner_factory)
        if not callable(factory):
            _die(f"--runner_factory must be callable. Got: {factory!r}")
        return factory(args)

    ReplayRunner = getattr(replay_mod, "ReplayRunner", None)
    if ReplayRunner is None:
        _die("replay.py must define ReplayRunner.")
    if not (hasattr(ReplayRunner, "from_cli") and callable(getattr(ReplayRunner, "from_cli"))):
        _die("ReplayRunner.from_cli not found. Either finish analysis replay implementation or use --runner_factory.")

    return ReplayRunner.from_cli(
        task=args.task,
        split=args.split,
        policy_ckpt=args.policy_ckpt,
        method=getattr(args, "method", None),
        seed=args.seed,
        device=getattr(args, "device", None),
        engine=getattr(args, "engine", "auto"),
        cache_dir=getattr(args, "cache_dir", None),
        tensor_parallel_size=getattr(args, "tensor_parallel_size", None),
        decode_defaults=dict(decode_defaults or {}),
    )


def _iter_restart_states(
    runner: Any,
    *,
    num_instances: int,
    seed: int,
    split: str,
    task: str,
    target_role: str,
    prefix_decoding: Mapping[str, Any] | None,
) -> Iterable[Any]:
    """RestartState protocol compatibility layer."""
    if hasattr(runner, "iter_restart_states") and callable(getattr(runner, "iter_restart_states")):
        return runner.iter_restart_states(
            task=task,
            split=split,
            target_role=target_role,
            limit=num_instances,
            seed=seed,
            prefix_decoding=prefix_decoding,
        )

    if hasattr(runner, "load_restart_states") and callable(getattr(runner, "load_restart_states")):
        return runner.load_restart_states(
            task=task,
            split=split,
            target_role=target_role,
            limit=num_instances,
            seed=seed,
            prefix_decoding=prefix_decoding,
        )

    if hasattr(runner, "restart_states"):
        states = getattr(runner, "restart_states")
        return list(states)[:num_instances]

    _die(
        "ReplayRunner must provide restart states via iter_restart_states/load_restart_states/restart_states. "
        "If your project uses a different protocol, supply --runner_factory."
    )


# -----------------------------------------------------------------------------
# Role helpers
# -----------------------------------------------------------------------------

def _canonical_role(name: str, roles_topo: Sequence[str]) -> str:
    for r in roles_topo:
        if r.lower() == name.lower():
            return r
    _die(f"Unknown role={name!r}. Available roles: {list(roles_topo)!r}")
    return name  # unreachable


def _default_next_role(target_role: str, roles_topo: Sequence[str]) -> Optional[str]:
    t = _canonical_role(target_role, roles_topo)
    idx = {r: i for i, r in enumerate(roles_topo)}.get(t)
    if idx is None:
        return None
    j = idx + 1
    return roles_topo[j] if 0 <= j < len(roles_topo) else None


# -----------------------------------------------------------------------------
# MAPPO V-critic loader (callable)
# -----------------------------------------------------------------------------

_MAPPO_V_CRITIC_CACHE: Dict[str, Callable[[Mapping[str, Any]], float]] = {}
_MAPP0_V_CRITIC_CACHE = _MAPPO_V_CRITIC_CACHE  # backward-compat alias (internal)


def _load_mappo_v_critic_callable(
    ckpt_path: str,
    *,
    device: str | None,
    base_model_path: str | None,
    max_len: int = 2048,
) -> Callable[[Mapping[str, Any]], float]:
    """
    Load MAPPO V-critic for credit_mode=mappo_v.

    Supports BOTH:
      1) HuggingFace-style critic dir (has config.json)
      2) DeepSpeed/Megatron-style dir (has mp_rank_*_model_states.pt only)

    NOTE:
      When `ckpt_path` is a DeepSpeed folder (no config.json), we need a base model
      directory to provide tokenizer/config for building the critic skeleton.

      IMPORTANT:
        For many MAPPO runs the critic base model is NOT the policy
        (e.g. critic=mmBERT-base hidden=768, policy=Qwen3-4B hidden=2560).
        In that case you MUST pass --mappo_critic_base_ckpt pointing to the critic base
        HF dir, otherwise you will hit score.weight (and/or other) size mismatches.

    Returns:
      v_critic(bucket_dict)->float
    """
    ckpt_path = str(ckpt_path)
    base_model_path = str(base_model_path) if base_model_path else ""

    cache_key = f"{ckpt_path}||{base_model_path or '<auto>'}||{device or '<auto>'}||{int(max_len)}"
    if cache_key in _MAPPO_V_CRITIC_CACHE:
        return _MAPPO_V_CRITIC_CACHE[cache_key]

    # Import locally to keep CLI import surface small.
    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as e:
        _die(f"transformers/torch are required to load MAPPO critic: {e}")

    try:
        from c3.mas.rollout_generator import _compose_mappo_state_text  # type: ignore
    except Exception as e:
        _die(f"Cannot import _compose_mappo_state_text (needed for critic state_text): {e}")

    def _has_hf_config(p: str) -> bool:
        return os.path.isfile(os.path.join(p, "config.json"))

    def _auto_base_model_path(critic_dir: str) -> str:
        """
        Best-effort inference when caller didn't provide a base model path.
        Common layout:
          <run_root>/_critic/eval_stepXXX/...
          <run_root>/final_hf
        """
        # If caller provided a non-empty path, prefer it (even if it isn't HF; it will fail loudly later).
        if base_model_path:
            return base_model_path

        # If the critic dir itself is HF, using it as base is fine.
        if _has_hf_config(critic_dir):
            return critic_dir

        # Try: <run_root>/final_hf
        run_root = os.path.dirname(os.path.dirname(os.path.abspath(critic_dir)))
        cand = os.path.join(run_root, "final_hf")
        if os.path.isdir(cand):
            return cand

        # Try one level up (in case critic_dir has extra nesting)
        run_root2 = os.path.dirname(run_root)
        cand2 = os.path.join(run_root2, "final_hf")
        if os.path.isdir(cand2):
            return cand2

        _die(
            "mappo_v critic checkpoint looks like a DeepSpeed/Megatron directory (no config.json), "
            "but no base model path was provided and auto-detect failed. "
            "Please pass --mappo_critic_base_ckpt (preferred) or --policy_ckpt pointing to a HF dir "
            "that contains config.json + tokenizer files."
        )
        return critic_dir  # unreachable

    def _extract_state_dict(blob: object) -> dict:
        """
        Extract a plausible state_dict from common DS/Megatron checkpoint containers, and
        apply light prefix-stripping to improve load compatibility.

        We keep this conservative and then optionally try a prefix-remap to match the
        CriticModel wrapper layout.
        """
        if isinstance(blob, dict):
            # common containers
            for k in ("module", "model", "state_dict", "model_state_dict"):
                v = blob.get(k)
                if isinstance(v, dict) and v:
                    blob = v
                    break
        if not isinstance(blob, dict):
            raise ValueError(f"Unexpected checkpoint container type: {type(blob)}")

        out: Dict[str, Any] = {}
        for key, val in blob.items():
            if not isinstance(key, str):
                continue
            k = key
            # strip typical top-level wrappers
            for prefix in ("module.", "model.", "critic.", "value_model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            out[k] = val
        return out

    def _maybe_prefix_remap(sd: Dict[str, Any], *, add_prefix: str, head_prefix: str = "score.") -> Dict[str, Any]:
        """
        Heuristic: if the model expects keys like '{add_prefix}embeddings....' but the checkpoint
        provides 'embeddings....', we can add the prefix for non-head keys.

        We do NOT touch the value head keys (default 'score.*').
        """
        out: Dict[str, Any] = {}
        for k, v in sd.items():
            if k.startswith(add_prefix) or k.startswith(head_prefix):
                out[k] = v
            else:
                out[f"{add_prefix}{k}"] = v
        return out

    # Resolve device
    if device is None or device == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer ALWAYS from base model (DS dirs typically lack tokenizer/config).
    base_for_tok_and_skeleton = _auto_base_model_path(ckpt_path)
    tok = AutoTokenizer.from_pretrained(base_for_tok_and_skeleton, trust_remote_code=True)
    if getattr(tok, "pad_token", None) is None:
        # Most causal LMs have eos_token; fall back safely if absent.
        tok.pad_token = getattr(tok, "eos_token", None) or getattr(tok, "unk_token", None) or tok.pad_token
        if getattr(tok, "pad_token", None) is None:
            # Last resort: set pad_token_id to 0 if tokenizer exposes it.
            try:
                tok.pad_token_id = 0
            except Exception:
                pass

    # Build OpenRLHF critic skeleton via the public factory.
    # NOTE: In this repo, CriticModel is defined inside _get_critic_model() and is NOT importable
    # as a top-level symbol. The supported API is get_llm_for_sequence_regression(..., model_type="critic").
    try:
        from openrlhf.models.model import get_llm_for_sequence_regression  # type: ignore
    except Exception as e:
        _die(f"failed to import get_llm_for_sequence_regression from openrlhf.models.model: {e}")


    if device.startswith("cuda") and torch.cuda.is_available():
        dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    else:
        dtype = torch.float32

    # Base src: if ckpt_path has config.json we can load base model from there,
    # otherwise load from provided base_for_tok_and_skeleton (critic base HF dir).
    base_src = ckpt_path if _has_hf_config(ckpt_path) else base_for_tok_and_skeleton

    # Build model with score head. We load from base_src then (if DS ckpt) overwrite with mp_rank state_dict.
    critic = get_llm_for_sequence_regression(
        base_src,
        "critic",
        bf16=bool(device.startswith("cuda") and torch.cuda.is_available()),
        load_in_4bit=False,
        normalize_reward=False,
        attn_implementation="flash_attention_2",
        ds_config=None,
        init_value_head=False,
        value_head_prefix="score",
        device_map=None,
        packing_samples=False,
    )

    # If DeepSpeed ckpt: locate mp_rank state and load into skeleton
    if not _has_hf_config(ckpt_path):
        from glob import glob

        cand = sorted(glob(os.path.join(ckpt_path, "mp_rank_*_model_states.pt")))
        if not cand:
            cand = sorted(glob(os.path.join(ckpt_path, "*", "mp_rank_*_model_states.pt")))
        if not cand:
            raise FileNotFoundError(f"No config.json and no mp_rank_*_model_states.pt found in: {ckpt_path}")

        state_blob = torch.load(cand[0], map_location="cpu")
        state_dict = _extract_state_dict(state_blob)

        # First attempt: load as-is.
        try:
            missing, unexpected = critic.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            msg = str(e)
            if "size mismatch" in msg and "score.weight" in msg:
                _die(
                    "MAPPO critic load failed due to shape mismatch on score.weight. "
                    "This almost always means you built the critic skeleton from the WRONG base model. "
                    "If your critic was trained on mmBERT-base (hidden=768), pass "
                    "--mappo_critic_base_ckpt <CRITIC_BASE_CKPT_PATH> (or your actual critic base). "
                    f"Original error: {msg}"
                )
            raise

        # If a large fraction of weights didn't land due to wrapper prefix mismatch, try prefix remap.
        # We pick the most likely wrapper prefix by inspecting the model's own state_dict keys.
        try:
            model_keys = list(critic.state_dict().keys())
        except Exception:
            model_keys = []

        pref_candidates: list[str] = []
        # Prefer the model's actual base_model_prefix (e.g., "bert.", "model.", "transformer.", ...)
        bmp = getattr(critic, "base_model_prefix", None)
        if isinstance(bmp, str) and bmp and any(k.startswith(bmp + ".") for k in model_keys):
            pref_candidates.append(bmp + ".")
        # Keep historical fallbacks
        if any(k.startswith("model.") for k in model_keys):
            pref_candidates.append("model.")
        if any(k.startswith("base_model.") for k in model_keys):
            pref_candidates.append("base_model.")

        if pref_candidates and (missing or unexpected):
            best_missing = missing
            best_unexpected = unexpected
            # Simple improvement criterion: fewer missing keys.
            best_count = len(best_missing) if best_missing is not None else 10**18

            for pref in pref_candidates:
                remapped = _maybe_prefix_remap(state_dict, add_prefix=pref, head_prefix="score.")
                try:
                    m2, u2 = critic.load_state_dict(remapped, strict=False)
                except Exception:
                    continue
                if m2 is not None and len(m2) < best_count:
                    best_count = len(m2)
                    best_missing, best_unexpected = m2, u2

            missing, unexpected = best_missing, best_unexpected

        if missing:
            _eprint(f"[mappo_v_critic] missing keys: {len(missing)} (sample: {missing[:5]})")
        if unexpected:
            _eprint(f"[mappo_v_critic] unexpected keys: {len(unexpected)} (sample: {unexpected[:5]})")

    critic.eval()
    critic.to(device)

    @torch.no_grad()
    def _score_text(text: str) -> float:
        inputs = tok(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # OpenRLHF critic forward requires action_mask (if None it asserts return_output=True).
        # For our scalar V(s) estimation, using a width-1 mask is stable across encoder/decoder models.
        bsz = int(inputs["input_ids"].size(0))
        action_mask = torch.ones((bsz, 1), device=inputs["input_ids"].device, dtype=torch.long)
        values = critic(action_mask=action_mask, **inputs)

        # Some implementations may return tuples; normalize to a Tensor
        if isinstance(values, (tuple, list)) and len(values) > 0:
            values = values[0]

        # values is typically [B, 1] (masked action-values). Sum over width -> scalar.
        if hasattr(values, "dim") and values.dim() == 2:
            values = values.sum(dim=1)
        else:
            values = values.view(values.size(0), -1)[:, 0]
        return float(values[0].float().cpu().item())

    def v_critic(bucket: Mapping[str, Any]) -> float:
        # MUST match metrics.credit_mode=mappo_v expectation: v_critic(bucket)
        restart = bucket.get("restart", {})
        if not isinstance(restart, Mapping):
            raise ValueError("bucket['restart'] missing or invalid")

        question = str(restart.get("question", ""))
        roles_topo = list(restart.get("roles_topo", []))

        prefix = restart.get("role_outputs_prefix", {})
        if not isinstance(prefix, Mapping):
            prefix = {}

        depth = len(prefix)
        next_role = roles_topo[depth] if (0 <= depth < len(roles_topo)) else (roles_topo[-1] if roles_topo else "actor")
        role_id_map = {r: i for i, r in enumerate(roles_topo)}

        # rollout_generator._compose_mappo_state_text signature evolved across forks.
        # Pass a superset of kwargs; _call_by_signature will keep only accepted ones.
        topo_so_far = list(roles_topo[:depth])
        role_outputs = {str(k): str(v) for k, v in dict(prefix).items() if v is not None}
        next_role_id = int(role_id_map.get(next_role, depth if depth >= 0 else 0))
        num_roles = int(len(roles_topo))

        state_text = _call_by_signature(
            _compose_mappo_state_text,
            # Newer signature (preferred)
            question=question,
            topo_so_far=topo_so_far,
            role_outputs=role_outputs,
            next_role=next_role,
            next_role_id=next_role_id,
            num_roles=num_roles,
            depth=depth,
            # Older signature (kept for backward compat)
            role_outputs_prefix=dict(prefix),
            roles_topo=roles_topo,
            role_id_map=role_id_map,
            max_len=max_len,
        )
        return float(_score_text(str(state_text)))

    _MAPPO_V_CRITIC_CACHE[cache_key] = v_critic
    return v_critic


# -----------------------------------------------------------------------------
# Subcommand: build-buckets
# -----------------------------------------------------------------------------

def _cmd_build_buckets(args: argparse.Namespace) -> None:
    # method is paper-metadata; keep it always defined to avoid downstream None surprises.
    if not getattr(args, "method", None):
        args.method = "unknown"
        _eprint("[c3_analysis][warn] --method missing; defaulting to 'unknown'.")

    replay_mod, buckets_mod, _metrics_mod = _import_analysis_modules()

    analysis_cfg = _try_load_yaml(Path(args.analysis_yaml)) if args.analysis_yaml else {}
    defaults_section = (
        ("credit" if args.target_role.lower() == "actor" else "influence")
        if args.defaults_section == "auto"
        else args.defaults_section
    )
    defaults = analysis_cfg.get(defaults_section, {})
    if not isinstance(defaults, dict):
        defaults = {}

    num_instances = args.limit if args.limit is not None else int(_cfg_get(defaults, ["num_instances"], 200))
    num_candidates = args.num_candidates if args.num_candidates is not None else int(_cfg_get(defaults, ["num_candidates"], 8))
    num_completions = args.num_completions if args.num_completions is not None else int(_cfg_get(defaults, ["num_completions"], 1))

    decoding: Dict[str, Any] = {}
    y_dec = _cfg_get(defaults, ["decoding"], {})
    if isinstance(y_dec, dict):
        decoding.update(y_dec)
    if args.decoding_json:
        try:
            decoding.update(json.loads(args.decoding_json))
        except Exception as e:
            _die(f"--decoding_json must be valid JSON object: {e}")

    prefix_decoding: Dict[str, Any] = {}
    y_pdec = _cfg_get(defaults, ["prefix_decoding"], {})
    if isinstance(y_pdec, dict):
        prefix_decoding.update(y_pdec)
    if args.prefix_decoding_json:
        try:
            prefix_decoding.update(json.loads(args.prefix_decoding_json))
        except Exception as e:
            _die(f"--prefix_decoding_json must be valid JSON object: {e}")

    inc_real_cli = _as_bool_tri(args.include_real_as_j0)
    include_real_as_j0 = (
        bool(inc_real_cli) if inc_real_cli is not None else bool(_cfg_get(defaults, ["fidelity", "include_real_as_j0"], False))
    )

    num_extra_v_samples = (
        int(args.num_extra_v_samples)
        if args.num_extra_v_samples is not None
        else int(_cfg_get(defaults, ["fidelity", "num_extra_samples_for_V"], 0))
    )
    num_extra_v_samples = max(int(num_extra_v_samples), 0)

    runner = _make_runner(args, replay_mod, decode_defaults=prefix_decoding)

    try:
        ReplayConfig = getattr(replay_mod, "ReplayConfig", None)
        if ReplayConfig is None:
            _die("replay.py must define ReplayConfig.")

        restart_states = _iter_restart_states(
            runner,
            num_instances=num_instances,
            seed=args.seed,
            split=args.split,
            task=args.task,
            target_role=args.target_role,
            prefix_decoding=prefix_decoding or None,
        )

        out_jsonl = Path(args.out)
        _ensure_parent_dir(out_jsonl)

        write_buckets_jsonl = getattr(buckets_mod, "write_buckets_jsonl", None)
        if write_buckets_jsonl is None:
            _die("buckets.py must define write_buckets_jsonl(path, buckets_iter, overwrite=...).")

        run_bucket = getattr(runner, "run_bucket", None)
        if not callable(run_bucket):
            _die("ReplayRunner must define run_bucket(restart_state, cfg, forced_actions=None).")

        next_role_cli = args.next_role
        record_next_teammate_cli = args.record_next_teammate

        n_written = 0

        def _gen() -> Iterator[Dict[str, Any]]:
            nonlocal n_written
            cfg_obj = None
            n = 0

            for rs in restart_states:
                roles_topo = getattr(rs, "roles_topo", None)
                if not isinstance(roles_topo, list) or not roles_topo:
                    _die("RestartState missing roles_topo; cannot canonicalize roles.")

                if cfg_obj is None:
                    canonical_target = _canonical_role(args.target_role, roles_topo)
                    canonical_next = _canonical_role(next_role_cli, roles_topo) if next_role_cli else _default_next_role(canonical_target, roles_topo)

                    if record_next_teammate_cli is None:
                        record_next = bool(_cfg_get(defaults, ["record_next_teammate"], canonical_target.lower() == "reasoner"))
                    else:
                        record_next = bool(record_next_teammate_cli)

                    if record_next and not canonical_next:
                        _die("record_next_teammate=True but next_role cannot be inferred; pass --next_role explicitly.")

                    cfg_obj = ReplayConfig(
                        target_role=canonical_target,
                        num_candidates=int(num_candidates),
                        num_completions_per_candidate=int(num_completions),
                        decoding=dict(decoding),
                        record_next_teammate=bool(record_next),
                        next_role=canonical_next,
                        include_real_as_j0=bool(include_real_as_j0),
                        num_extra_v_samples=int(num_extra_v_samples),
                        prefix_decoding=dict(prefix_decoding),
                    )

                bucket_obj = run_bucket(rs, cfg_obj, forced_actions=None)
                n_written += 1
                yield _as_dict(bucket_obj)

                n += 1
                if n >= num_instances:
                    break

        _call_by_signature(write_buckets_jsonl, path=str(out_jsonl), buckets_iter=_gen(), overwrite=bool(args.overwrite))

        if n_written == 0:
            _die(
                f"No buckets were written. Check task/split mismatch or dataset availability. "
                f"(task={args.task}, split={args.split}, target_role={args.target_role})"
            )

        _eprint(
            f"[c3_analysis] Wrote buckets: {out_jsonl} "
            f"(method={args.method}, target_role={args.target_role}, limit={num_instances}, "
            f"candidates={num_candidates}, completions={num_completions}, "
            f"include_real_as_j0={include_real_as_j0}, num_extra_v_samples={num_extra_v_samples})"
        )

    finally:
        close = getattr(runner, "close", None)
        if callable(close):
            close()


# -----------------------------------------------------------------------------
# Subcommand: credit
# -----------------------------------------------------------------------------

def _cmd_credit(args: argparse.Namespace) -> None:
    _replay_mod, buckets_mod, metrics_mod = _import_analysis_modules()

    read_buckets_jsonl = getattr(buckets_mod, "read_buckets_jsonl", None)
    validate_bucket = getattr(buckets_mod, "validate_bucket", None)
    if read_buckets_jsonl is None or validate_bucket is None:
        _die("buckets.py must define read_buckets_jsonl + validate_bucket.")

    credit_var_report = getattr(metrics_mod, "credit_var_report", None)
    build_fidelity_pairs = getattr(metrics_mod, "build_fidelity_pairs", None)
    credit_fidelity = getattr(metrics_mod, "credit_fidelity", None)
    if credit_fidelity is None:
        _die("metrics.py must define credit_fidelity (and ideally credit_var_report/build_fidelity_pairs).")

    buckets_path = Path(args.bucket)
    if not buckets_path.exists():
        _die(f"Buckets file not found: {str(buckets_path)!r}")

    def _validated_iter() -> Iterator[Dict[str, Any]]:
        for b in read_buckets_jsonl(str(buckets_path)):
            validate_bucket(b)
            yield b

    v_critic: Any = None
    if args.mode == "mappo_v":
        if not args.mappo_critic_ckpt:
            _die("--mappo_critic_ckpt is required when --mode mappo_v.")

        # Prefer explicit critic base ckpt if provided; fallback to policy_ckpt.
        critic_base = getattr(args, "mappo_critic_base_ckpt", None) or getattr(args, "policy_ckpt", None)
        if critic_base is None:
            _die("--mode mappo_v requires either --mappo_critic_base_ckpt or --policy_ckpt (HF dir for tokenizer/config).")

        v_critic = _load_mappo_v_critic_callable(
            args.mappo_critic_ckpt,
            device=args.critic_device,
            base_model_path=critic_base,
            max_len=int(args.critic_max_len),
        )

    if not callable(credit_var_report):
        _die("metrics.credit_var_report is missing; please implement Step3 metrics.py.")
    var_report = _call_by_signature(credit_var_report, buckets_iter=_validated_iter(), mode=args.mode, v_critic=v_critic)

    if not callable(build_fidelity_pairs):
        _die("metrics.build_fidelity_pairs is missing; please implement Step3 metrics.py.")
    pairs_obj = _call_by_signature(
        build_fidelity_pairs,
        buckets_iter=_validated_iter(),
        mode=args.mode,
        v_critic=v_critic,
        estimate_v_by_extra_samples=bool(args.estimate_v_by_extra_samples),
    )

    pairs_real: Optional[Sequence[Tuple[float, float]]] = None
    pairs_all: Optional[Sequence[Tuple[float, float]]] = None

    if isinstance(pairs_obj, Mapping):
        if isinstance(pairs_obj.get("real_only"), list):
            pairs_real = pairs_obj["real_only"]  # type: ignore[assignment]
        if isinstance(pairs_obj.get("all_candidates"), list):
            pairs_all = pairs_obj["all_candidates"]  # type: ignore[assignment]
    elif isinstance(pairs_obj, (tuple, list)):
        if len(pairs_obj) >= 1 and isinstance(pairs_obj[0], list):
            pairs_real = pairs_obj[0]  # type: ignore[assignment]
        if len(pairs_obj) >= 2 and isinstance(pairs_obj[1], list):
            pairs_all = pairs_obj[1]  # type: ignore[assignment]
    elif isinstance(pairs_obj, list):
        pairs_all = pairs_obj  # type: ignore[assignment]

    fidelity: Dict[str, Any] = {
        "estimate_v_by_extra_samples": bool(args.estimate_v_by_extra_samples),
        "spearman_real_only": float(credit_fidelity(pairs_real)) if pairs_real else None,
        "n_real_only": len(pairs_real) if pairs_real else 0,
        "spearman_all_candidates": float(credit_fidelity(pairs_all)) if pairs_all else None,
        "n_all_candidates": len(pairs_all) if pairs_all else 0,
    }

    out = {
        "kind": "credit_metrics",
        "mode": args.mode,
        "var": var_report,
        "fidelity": fidelity,
        "meta": {
            "bucket": str(buckets_path),
            "mappo_critic_ckpt": args.mappo_critic_ckpt,
            "mappo_critic_base_ckpt": getattr(args, "mappo_critic_base_ckpt", None),
            "policy_ckpt": getattr(args, "policy_ckpt", None),
        },
    }

    if args.out:
        _write_json(Path(args.out), out)
        _eprint(f"[c3_analysis] Wrote credit metrics: {args.out}")
    else:
        out2 = _sanitize_for_json(out)
        print(json.dumps(out2, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))

    # Optional: dump per-candidate details for calibration/casebook appendices.
    if args.out_details:
        aggregate_candidate_returns = getattr(buckets_mod, "aggregate_candidate_returns", None)
        if not callable(aggregate_candidate_returns):
            _die("buckets.py must define aggregate_candidate_returns for --out_details.")

        # Prefer metric helpers if available; otherwise use local fallbacks.
        credit_from_barR = getattr(metrics_mod, "_credit_from_barR", None)
        baseline_loo = getattr(metrics_mod, "_baseline_loo", None)
        weighted_mean = getattr(metrics_mod, "_weighted_mean", None)
        as_weights = getattr(metrics_mod, "_as_weights", None)

        def _clip_int(x: Any, lo: int, hi: int, default: int) -> int:
            try:
                v = int(x)
            except Exception:
                return default
            return max(lo, min(hi, v))

        def _credit_idx(meta: Mapping[str, Any], n_actions: int) -> np.ndarray:
            if n_actions <= 0:
                return np.zeros((0,), dtype=np.int64)
            credit_n = _clip_int(meta.get("credit_n", n_actions), 0, n_actions, n_actions)
            return np.arange(credit_n, dtype=np.int64)

        def _v_extra_idx(meta: Mapping[str, Any], n_actions: int) -> np.ndarray:
            if n_actions <= 0:
                return np.zeros((0,), dtype=np.int64)
            start = _clip_int(meta.get("v_extra_start", n_actions), 0, n_actions, n_actions)
            n = _clip_int(meta.get("v_extra_n", 0), 0, n_actions, 0)
            end = min(n_actions, start + n)
            if end <= start:
                return np.zeros((0,), dtype=np.int64)
            return np.arange(start, end, dtype=np.int64)

        def _as_weights_local(counts: Optional[np.ndarray], n: int) -> np.ndarray:
            if counts is None:
                return np.ones((n,), dtype=np.float64)
            w = np.asarray(counts, dtype=np.float64)
            if w.shape != (n,):
                w = np.ones((n,), dtype=np.float64)
            if not np.all(np.isfinite(w)):
                w = np.ones((n,), dtype=np.float64)
            w = np.maximum(w, 0.0)
            if float(np.sum(w)) <= 0.0:
                w = np.ones((n,), dtype=np.float64)
            return w

        def _weighted_mean_local(values: np.ndarray, weights: np.ndarray) -> float:
            denom = float(np.sum(weights))
            if denom <= 0.0 or not np.isfinite(denom):
                return float(np.mean(values)) if values.size else 0.0
            num = float(np.sum(values * weights))
            if not np.isfinite(num):
                return float(np.mean(values)) if values.size else 0.0
            return num / denom

        def _baseline_loo_local(values: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
            values = np.asarray(values, dtype=np.float64)
            n = int(values.size)
            if n <= 1:
                return values.astype(np.float64, copy=True)
            w = _as_weights_local(weights, n)
            sw = float(np.sum(w))
            sv = float(np.sum(w * values))
            outb = np.empty((n,), dtype=np.float64)
            for j in range(n):
                denom = sw - float(w[j])
                if denom <= 0.0 or not np.isfinite(denom):
                    outb[j] = float(values[j])
                else:
                    outb[j] = (sv - float(w[j]) * float(values[j])) / denom
            return outb

        def _credit_from_barR_local(
            barR: np.ndarray,
            mode: str,
            *,
            counts: Optional[np.ndarray],
            v_critic_local: Any,
            bucket_local: Mapping[str, Any],
        ) -> np.ndarray:
            barR = np.asarray(barR, dtype=np.float64)
            if mode in ("c3_loo", "magrpo_rloo"):
                base = _baseline_loo_local(barR, weights=counts)
                return barR - base
            if mode in ("c3_full_mean", "magrpo_mean"):
                w = _as_weights_local(counts, int(barR.size))
                m = _weighted_mean_local(barR, w)
                return barR - float(m)
            if mode == "mappo_v":
                if v_critic_local is None:
                    raise ValueError("mode='mappo_v' requires v_critic")
                if callable(v_critic_local):
                    v = float(v_critic_local(bucket_local))
                else:
                    v = float(v_critic_local)
                return barR - v
            raise ValueError(f"Unknown mode: {mode}")

        p_out = Path(args.out_details)
        _ensure_parent_dir(p_out)
        n_rows = 0
        with p_out.open("w", encoding="utf-8") as fh:
            for b in _validated_iter():
                meta = b.get("meta", {})
                if not isinstance(meta, Mapping):
                    meta = {}

                barR_all, counts_all = aggregate_candidate_returns(b)
                n_actions = int(barR_all.size)
                if n_actions == 0:
                    continue

                cidx = _credit_idx(meta, n_actions)
                if cidx.size == 0:
                    continue

                barR = barR_all[cidx]
                counts = counts_all[cidx] if counts_all is not None else None

                if callable(credit_from_barR):
                    A = credit_from_barR(barR, args.mode, counts=counts, v_critic=v_critic, bucket=b)
                else:
                    A = _credit_from_barR_local(barR, args.mode, counts=counts, v_critic_local=v_critic, bucket_local=b)

                # Always compute LOO delta as the calibration target (paper LOO estimand).
                if callable(baseline_loo):
                    b_loo = baseline_loo(barR, weights=counts)
                else:
                    b_loo = _baseline_loo_local(barR, weights=counts)
                delta_loo = barR - b_loo

                # V-hat for appendix auditing (same choice as fidelity pipeline).
                if bool(args.estimate_v_by_extra_samples):
                    vidx = _v_extra_idx(meta, n_actions)
                else:
                    vidx = np.zeros((0,), dtype=np.int64)

                if vidx.size > 0:
                    vals = barR_all[vidx]
                    wts = counts_all[vidx] if counts_all is not None else None
                else:
                    vals = barR
                    wts = counts

                if callable(as_weights) and callable(weighted_mean):
                    V_hat = float(weighted_mean(vals, as_weights(wts, int(vals.size))))
                else:
                    V_hat = _weighted_mean_local(vals, _as_weights_local(wts, int(vals.size)))

                real_j = int(meta.get("real_j", 0)) if str(meta.get("real_j", "")).strip() != "" else 0
                pos_of = {int(j): int(i) for i, j in enumerate(cidx.tolist())}
                cands = b.get("candidates", [])
                if not isinstance(cands, Sequence):
                    cands = []

                for p, j0 in enumerate(cidx.tolist()):
                    cand_obj: Mapping[str, Any] = cands[int(j0)] if int(j0) < len(cands) and isinstance(cands[int(j0)], Mapping) else {}
                    row = {
                        "bucket_id": b.get("bucket_id"),
                        "question_id": b.get("question_id"),
                        "target_role": b.get("target_role"),
                        "method": meta.get("method"),
                        "mode": args.mode,
                        "candidate_j": int(j0),
                        "is_real_j": bool(int(j0) == real_j),
                        "action_text": cand_obj.get("action_text"),
                        "returns_count": int(counts[p]) if counts is not None else None,
                        "barR": float(barR[p]),
                        "baseline_loo": float(b_loo[p]),
                        "deltaR_loo": float(delta_loo[p]),
                        "A_hat": float(A[p]),
                        "V_hat": float(V_hat),
                        "deltaR_vhat": float(barR[p] - V_hat),
                        "credit_n": int(len(cidx)),
                        "real_j": int(real_j),
                        "estimate_v_by_extra_samples": bool(args.estimate_v_by_extra_samples),
                    }
                    fh.write(json.dumps(_sanitize_for_json(row), ensure_ascii=False) + "\n")
                    n_rows += 1

        _eprint(f"[c3_analysis] Wrote credit details JSONL: {args.out_details} (rows={n_rows})")


# -----------------------------------------------------------------------------
# Subcommand: influence
# -----------------------------------------------------------------------------

def _cmd_influence(args: argparse.Namespace) -> None:
    _replay_mod, buckets_mod, metrics_mod = _import_analysis_modules()

    read_buckets_jsonl = getattr(buckets_mod, "read_buckets_jsonl", None)
    validate_bucket = getattr(buckets_mod, "validate_bucket", None)
    if read_buckets_jsonl is None or validate_bucket is None:
        _die("buckets.py must define read_buckets_jsonl + validate_bucket.")

    influence_report = getattr(metrics_mod, "influence_report", None)
    if not callable(influence_report):
        _die("metrics.py must define influence_report (Step3).")

    buckets_path = Path(args.bucket)
    if not buckets_path.exists():
        _die(f"Buckets file not found: {str(buckets_path)!r}")

    def _validated_iter() -> Iterator[Dict[str, Any]]:
        for b in read_buckets_jsonl(str(buckets_path)):
            validate_bucket(b)
            yield b

    out_points_fh = None
    if args.out_points:
        p = Path(args.out_points)
        _ensure_parent_dir(p)
        out_points_fh = p.open("w", encoding="utf-8")

    report = _call_by_signature(
        influence_report,
        buckets_iter=_validated_iter(),
        top_k=int(args.top_k),
        alpha=float(args.alpha),
        out_points_fh=out_points_fh,
    )

    if out_points_fh is not None:
        out_points_fh.close()

    out = {
        "kind": "influence_metrics",
        "mi": report,
        "params": {"top_k": int(args.top_k), "alpha": float(args.alpha)},
        "meta": {"bucket": str(buckets_path), "out_points": args.out_points},
    }

    if args.out:
        _write_json(Path(args.out), out)
        _eprint(f"[c3_analysis] Wrote influence metrics: {args.out}")
    else:
        out2 = _sanitize_for_json(out)
        print(json.dumps(out2, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


# -----------------------------------------------------------------------------
# Subcommand: latex
# -----------------------------------------------------------------------------

def _extract(d: Mapping[str, Any], keys: Sequence[str]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _fmt_num(x: Any, *, ndigits: int = 3) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "—"


def _cmd_latex(args: argparse.Namespace) -> None:
    credit = json.loads(_read_text(Path(args.credit_json)))
    influence = json.loads(_read_text(Path(args.influence_json))) if args.influence_json else None

    fid_real = _extract(credit, ("fidelity", "spearman_real_only"))
    fid_all = _extract(credit, ("fidelity", "spearman_all_candidates"))
    var_mean = _extract(credit, ("var", "mean"))
    mi_mean = _extract(influence, ("mi", "mean")) if influence is not None else None

    row = (
        f"{args.method} & "
        f"{_fmt_num(fid_real)} & "
        f"{_fmt_num(fid_all)} & "
        f"{_fmt_num(var_mean)} & "
        f"{_fmt_num(mi_mean)} \\\\"
    )

    if args.out:
        _write_text(Path(args.out), row + "\n")
        _eprint(f"[c3_analysis] Wrote LaTeX row: {args.out}")
    else:
        print(row)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m c3.analysis.c3_analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # build-buckets
    pb = sub.add_parser("build-buckets", help="Generate buckets.jsonl via replay.")

    # aliases to keep bash scripts stable across refactors
    pb.add_argument("--task", "--task_yaml", dest="task", required=True, help="Task name or task YAML path.")
    pb.add_argument("--split", "--suite", dest="split", required=True, help="Eval suite / split name (case-insensitive).")
    pb.add_argument("--policy_ckpt", "--policy", dest="policy_ckpt", required=True, help="Policy checkpoint path/ID.")
    pb.add_argument("--method", "--algo", dest="method", default=None, help="Method label written into bucket.meta['method'] (optional).")

    pb.add_argument("--target_role", "--target", dest="target_role", required=True, help="Target role (case-insensitive), e.g., actor/reasoner.")
    pb.add_argument("--next_role", "--next", dest="next_role", default=None, help="Next role for Influence; default is topo successor of target_role.")

    pb.add_argument(
        "--record_next_teammate",
        "--record-next-teammate",
        dest="record_next_teammate",
        default=None,
        action="store_true",
        help="If set, record next teammate actions. If omitted, derived from YAML/defaults.",
    )

    # sizes
    pb.add_argument("--limit", type=int, default=None, help="Number of contexts (buckets) to generate.")
    pb.add_argument("--num_candidates", type=int, default=None, help="Credit candidates per bucket (excluding V-extra).")
    pb.add_argument("--num_completions", type=int, default=None, help="Completions per candidate.")
    pb.add_argument("--seed", type=int, default=0, help="Sampling seed.")

    # fidelity knobs
    pb.add_argument(
        "--include_real_as_j0",
        default="auto",
        choices=["auto", "true", "false"],
        help="Whether to include one reference action at j=0 (writes meta['real_j']=0).",
    )
    pb.add_argument(
        "--num_extra_v_samples",
        type=int,
        default=None,
        help="Extra actions used only for V(h) estimation (ignored by credit_var/influence).",
    )

    # decoding defaults / overrides
    pb.add_argument("--analysis_yaml", "--analysis-yaml", dest="analysis_yaml", default=None, help="Optional analysis.yaml for defaults.")
    pb.add_argument(
        "--defaults_section",
        default="auto",
        choices=["auto", "credit", "influence"],
        help="Which analysis.yaml section to use for defaults.",
    )
    pb.add_argument("--decoding_json", default=None, help="JSON dict overriding rollout decoding params.")
    pb.add_argument("--prefix_decoding_json", default=None, help="JSON dict overriding prefix decoding params.")

    # runner controls
    pb.add_argument("--device", default=None, help="Device for policy load (if supported).")
    pb.add_argument("--engine", "--inference_engine", dest="engine", default="auto", choices=["auto", "hf", "vllm"], help="Policy engine (if supported).")
    pb.add_argument("--cache_dir", default=None, help="Dataset cache dir (if supported).")
    pb.add_argument("--tensor_parallel_size", "--tp", dest="tensor_parallel_size", type=int, default=None, help="vLLM tensor parallel size.")

    pb.add_argument("--out", "--out_jsonl", dest="out", required=True, help="Output buckets.jsonl path.")
    pb.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    pb.add_argument("--runner_factory", default=None, help="Optional factory: 'pkg.mod:make_runner'.")
    pb.set_defaults(func=_cmd_build_buckets)

    # credit
    pc = sub.add_parser("credit", help="Compute Fidelity/Var from buckets.jsonl.")
    pc.add_argument("--bucket", "--buckets_jsonl", "--bucket_jsonl", dest="bucket", required=True, help="Input buckets.jsonl path.")
    pc.add_argument(
        "--mode",
        required=True,
        choices=["c3_loo", "c3_full_mean", "magrpo_mean", "magrpo_rloo", "mappo_v"],
        help="Credit definition to evaluate.",
    )
    pc.add_argument(
        "--estimate_v_by_extra_samples",
        action="store_true",
        help="Use V-extra segment (meta['v_extra_*']) for V(h) if available; else fallback to credit segment.",
    )
    pc.add_argument("--mappo_critic_ckpt", default=None, help="MAPPO value critic checkpoint (required for mode=mappo_v).")

    pc.add_argument(
        "--mappo_critic_base_ckpt",
        default=None,
        help="(mappo_v) HF dir for critic BASE model (tokenizer/config). "
             "Required when critic arch != policy arch (e.g. critic=mmBERT-base, policy=Qwen3-4B).",
    )

    # Base policy checkpoint for DS critic loading fallback (tokenizer + model skeleton)
    pc.add_argument(
        "--policy_ckpt",
        "--policy",
        dest="policy_ckpt",
        default=None,
        help="Fallback HF dir used to load tokenizer/model skeleton when mappo_critic_ckpt is a DeepSpeed checkpoint "
             "(historically: <run_root>/final_hf). Prefer --mappo_critic_base_ckpt when critic!=policy.",
    )

    pc.add_argument("--critic_device", default=None, help="Device for critic (optional).")
    pc.add_argument("--critic_max_len", type=int, default=4096, help="Max length for critic state_text compose / tokenize.")
    pc.add_argument("--out", "--out_json", dest="out", default=None, help="Write metrics JSON to this path (else print).")
    pc.add_argument(
        "--out_details",
        default=None,
        help="Optional JSONL path for per-candidate rows (barR, A_hat, deltaR_loo, deltaR_vhat) used by calibration/casebook.",
    )
    pc.set_defaults(func=_cmd_credit)

    # influence
    pi = sub.add_parser("influence", help="Compute conditional MI Influence from buckets.jsonl.")
    pi.add_argument("--bucket", "--buckets_jsonl", "--bucket_jsonl", dest="bucket", required=True, help="Input buckets.jsonl (should contain next_actions).")
    pi.add_argument("--top_k", type=int, default=64, help="Top-K frequent Y symbols; rest mapped to __OTHER__.")
    pi.add_argument("--alpha", type=float, default=1e-2, help="Additive smoothing parameter.")
    pi.add_argument("--out_points", default=None, help="Optional per-bucket MI points JSONL output.")
    pi.add_argument("--out", "--out_json", dest="out", default=None, help="Write metrics JSON to this path (else print).")
    pi.set_defaults(func=_cmd_influence)

    # latex
    pl = sub.add_parser("latex", help="Format metric JSONs into a LaTeX tabular row.")
    pl.add_argument("--method", required=True, help="Row label (e.g., C3, MAPPO, MAGRPO).")
    pl.add_argument("--credit_json", required=True, help="Output JSON from credit subcommand.")
    pl.add_argument("--influence_json", default=None, help="Output JSON from influence subcommand (optional).")
    pl.add_argument("--out", default=None, help="Write LaTeX row to file (else print).")
    pl.set_defaults(func=_cmd_latex)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
