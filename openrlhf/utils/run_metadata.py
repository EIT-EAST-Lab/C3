# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

"""Experiment run metadata utilities.

- Provide canonical run_dir under ckpt_path
- Stamp run_id into dumps
- Provide W&B tags/config without leaking secrets

Best-effort: failures should never crash training/eval.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def safe_slug(text: str) -> str:
    """Filesystem-safe slug."""
    s = str(text or "").strip()
    if not s:
        return "run"
    for ch in ["/", "\\", ":", "|", "\n", "\r", "\t"]:
        s = s.replace(ch, "_")
    # collapse spaces
    s = "_".join([p for p in s.split(" ") if p])
    return s[:200]


def _jsonable(x: Any) -> Any:
    """Best-effort JSON conversion."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return str(x)


def sanitize_args(args) -> Dict[str, Any]:
    """Convert argparse namespace to jsonable dict and mask secrets."""
    try:
        d = vars(args).copy()
    except Exception:
        return {}

    # avoid leaking secrets
    if "use_wandb" in d and isinstance(d["use_wandb"], str) and d["use_wandb"]:
        d["use_wandb"] = "***"

    return {str(k): _jsonable(v) for k, v in d.items()}


def init_run_artifacts(args) -> None:
    """Mutates args: set run_dir/run_id; set default dump paths (when enabled); write run_config.json.

    Call this on the driver before `train(args)`.
    """
    try:
        ckpt_path = str(getattr(args, "ckpt_path", "./ckpt"))
        run_name = safe_slug(getattr(args, "wandb_run_name", "run"))

        if not getattr(args, "run_dir", None):
            setattr(args, "run_dir", os.path.join(ckpt_path, "_runs", run_name))

        if not getattr(args, "run_id", None):
            # stable id (run_dir scopes uniqueness)
            setattr(args, "run_id", run_name)

        run_dir = Path(str(getattr(args, "run_dir")))
        run_dir.mkdir(parents=True, exist_ok=True)

        # ---- normalize default save_path to avoid clutter under run_dir / cwd ----
        # NOTE:
        # - --disable_ds_ckpt only disables DeepSpeed checkpoints.
        # - ActorModelActor.async_save_model() still saves final weights to args.save_path.
        # - Default save_path="./ckpt" is often relative to current working dir (may end up under run_dir),
        #   so we redirect the default to: <ckpt_path>/checkpoints/<run_id>/.
        try:
            sp = getattr(args, "save_path", None)
            sp_str = str(sp).strip() if sp is not None else ""
            if (not sp_str) or (sp_str == "./ckpt"):
                setattr(args, "save_path", os.path.join(ckpt_path, "checkpoints", str(getattr(args, "run_id"))))
            Path(str(getattr(args, "save_path"))).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # default dump paths only when feature is enabled
        dump_every = int(getattr(args, "dump_rollouts_every", 0) or 0)

        if dump_every > 0 and not getattr(args, "dump_rollouts_jsonl_path", None):
            setattr(args, "dump_rollouts_jsonl_path", str(run_dir / "train_rollouts.jsonl"))

        if str(getattr(args, "marl_algorithm", "")).lower().strip() == "c3":
            # Keep C3 batch dumps consistent with rollout dumps: only auto-enable when dumping is enabled.
            if dump_every > 0 and not getattr(args, "dump_c3_batch_data_path", None):
                setattr(args, "dump_c3_batch_data_path", str(run_dir / "c3_batch_data.jsonl"))

        eval_steps = int(getattr(args, "eval_steps", -1) or -1)
        if eval_steps > 0 and not getattr(args, "eval_dump_path", None):
            setattr(args, "eval_dump_path", str(run_dir / "eval_during_train.jsonl"))

        if bool(getattr(args, "eval_only", False)) and not getattr(args, "eval_dump_path", None):
            setattr(args, "eval_dump_path", str(run_dir / "eval_only.jsonl"))

        # write run config
        cfg = {
            "ts": float(time.time()),
            "hostname": socket.gethostname(),
            "argv": list(sys.argv),
            "run_id": str(getattr(args, "run_id")),
            "run_dir": str(run_dir),
            "args": sanitize_args(args),
        }
        (run_dir / "run_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # copy task yaml (best-effort)
        task_path = getattr(args, "c3_task", None)
        if task_path and isinstance(task_path, str) and os.path.exists(task_path):
            try:
                content = Path(task_path).read_text(encoding="utf-8")
                (run_dir / "c3_task.yaml").write_text(content, encoding="utf-8")
            except Exception:
                pass

    except Exception:
        # never crash
        return


def build_wandb_tags(args) -> List[str]:
    """Small, informative tags for W&B."""
    tags: List[str] = []
    try:
        alg = str(getattr(args, "marl_algorithm", "") or "").strip()
        if alg:
            tags.append(alg)
        mode = str(getattr(args, "policy_sharing_mode", "shared") or "shared")
        tags.append(f"policy:{mode}")

        rp = str(getattr(args, "reward_provider_cls", "") or "").strip()
        if rp:
            tags.append(f"reward:{rp}")

        task = getattr(args, "c3_task", None)
        if task:
            tags.append(f"task:{safe_slug(os.path.basename(str(task)))}")

        if bool(getattr(args, "eval_only", False)):
            tags.append("eval_only")
    except Exception:
        pass

    # de-dup
    out: List[str] = []
    for t in tags:
        if t and t not in out:
            out.append(t)
    return out


def build_wandb_config(args) -> Dict[str, Any]:
    """W&B config dict (sanitized)."""
    cfg = sanitize_args(args)
    # keep it small-ish
    for k in ["eval_dataset", "prompt_data"]:
        if k in cfg and isinstance(cfg[k], str) and len(cfg[k]) > 512:
            cfg[k] = cfg[k][:512] + "..."
    return cfg
