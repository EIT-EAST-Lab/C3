"""Budget ledger writer for Appendix B reporting.

We record *terminal evaluator call* accounting in a stable, machine-readable JSONL file
under each run directory. This enables third parties to audit that training obeys the
paper's evaluator-call budget (B=8 in the paper runs) and to reproduce reporting.

Design goals:
- No external dependencies.
- Append-only JSONL (one record per training rollout-generation call).
- Defensive: never crash training if the ledger cannot be written.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def append_ledger(
    run_dir: str,
    record: Dict[str, Any],
    *,
    filename: str = "budget_ledger.jsonl",
) -> None:
    """Append one JSON record to <run_dir>/<filename>.

    The function is best-effort: exceptions are caught and suppressed to avoid
    disrupting training. Callers should pass only JSON-serializable values; any
    non-serializable values are stringified.

    Args:
        run_dir: Training run directory that contains checkpoints/logs.
        record: Dictionary with budget fields.
        filename: Output ledger file name (default: budget_ledger.jsonl).
    """
    if not run_dir:
        return

    try:
        os.makedirs(run_dir, exist_ok=True)
    except Exception:
        # If the directory cannot be created, do not crash training.
        return

    # Add minimal stable envelope.
    payload: Dict[str, Any] = {}
    payload.update(record or {})

    # Normalize common numeric fields if present.
    for k in (
        "global_step",
        "epoch_idx",
        "iter_in_epoch",
        "n_questions_in_batch",
        "n_samples_per_prompt",
        "total_eval_calls",
    ):
        if k in payload:
            payload[k] = _safe_int(payload.get(k), default=-1)

    # Timestamp for offline debugging / sorting; not used by determinism.
    payload.setdefault("ts_unix", int(time.time()))

    # Ensure JSON-serializable values (stringify unknowns).
    def _jsonable(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        if isinstance(v, dict):
            return {str(kk): _jsonable(vv) for kk, vv in v.items()}
        return str(v)

    payload = _jsonable(payload)  # type: ignore[assignment]

    path = os.path.join(run_dir, filename)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        # Best-effort: never crash training because of ledger IO.
        return


def make_budget_record(
    *,
    global_step: Optional[int],
    epoch_idx: Optional[int],
    iter_in_epoch: Optional[int],
    marl_algorithm: str,
    n_questions_in_batch: int,
    n_samples_per_prompt: int,
    roles_topo: Optional[list[str]] = None,
    fanout: Optional[list[int]] = None,
) -> Dict[str, Any]:
    """Create a normalized budget record dict.

    total_eval_calls is computed as n_questions_in_batch * n_samples_per_prompt.

    Args:
        global_step/epoch_idx/iter_in_epoch: Training loop indices (best-effort).
        marl_algorithm: Lowercase algorithm identifier.
        n_questions_in_batch: Number of prompts in the rollout batch.
        n_samples_per_prompt: Terminal evaluator budget B per prompt for this call.
        roles_topo: Optional role topology list (C3/MAS).
        fanout: Optional per-role fanout list (C3/MAS).

    Returns:
        A dict ready to be passed to append_ledger().
    """
    nq = _safe_int(n_questions_in_batch, default=0)
    ns = _safe_int(n_samples_per_prompt, default=0)
    total = int(nq) * int(ns)

    rec: Dict[str, Any] = {
        "global_step": _safe_int(global_step, default=-1),
        "epoch_idx": _safe_int(epoch_idx, default=-1),
        "iter_in_epoch": _safe_int(iter_in_epoch, default=-1),
        "marl_algorithm": str(marl_algorithm or "").strip().lower(),
        "n_questions_in_batch": int(nq),
        "n_samples_per_prompt": int(ns),
        "total_eval_calls": int(total),
    }
    if roles_topo is not None:
        rec["roles_topo"] = list(roles_topo)
    if fanout is not None:
        rec["fanout"] = [int(x) for x in fanout]
    return rec
