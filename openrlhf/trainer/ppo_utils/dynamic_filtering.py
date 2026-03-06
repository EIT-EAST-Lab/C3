# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

# Small epsilon used to avoid edge-triggering at exact min/max boundaries.
_EPS = 1e-6

# dyn_filter_update state:
#   - buf_groups: list of accepted prompt-groups (each group is a list of samples)
#   - n_total_prompts: total number of prompt-groups observed so far (accepted + rejected)
DynFilterState = Tuple[List[List[Any]], int]


def _scalar(x: Any) -> Optional[float]:
    """Best-effort extract a python float scalar from tensor/list/number; returns None if impossible."""
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        if x.numel() <= 0:
            return None
        return float(x.view(-1)[0].item())

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        return _scalar(x[0])

    # numpy scalar / tensor-like
    try:
        return float(x.item())
    except Exception:
        pass

    try:
        return float(x)
    except Exception:
        return None


def _get_info(sample: Any) -> Dict[str, Any]:
    """Return sample.info as a dict; never returns None."""
    info = getattr(sample, "info", None)
    return info if isinstance(info, dict) else {}


def _is_c3_like_rollout(rollout_samples: List[Any]) -> bool:
    """
    Heuristic: C3/MAS rollout samples usually carry C3-specific info keys.
    If true, missing question_id must be treated as fatal (fallback chunking corrupts Full-K).
    """
    if not rollout_samples:
        return False
    try:
        info0 = _get_info(rollout_samples[0])
        return ("c3_node_id" in info0) or ("is_leaf" in info0)
    except Exception:
        return False


def _get_qid(sample: Any) -> Optional[int]:
    """Extract question_id as int if present/parsable; otherwise None."""
    info = _get_info(sample)
    if "question_id" not in info:
        return None
    v = _scalar(info.get("question_id"))
    if v is None:
        return None
    return int(v)


def _get_is_leaf(sample: Any) -> Optional[bool]:
    """Extract is_leaf as bool if present/parsable; otherwise None."""
    info = _get_info(sample)
    if "is_leaf" not in info:
        return None
    v = _scalar(info.get("is_leaf"))
    if v is None:
        return None
    return bool(int(v) == 1)


def _chunk_by_k(rollout_samples: List[Any], k: int) -> List[List[Any]]:
    """
    Legacy fallback: chunk samples into groups of exactly K.
    Drops the last incomplete chunk (keeps historical semantics).
    """
    kk = max(1, int(k or 1))
    out: List[List[Any]] = []
    for i in range(0, len(rollout_samples), kk):
        g = rollout_samples[i : i + kk]
        if len(g) == kk:
            out.append(g)
    return out


def group_rollout_samples_by_prompt(rollout_samples: List[Any], k: int) -> List[List[Any]]:
    """
    Group rollout samples into prompt groups: List[List[sample]].

    Preferred path:
      - group by info['question_id'] (works for MAS / C3 tasks, and any generator that sets question_id)

    Fallback path:
      - chunk by K (legacy behavior for generators without question_id)

    Safety:
      - For C3/MAS-like rollouts (tree nodes), missing question_id is fatal:
        fallback chunk-by-K would corrupt Full-K invariants downstream.
    """
    if not rollout_samples:
        return []

    c3_like = _is_c3_like_rollout(rollout_samples)

    # Try prompt grouping by question_id. If ANY sample lacks question_id:
    #   - C3-like: fail-fast
    #   - otherwise: fallback to chunk-by-K
    groups: Dict[int, List[Any]] = {}
    order: List[int] = []

    for s in rollout_samples:
        q = _get_qid(s)
        if q is None:
            if c3_like:
                raise RuntimeError(
                    "[DynamicFiltering][FAIL-FAST] Missing question_id in rollout_samples for C3/MAS rollout. "
                    "Fallback chunk-by-K would corrupt Full-K invariants."
                )
            return _chunk_by_k(rollout_samples, k)

        if q not in groups:
            groups[q] = []
            order.append(q)
        groups[q].append(s)

    return [groups[q] for q in order]


def _extract_reward_scalar(sample: Any) -> Optional[float]:
    """Best-effort extract a scalar reward from sample.scores."""
    try:
        sc = getattr(sample, "scores", None)
    except Exception:
        return None

    if sc is None:
        return None

    # Common case: torch tensor
    if isinstance(sc, torch.Tensor):
        if sc.numel() <= 0:
            return None
        return float(sc.view(-1)[0].item())

    # Some code uses list/tuple
    if isinstance(sc, (list, tuple)):
        if len(sc) == 0:
            return None
        return _extract_reward_scalar(type("Tmp", (), {"scores": sc[0]})())  # minimal adapter

    # Fallback: numeric
    try:
        return float(sc)
    except Exception:
        return None


def _group_avg_reward(group: List[Any]) -> Optional[float]:
    """
    Compute average reward for filtering.

    Semantics:
      - If ANY sample in the group has is_leaf information, average over leaf-only samples.
      - Otherwise, average over all samples in the group.
    """
    if not group:
        return None

    leaf_flags = [_get_is_leaf(s) for s in group]
    has_leaf_flag = any(f is not None for f in leaf_flags)

    vals: List[float] = []

    # Primary path: leaf-only when leaf flags exist
    if has_leaf_flag:
        for s, flg in zip(group, leaf_flags):
            if flg is not True:
                continue
            v = _extract_reward_scalar(s)
            if v is not None:
                vals.append(float(v))

    # Fallback: use all samples if leaf-only produced nothing (or leaf flags absent)
    if not vals:
        for s in group:
            v = _extract_reward_scalar(s)
            if v is not None:
                vals.append(float(v))

    if not vals:
        return None
    return float(sum(vals)) / float(len(vals))


def dyn_filter_update(
    rollout_samples: List[Any],
    *,
    k: int,
    rollout_batch_size: int,
    reward_range: Tuple[float, float],
    state: DynFilterState,
):
    """
    Prompt-level dynamic filtering update.

    Filtering is performed per-prompt:
      - group rollout samples by prompt (question_id when available)
      - compute avg reward per prompt:
          * leaf-only if is_leaf exists
          * otherwise avg over all nodes for that prompt
      - if avg reward within (min,max), accept the prompt and keep ALL its nodes
      - once we have rollout_batch_size accepted prompts, return flattened samples

    Args:
      rollout_samples: newly generated samples (any mix of A*K nodes or tree nodes).
      k: legacy K used only for fallback chunking when question_id is unavailable.
      rollout_batch_size: number of accepted prompts to accumulate before yielding.
      reward_range: (min_reward, max_reward) filtering range, exclusive with epsilon margins.
      state: (buf_groups, n_total_prompts)

    Returns:
      (selected_or_none, pass_rate_or_none, new_state)
        - selected_or_none: List[sample] once enough prompts accepted, else None
        - pass_rate_or_none: float (%) once selected is returned, else None
        - new_state: updated state tuple
    """
    buf_groups, n_total_prompts = state

    prompt_groups = group_rollout_samples_by_prompt(list(rollout_samples or []), int(k or 1))
    n_total_prompts += int(len(prompt_groups))

    min_reward, max_reward = float(reward_range[0]), float(reward_range[1])
    rb = int(rollout_batch_size)

    for g in prompt_groups:
        avg = _group_avg_reward(g)
        if avg is None:
            continue
        if (min_reward + _EPS) < float(avg) < (max_reward - _EPS):
            # Keep ALL nodes for this prompt (A*K or tree nodes)
            buf_groups.append(g)

    if len(buf_groups) < rb:
        return None, None, (buf_groups, n_total_prompts)

    pass_rate = float(len(buf_groups)) / float(max(n_total_prompts, 1)) * 100.0
    selected_groups = buf_groups[:rb]
    selected = [s for g in selected_groups for s in g]
    return selected, pass_rate, ([], 0)
