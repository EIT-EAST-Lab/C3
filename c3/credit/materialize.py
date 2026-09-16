# -*- coding: utf-8 -*-
"""
C3 materialization helpers (Rule-B nested prefix-tree rollouts).

Public APIs:
- materialize_c3_batch_data: leaf-only, full-K (qid x k) aggregation for Q-critic training.
- materialize_c3_tree_groups: all-nodes grouping by adv_group_id for Rule-B LOO credit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Row refs
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RowRef:
    exp_idx: int
    row_idx: int


# ---------------------------------------------------------------------
# Primitive converters
# ---------------------------------------------------------------------


def _as_int(x: Any) -> int:
    if isinstance(x, int):
        return int(x)
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return int(x.view(-1)[0].item())
    try:
        return int(x)
    except Exception as e:
        raise TypeError(f"cannot cast to int: {type(x)} -> {x!r}") from e


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, Mapping):
        for k in ("text", "generated_text", "output_text", "content"):
            v = x.get(k, None)
            if isinstance(v, str):
                return v
    return str(x)


# ---------------------------------------------------------------------
# Experience.info helpers
# ---------------------------------------------------------------------


def _info_dict(exp: Any) -> Dict[str, Any]:
    info = getattr(exp, "info", None) or {}
    return info if isinstance(info, dict) else {}


def _info_at_row(v: Any, row: int) -> Any:
    """Pick row element from scalar/tensor/list-like info values."""
    if v is None:
        return None

    if isinstance(v, torch.Tensor):
        vv = v.view(-1)
        n = int(vv.numel())
        if n == 0:
            return None
        if n == 1:
            return vv[0]
        return vv[row] if 0 <= row < n else vv[0]

    if isinstance(v, list):
        n = len(v)
        if n == 0:
            return None
        return v[row] if 0 <= row < n else v[0]

    return v


def _info_int(exp: Any, row: int, key: str, *, default: Any = None, required: bool = False) -> int:
    info = _info_dict(exp)
    vv = _info_at_row(info.get(key, None), row)
    if vv is None:
        if required:
            raise KeyError(f"Experience.info missing {key!r}")
        return _as_int(default)
    return _as_int(vv)


def _info_str(exp: Any, row: int, key: str, *, default: str = "") -> str:
    info = _info_dict(exp)
    vv = _info_at_row(info.get(key, None), row)
    return str(vv) if vv is not None else str(default)


# ---------------------------------------------------------------------
# Experience row access
# ---------------------------------------------------------------------


def _get_batch_size(exp: Any) -> int:
    # Prefer common batched tensor/list attributes.
    for attr in ("sequences", "prompt_ids", "input_ids", "prompts"):
        v = getattr(exp, attr, None)
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            if v.dim() >= 1:
                return int(v.shape[0])
            if v.numel() == 1:
                return 1
        if isinstance(v, (list, tuple)):
            return len(v)

    # Fallback to info fields that are typically per-row.
    info = _info_dict(exp)
    for key in ("question_id", "k_id", "role"):
        v = info.get(key, None)
        if isinstance(v, torch.Tensor):
            return int(v.numel())
        if isinstance(v, list):
            return len(v)

    raise ValueError("unable to infer experience batch size")


def _get_prompt_at_row(exp: Any, row: int) -> str:
    info = _info_dict(exp)

    q = info.get("question", None)
    if isinstance(q, list) and 0 <= row < len(q):
        s = str(q[row])
        if s.strip():
            return s
    if isinstance(q, str) and q.strip():
        return q

    p = getattr(exp, "prompts", None)
    if isinstance(p, list) and 0 <= row < len(p):
        return str(p[row])

    return ""


def _traj_map_at_row(exp: Any, row: int, key: str) -> Dict[str, str]:
    """
    key: 'traj_role_outputs' or 'traj_role_prompts'
    Expected:
      - dict[role] -> list[str] (len=B)  OR
      - dict[role] -> str
    """
    m = _info_dict(exp).get(key, None)
    if not isinstance(m, Mapping):
        if key == "traj_role_outputs":
            raise KeyError("Experience.info missing 'traj_role_outputs' mapping")
        return {}

    out: Dict[str, str] = {}
    for r, v in m.items():
        rr = str(r)

        if isinstance(v, list):
            if not v:
                out[rr] = ""
            else:
                out[rr] = _to_text(v[row] if 0 <= row < len(v) else v[0])
            continue

        if isinstance(v, torch.Tensor):
            vv = v.view(-1)
            n = int(vv.numel())
            if n == 0:
                out[rr] = ""
            elif n == 1:
                out[rr] = _to_text(vv[0].item())
            else:
                out[rr] = _to_text(vv[row].item() if 0 <= row < n else vv[0].item())
            continue

        out[rr] = _to_text(v)

    return out


def _get_traj_role_outputs_at_row(exp: Any, row: int) -> Dict[str, str]:
    return _traj_map_at_row(exp, row, "traj_role_outputs")


def _get_traj_role_prompts_at_row(exp: Any, row: int) -> Dict[str, str]:
    return _traj_map_at_row(exp, row, "traj_role_prompts")


def _get_action_mask_1d(exp: Any, row: int, expected_len: int) -> Optional[torch.Tensor]:
    am = getattr(exp, "action_mask", None)
    if not isinstance(am, torch.Tensor):
        return None

    m: Optional[torch.Tensor] = None
    if am.dim() == 2 and 0 <= row < int(am.shape[0]):
        m = am[row].detach().view(-1)
    elif am.dim() == 1:
        m = am.detach().view(-1)

    if m is None or int(m.numel()) != int(expected_len):
        return None
    return m.to(dtype=torch.bool, device=m.device)


def _sum_token_rewards(tok: torch.Tensor, *, exp: Any | None = None, row: int | None = None) -> float:
    t = tok.detach().view(-1)
    if int(t.numel()) == 0:
        return 0.0

    if exp is not None and row is not None:
        m = _get_action_mask_1d(exp, int(row), int(t.numel()))
        if m is not None:
            t = t[m]
            if int(t.numel()) == 0:
                return 0.0

    return float(t.sum().item())


def _reward_to_scalar(x: Any, *, exp: Any | None = None, row: int | None = None) -> float:
    """
    Convert reward-like values to a scalar.
    - Token rewards: sum (optionally masked by action_mask if compatible).
    - Per-row rewards: index by row when shape/length matches batch size.
    """
    if x is None:
        raise TypeError("reward is None")

    if isinstance(x, (float, int, bool)):
        return float(x)

    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.view(-1)[0].item())

        if exp is not None and row is not None:
            B = _get_batch_size(exp)
            if x.dim() >= 1 and int(x.shape[0]) == int(B):
                if x.dim() == 1:
                    return float(x[int(row)].item())
                return _sum_token_rewards(x[int(row)], exp=exp, row=int(row))

        return _sum_token_rewards(x, exp=None, row=None)

    if isinstance(x, (list, tuple)):
        if exp is not None and row is not None:
            B = _get_batch_size(exp)
            if len(x) == B:
                return _reward_to_scalar(x[int(row)], exp=exp, row=int(row))

            m = _get_action_mask_1d(exp, int(row), len(x))
            if m is not None:
                keep = m.cpu().tolist()
                s = 0.0
                for v, k in zip(x, keep):
                    if k:
                        s += _reward_to_scalar(v, exp=None, row=None)
                return float(s)

        s = 0.0
        for v in x:
            s += _reward_to_scalar(v, exp=None, row=None)
        return float(s)

    return float(x)


def _get_reward_at_row(exp: Any, row: int) -> float:
    for attr in ("reward", "rewards"):
        v = getattr(exp, attr, None)
        if v is None:
            continue
        try:
            return _reward_to_scalar(v, exp=exp, row=row)
        except Exception:
            pass

    info = _info_dict(exp)
    for key in ("reward", "reward_score", "score", "rm_score", "rewards"):
        v = info.get(key, None)
        if v is None:
            continue
        try:
            return _reward_to_scalar(v, exp=exp, row=row)
        except Exception:
            pass

    raise KeyError(f"cannot find usable reward in experience (row={row})")


def _get_qid_kid_role_at_row(exp: Any, row: int) -> Tuple[int, int, str]:
    qid = _info_int(exp, row, "question_id", required=True)
    kid = _info_int(exp, row, "k_id", required=True)
    role = _info_str(exp, row, "role", default="")
    return int(qid), int(kid), str(role)


def _is_leaf_row(exp: Any, row: int, kid: int) -> bool:
    if kid < 0:
        return False

    info = _info_dict(exp)
    v = info.get("is_leaf", None)
    if v is None:
        return True

    try:
        return bool(_info_int(exp, row, "is_leaf", default=1))
    except Exception:
        return True


# ---------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------


def infer_roles_from_experiences(experiences: Sequence[Any]) -> List[str]:
    roles: set[str] = set()
    for exp in experiences:
        tro = _info_dict(exp).get("traj_role_outputs", None)
        if isinstance(tro, Mapping):
            for r in tro.keys():
                roles.add(str(r))
    return sorted(roles)


# ---------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------


def materialize_c3_batch_data(
    experiences: Sequence[Any],
    *,
    roles: Optional[Sequence[str]] = None,
    k_rollouts: int,
    require_full_k: bool = True,
) -> Tuple[List[dict], Dict[int, int], Dict[Tuple[int, int, str], List[RowRef]]]:
    """
    Leaf-only C3 batch_data for Q-critic training.

    Returns:
      - batch_data: per-question (qid) aggregation with full-K candidates.
      - qid_to_b: stable (qid -> batch index) mapping.
      - row_refs: (qid,kid,role) -> [RowRef] for leaf rows (all refs retained).
    """
    K = int(k_rollouts)
    if K <= 0:
        raise ValueError("k_rollouts must be >= 1")

    roles_list = list(roles) if roles is not None else infer_roles_from_experiences(experiences)
    if not roles_list:
        raise ValueError("roles is empty; pass roles=... or ensure traj_role_outputs exists")

    if require_full_k is False:
        logger.warning("materialize_c3_batch_data: require_full_k=False is ignored; full-K is always enforced.")

    # (qid,kid) -> (prompt, {role: text}, reward)
    traj: Dict[Tuple[int, int], Tuple[str, Dict[str, str], float]] = {}
    # (qid,kid,role) -> leaf row refs
    row_refs: Dict[Tuple[int, int, str], List[RowRef]] = {}

    for ei, exp in enumerate(experiences):
        B = _get_batch_size(exp)
        for j in range(B):
            qid, kid, role = _get_qid_kid_role_at_row(exp, j)

            if not _is_leaf_row(exp, j, kid):
                continue

            row_refs.setdefault((qid, kid, role), []).append(RowRef(exp_idx=ei, row_idx=j))

            kk = (qid, kid)
            if kk in traj:
                continue  # keep first record for (qid,kid), but retain all row_refs

            traj[kk] = (
                _get_prompt_at_row(exp, j),
                _get_traj_role_outputs_at_row(exp, j),
                float(_get_reward_at_row(exp, j)),
            )

    # qid -> kid -> record
    by_q: Dict[int, Dict[int, Tuple[str, Dict[str, str], float]]] = {}
    for (qid, kid), rec in traj.items():
        by_q.setdefault(int(qid), {})[int(kid)] = rec

    expected = set(range(K))
    bad: Dict[int, Dict[str, List[int]]] = {}
    for qid, kids in by_q.items():
        present = set(kids.keys())
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        if missing or extra:
            bad[int(qid)] = {"missing": missing, "extra": extra}

    if bad:
        examples = list(bad.items())[:5]
        example_str = "; ".join([f"qid={qid} missing={v['missing']} extra={v['extra']}" for qid, v in examples])

        msg2 = ""
        try:
            leaf_counts = [len(v) for v in by_q.values()]
            if leaf_counts:
                msg2 = (
                    f" leaf_per_q min={min(leaf_counts)} max={max(leaf_counts)} "
                    f"mean={sum(leaf_counts)/len(leaf_counts):.2f}"
                )
        except Exception:
            msg2 = ""

        raise RuntimeError(
            f"Full-K violation in C3 batch materialization: expected k_id in [0..{K-1}] for every question_id. "
            f"Bad questions: {len(bad)}/{len(by_q)}. Examples: {example_str}.{msg2}"
        )

    qids_sorted = sorted(by_q.keys())
    qid_to_b = {qid: i for i, qid in enumerate(qids_sorted)}

    roles_list = list(roles_list)
    batch_data: List[dict] = []

    for qid in qids_sorted:
        kids = by_q[qid]

        obs_prompt, _, _ = kids[0]  # full-K enforced -> k=0 exists
        candidates: Dict[str, List[str]] = {r: [] for r in roles_list}
        group_rewards: List[float] = []
        joint_actions_by_k: List[Dict[str, str]] = []

        for k in range(K):
            _p_k, tro_k, rew_k = kids[k]
            group_rewards.append(float(rew_k))

            ja_k = {r: _to_text(tro_k.get(r, "")) for r in roles_list}
            joint_actions_by_k.append(ja_k)
            for r in roles_list:
                candidates[r].append(ja_k.get(r, ""))

        batch_data.append(
            {
                "observation": str(obs_prompt),
                "candidates": candidates,
                "group_rewards": [float(x) for x in group_rewards],
                "joint_actions_by_k": joint_actions_by_k,
                "info": {"question_id": int(qid)},
            }
        )

    return batch_data, qid_to_b, row_refs


def materialize_c3_tree_groups(
    experiences: Sequence[Any],
    *,
    roles: Optional[Sequence[str]] = None,
) -> Tuple[List[dict], Dict[int, int], Dict[str, float]]:
    """
    Group all C3 nodes by adv_group_id (sibling groups for Rule-B LOO credit).
    """
    roles_list = list(roles) if roles is not None else infer_roles_from_experiences(experiences)

    groups_by_gid: Dict[int, dict] = {}
    seen_rows: set[tuple[int, int]] = set()

    num_rows = 0
    num_leaf = 0
    num_internal = 0

    for ei, exp in enumerate(experiences):
        B = _get_batch_size(exp)
        for j in range(B):
            num_rows += 1
            qid, kid, role = _get_qid_kid_role_at_row(exp, j)

            try:
                gid = _info_int(exp, j, "adv_group_id", required=True)
            except Exception as e:
                raise RuntimeError(f"[C3][FAIL-FAST] Missing adv_group_id at (exp={ei}, row={j}): {e}") from e

            role_id = _info_int(exp, j, "role_id", default=0)
            parent_id = _info_int(exp, j, "c3_parent_id", default=-1)
            node_id = _info_int(exp, j, "c3_node_id", default=-1)
            depth = _info_int(exp, j, "c3_depth", default=-1)
            is_leaf = _info_int(exp, j, "is_leaf", default=(1 if int(kid) >= 0 else 0))
            leaf_start = _info_int(exp, j, "c3_leaf_start", default=-1)
            leaf_size = _info_int(exp, j, "c3_leaf_size", default=-1)

            if int(is_leaf) != 0:
                num_leaf += 1
            else:
                num_internal += 1

            rew = _get_reward_at_row(exp, j)
            obs = _get_prompt_at_row(exp, j)
            tro = _get_traj_role_outputs_at_row(exp, j)
            trp = _get_traj_role_prompts_at_row(exp, j)

            g = groups_by_gid.get(int(gid))
            if g is None:
                g = {
                    "adv_group_id": int(gid),
                    "question_id": int(qid),
                    "role": str(role),
                    "role_id": int(role_id),
                    "parent_id": int(parent_id),
                    "depth": int(depth),
                    "observation": str(obs),
                    "roles": list(roles_list),
                    "node_refs": [],
                    "node_rewards": [],
                    "node_k_ids": [],
                    "node_ids": [],
                    "node_is_leaf": [],
                    "node_leaf_start": [],
                    "node_leaf_size": [],
                    "traj_role_outputs": [],
                    "traj_role_prompts": [],
                }
                groups_by_gid[int(gid)] = g
            else:
                if (
                    int(g.get("question_id", -1)) != int(qid)
                    or str(g.get("role")) != str(role)
                    or int(g.get("parent_id", -999)) != int(parent_id)
                ):
                    raise RuntimeError(
                        f"[C3][FAIL-FAST] adv_group_id collision/inconsistent metadata: gid={gid} "
                        f"existing(qid={g.get('question_id')}, role={g.get('role')}, parent={g.get('parent_id')}) "
                        f"new(qid={qid}, role={role}, parent={parent_id})"
                    )
                if int(g.get("role_id", -1)) != int(role_id):
                    raise RuntimeError(
                        f"[C3][FAIL-FAST] Inconsistent role_id within adv_group_id: gid={gid} "
                        f"existing(role_id={g.get('role_id')}) new(role_id={role_id})"
                    )
                if int(g.get("depth", -1)) != int(depth):
                    raise RuntimeError(
                        f"[C3][FAIL-FAST] Inconsistent c3_depth within adv_group_id: gid={gid} "
                        f"existing(depth={g.get('depth')}) new(depth={depth})"
                    )
                ex_obs = str(g.get("observation", "") or "")
                new_obs = str(obs or "")
                if ex_obs and new_obs and ex_obs != new_obs:
                    raise RuntimeError(
                        f"[C3][FAIL-FAST] Inconsistent observation within adv_group_id: gid={gid} "
                        f"existing(obs_len={len(ex_obs)}) new(obs_len={len(new_obs)})"
                    )

            rr_key = (int(ei), int(j))
            if rr_key in seen_rows:
                raise RuntimeError(
                    f"[C3][FAIL-FAST] Duplicate Experience row mapped into C3 groups: exp={ei}, row={j}."
                )
            seen_rows.add(rr_key)

            g["node_refs"].append(RowRef(exp_idx=ei, row_idx=j))
            g["node_rewards"].append(float(rew))
            g["node_k_ids"].append(int(kid))
            g["node_ids"].append(int(node_id))
            g["node_is_leaf"].append(int(is_leaf))
            g["node_leaf_start"].append(int(leaf_start))
            g["node_leaf_size"].append(int(leaf_size))
            g["traj_role_outputs"].append({str(r): _to_text(tro.get(r, "")) for r in roles_list})
            g["traj_role_prompts"].append({str(r): _to_text(trp.get(r, "")) for r in roles_list})

    groups = list(groups_by_gid.values())
    groups.sort(
        key=lambda g: (
            int(g.get("question_id", 0)),
            int(g.get("depth", 0)),
            int(g.get("parent_id", -1)),
            int(g.get("role_id", 0)),
            int(g.get("adv_group_id", 0)),
        )
    )

    gid_to_g = {int(g["adv_group_id"]): i for i, g in enumerate(groups)}
    sizes = [len(g.get("node_refs", [])) for g in groups]

    diag: Dict[str, float] = {
        "c3/tree_groups": float(len(groups)),
        "c3/tree_rows": float(num_rows),
        "c3/tree_leaf_rows": float(num_leaf),
        "c3/tree_internal_rows": float(num_internal),
        "c3/tree_group_size_min": float(min(sizes) if sizes else 0.0),
        "c3/tree_group_size_max": float(max(sizes) if sizes else 0.0),
        "c3/tree_group_size_mean": float(sum(sizes) / max(1, len(sizes))) if sizes else 0.0,
    }

    return groups, gid_to_g, diag
