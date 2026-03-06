# -*- coding: utf-8 -*-
"""
c3.credit.c3.provider

Rule-B C3 credit provider.

Computes per-node scalar advantages on a prefix-tree rollout using within-group baselines
(grouped by adv_group_id).

Variants:
  - reward_only     : signal = subtree_mean_reward
  - value_only      : signal = Q(prefix)
  - value_assisted  : signal = subtree_mean_reward, baseline mixes reward/Q baselines

Baseline modes:
  - loo       : leave-one-out mean
  - full_mean : mean including self

No-replay ablation:
  - Rollout side collapses parent_id for non-root depths (adv_group_id ignores transcript parent).
  - Thus a "sibling group" at depth>0 can become size=product(fanout[:depth+1]) rather than fanout[depth].
  - Provider accepts that size when args.c3_no_replay=True.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from c3.integration.marl_specs import RoleSpec

from .baselines import build_dependency_from_roles, format_for_q
from .scoring import score_texts_batched


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read from dict-like or attribute-like objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _infer_batch_size(exp: Any) -> int:
    """Best-effort batch size inference for Experience-like objects."""
    seq = getattr(exp, "sequences", None)
    if isinstance(seq, torch.Tensor) and seq.dim() >= 1:
        return int(seq.shape[0])

    info = getattr(exp, "info", None)
    if isinstance(info, dict):
        qid = info.get("question_id", None)
        if isinstance(qid, torch.Tensor):
            return int(qid.view(-1).numel())
        if isinstance(qid, list):
            return int(len(qid))

    prompts = getattr(exp, "prompts", None)
    if isinstance(prompts, list):
        return int(len(prompts))

    raise RuntimeError("[C3][FAIL-FAST] Failed to infer Experience batch size.")


def _ensure_1d(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"[C3][FAIL-FAST] {name} must be a torch.Tensor.")
    if x.dim() != 1:
        raise ValueError(f"[C3][FAIL-FAST] {name} must be 1D, got shape={tuple(x.shape)}.")
    return x


def _baseline_loo(v: torch.Tensor) -> torch.Tensor:
    """LOO baseline: baseline[i] = mean(v[j] for j!=i), requires n>=2."""
    v = _ensure_1d(v, name="values")
    n = int(v.numel())
    if n <= 1:
        raise RuntimeError("[C3][FAIL-FAST] LOO baseline requires group size >= 2.")
    s = v.sum()
    return (s - v) / float(n - 1)


def _baseline_full_mean(v: torch.Tensor) -> torch.Tensor:
    """Full-mean baseline: baseline[i] = mean(v)."""
    v = _ensure_1d(v, name="values")
    m = v.mean()
    return m.expand_as(v)


def _load_preamble_json(path: str) -> str:
    """Load a preamble string from a json file; return '' on any error."""
    path = str(path or "").strip()
    if not path:
        return ""
    try:
        p = Path(path)
        if not p.exists():
            return ""
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for k in ("preamble", "system", "prompt", "text"):
                v = data.get(k, None)
                if isinstance(v, str) and v.strip():
                    return v
    except Exception:
        pass
    return ""


def _prepend_preamble(preamble: str, text: str) -> str:
    pre = str(preamble or "").strip()
    if not pre:
        return str(text or "")
    t = str(text or "")
    return t if t.startswith(pre) else (pre + "\n\n" + t)

# ---------------------------------------------------------------------------
# Public API (backwards-compatible aliases)
# ---------------------------------------------------------------------------

def load_critic_preamble_from_json(path: str) -> str:
    """
    Backwards-compatible alias for older code paths (e.g. openrlhf PPO critic).
    Loads a preamble string from a json file; returns '' on any error.
    """
    return _load_preamble_json(path)


def prepend_preamble(preamble: str, text: str) -> str:
    """
    Backwards-compatible alias for older code paths.
    Prepends preamble to text if not already present.
    """
    return _prepend_preamble(preamble, text)


def _infer_device(model: Any) -> torch.device:
    """Infer device from a torch.nn.Module or wrapper; fallback cpu."""
    try:
        d = getattr(model, "device", None)
        if d is not None:
            return d if isinstance(d, torch.device) else torch.device(str(d))
    except Exception:
        pass
    try:
        if hasattr(model, "parameters"):
            for p in model.parameters():
                return p.device
    except Exception:
        pass
    return torch.device("cpu")


def _maybe_ray_get(x: Any) -> Any:
    """ray.get(ObjectRef) if needed; otherwise passthrough."""
    t = type(x)
    if getattr(t, "__name__", "") == "ObjectRef" and "ray" in getattr(t, "__module__", ""):
        import ray  # lazy

        return ray.get(x)
    return x


def _prod_int(xs: Sequence[int]) -> int:
    p = 1
    for a in xs:
        p *= int(a)
    return int(p)


def _check_group_size(
    *,
    n: int,
    depth: int,
    adv_group_id: Any,
    fanout_list: Optional[Sequence[int]],
    no_replay: bool,
) -> None:
    """Fail-fast sizing check against fanout; accepts no-replay alt sizing."""
    if not isinstance(fanout_list, (list, tuple)):
        return

    if depth < 0 or depth >= len(fanout_list):
        raise RuntimeError(
            "[C3][FAIL-FAST] Missing/invalid depth for fanout check. "
            f"depth={depth} fanout_list_len={len(fanout_list)} adv_group_id={adv_group_id}. "
            "Ensure rollout generator writes info['c3_depth'] for every node."
        )

    expected = int(fanout_list[depth])
    if n == expected:
        return

    if no_replay and depth > 0:
        alt_expected = _prod_int([int(x) for x in fanout_list[: depth + 1]])
        if n == alt_expected:
            return
        raise RuntimeError(
            "[C3][FAIL-FAST] Sibling group size mismatch (no-replay). "
            f"depth={depth} expected_fanout={expected} alt_expected={alt_expected} got_size={n} "
            f"adv_group_id={adv_group_id}. "
            "This indicates missing/extra nodes or incorrect adv_group_id encoding."
        )

    raise RuntimeError(
        "[C3][FAIL-FAST] Sibling group size mismatch. "
        f"depth={depth} expected_fanout={expected} got_size={n} adv_group_id={adv_group_id}. "
        "This indicates missing/extra nodes or incorrect adv_group_id encoding."
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class C3CreditProvider:
    """Rule-B C3 advantage provider (prefix-only Q views)."""

    _PREFIX_SCOPE: str = "topo_prefix"
    _VARIANTS = {"reward_only", "value_only", "value_assisted"}
    _BASELINE_MODES = {"loo", "full_mean"}

    def __init__(
        self,
        *,
        args: Any,
        roles: Sequence[RoleSpec],
        q_critic: Any,
        generate_for_roles=None,  # kept for ctor compatibility; unused in Rule-B
        critic_preamble_path: str = "",
    ):
        self.args = args
        self.roles_spec: Tuple[RoleSpec, ...] = tuple(roles)
        self.roles: List[str] = [r.name for r in self.roles_spec]
        self.q_critic = q_critic

        # Canonical dependency structure for formatting Q text.
        self.parents, self.layers, self.topo_order, _, _, _ = build_dependency_from_roles(self.roles_spec)

        # Throughput knobs (not view knobs).
        self.critic_ctx_limit = int(getattr(args, "critic_ctx_limit", 2048) or 2048)
        self.critic_forward_bs = int(getattr(args, "critic_forward_bs", 4096) or 4096)

        # Optional critic preamble.
        pre_path = (
            str(critic_preamble_path or "").strip()
            or str(getattr(args, "critic_preamble_path", "") or "").strip()
        )
        self.critic_preamble = _load_preamble_json(pre_path) if pre_path else ""

        # Device hint for non-RPC scoring.
        self._critic_device = _infer_device(self.q_critic) if self.q_critic is not None else torch.device("cpu")

        _ = generate_for_roles  # explicitly ignored

    @contextmanager
    def _eval_mode(self):
        m = self.q_critic
        if m is None or not hasattr(m, "training"):
            yield
            return
        was_training = bool(m.training)
        try:
            if was_training:
                m.eval()
            yield
        finally:
            if was_training:
                m.train()

    def _q_text(self, *, question: str, actions: Mapping[str, str], up_to_role: str) -> str:
        text = format_for_q(
            question=str(question or ""),
            actions={str(k): str(v) for k, v in dict(actions).items()},
            mode="prefix",
            up_to_role=str(up_to_role),
            layers=self.layers,
            parents=self.parents,
            prefix_scope=self._PREFIX_SCOPE,
            strict=True,
        )
        return _prepend_preamble(self.critic_preamble, text)

    def _score_texts(self, texts: List[str]) -> torch.Tensor:
        """Return CPU float32 scores [N] from either RPC or local critic."""
        if not texts:
            return torch.empty(0, dtype=torch.float32)

        fn = getattr(self.q_critic, "score_texts", None)
        if callable(fn):
            out = fn(texts=texts, max_len=int(self.critic_ctx_limit), forward_bs=int(self.critic_forward_bs))
            out = _maybe_ray_get(out)

            if isinstance(out, torch.Tensor):
                return out.detach().to(torch.float32).view(-1).cpu()

            if isinstance(out, list):
                if not out:
                    return torch.empty(0, dtype=torch.float32)
                if isinstance(out[0], torch.Tensor):
                    ts = [o.detach().to(torch.float32).view(-1).cpu() for o in out]
                    return torch.cat(ts, dim=0) if len(ts) > 1 else ts[0]
                flat: List[float] = []
                for row in out:
                    if isinstance(row, list):
                        flat.extend([float(x) for x in row])
                    else:
                        flat.append(float(row))
                return torch.tensor(flat, dtype=torch.float32)

            return torch.tensor([float(out)], dtype=torch.float32)

        with self._eval_mode():
            return score_texts_batched(
                self.q_critic,
                texts=texts,
                device=self._critic_device,
                max_len=int(self.critic_ctx_limit),
                forward_bs=int(self.critic_forward_bs),
            )

    @torch.no_grad()
    def compute(
        self,
        tree_groups: List[dict],
        *,
        experiences: Optional[Sequence[Any]] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        if not isinstance(tree_groups, list) or (
            tree_groups and (not isinstance(tree_groups[0], dict) or "node_refs" not in tree_groups[0])
        ):
            raise RuntimeError(
                "[C3][FAIL-FAST] compute expects Rule-B tree_groups from materialize_c3_tree_groups()."
            )
        if experiences is None:
            raise RuntimeError("[C3][FAIL-FAST] experiences is required for Rule-B routing.")

        cfg = cfg or {}
        no_replay = bool(getattr(self.args, "c3_no_replay", False))

        # Guardrail: legacy knobs not supported by this provider.
        removed_cfg_keys = {
            "cf_mode",
            "variance_threshold",
            "regen_kwargs",
            "k_rollouts",
            "return_all_k",
            "include_full",
            "expand_prefix",
            "prefix_scope",
            "max_texts_per_sample",
        }
        bad = removed_cfg_keys.intersection(cfg.keys())
        if bad:
            raise RuntimeError(f"[C3][FAIL-FAST] Unsupported/removed C3 config keys in Rule-B provider: {sorted(bad)}")

        # Throughput overrides only.
        self.critic_ctx_limit = int(cfg.get("critic_ctx_limit", self.critic_ctx_limit) or self.critic_ctx_limit)
        self.critic_forward_bs = int(cfg.get("critic_forward_bs", self.critic_forward_bs) or self.critic_forward_bs)

        variant = str(
            cfg.get("credit_variant", getattr(self.args, "c3_credit_variant", "value_assisted")) or "value_assisted"
        ).strip().lower()
        if variant not in self._VARIANTS:
            raise RuntimeError(f"[C3][FAIL-FAST] Unknown credit_variant={variant!r}")

        baseline_mode = str(
            cfg.get("baseline_mode", getattr(self.args, "c3_baseline_mode", "loo")) or "loo"
        ).strip().lower()
        if baseline_mode not in self._BASELINE_MODES:
            raise RuntimeError(
                f"[C3][FAIL-FAST] Invalid baseline_mode={baseline_mode!r}, must be one of {sorted(self._BASELINE_MODES)}"
            )
        baseline_fn = _baseline_loo if baseline_mode == "loo" else _baseline_full_mean

        try:
            alpha = float(cfg.get("va_alpha", getattr(self.args, "c3_va_alpha", 1.0)))
        except Exception:
            alpha = 1.0
        if not (0.0 <= alpha <= 1.0):
            raise RuntimeError(f"[C3][FAIL-FAST] va_alpha must be in [0,1], got {alpha}")

        need_q = variant in ("value_only", "value_assisted")
        if need_q and self.q_critic is None:
            raise RuntimeError(f"[C3][FAIL-FAST] credit_variant={variant} requires q_critic, but q_critic is None.")

        # Output buffers (one scalar per Experience row).
        per_exp_scalar: List[torch.Tensor] = [
            torch.full((_infer_batch_size(e),), float("nan"), dtype=torch.float32) for e in experiences
        ]

        # Fanout sizing check inputs (optional).
        fanout_list = getattr(self.args, "c3_fanout_list", None)
        if isinstance(fanout_list, (list, tuple)):
            fanout_list = [int(x) for x in fanout_list]

        # Pre-score Q in one batched pass.
        q_scores: Optional[torch.Tensor] = None
        q_slices: Optional[List[Tuple[int, int]]] = None
        if need_q:
            all_texts: List[str] = []
            q_slices = []

            for g in tree_groups:
                start = len(all_texts)

                node_refs = g.get("node_refs", None)
                if not isinstance(node_refs, list) or not node_refs:
                    q_slices.append((start, start))
                    continue

                up_to_role = str(g.get("role", "") or "")
                if not up_to_role:
                    raise RuntimeError(
                        "[C3][FAIL-FAST] tree_groups missing group['role'] for Q scoring "
                        f"(adv_group_id={g.get('adv_group_id')})."
                    )
                if up_to_role not in self.roles:
                    raise RuntimeError(
                        "[C3][FAIL-FAST] tree_groups role not in provider roles: "
                        f"role={up_to_role!r}, provider_roles={self.roles}."
                    )

                trajs = g.get("traj_role_outputs", None)
                if not isinstance(trajs, list) or not trajs:
                    raise RuntimeError(
                        "[C3][FAIL-FAST] tree_groups missing traj_role_outputs for Q scoring "
                        f"(adv_group_id={g.get('adv_group_id')})."
                    )
                if len(trajs) != len(node_refs):
                    raise RuntimeError(
                        "[C3][FAIL-FAST] traj_role_outputs must align with node_refs. "
                        f"len(traj_role_outputs)={len(trajs)} len(node_refs)={len(node_refs)} "
                        f"adv_group_id={g.get('adv_group_id')}."
                    )

                question = g.get("question", None) or g.get("observation", "")
                for actions in trajs:
                    all_texts.append(self._q_text(question=str(question), actions=actions, up_to_role=up_to_role))

                q_slices.append((start, len(all_texts)))

            q_scores = self._score_texts(all_texts)
            if int(q_scores.numel()) != int(len(all_texts)):
                raise RuntimeError(
                    "[C3][FAIL-FAST] q_critic returned unexpected size: "
                    f"got {q_scores.numel()} expected {len(all_texts)}"
                )

        # Streaming diagnostics.
        groups_seen = 0
        nodes_seen = 0
        gmin, gmax, gsum = None, None, 0

        adv_sum = 0.0
        adv_sumsq = 0.0
        adv_n = 0

        for gi, g in enumerate(tree_groups):
            node_refs = g.get("node_refs", None)
            if not isinstance(node_refs, list) or not node_refs:
                continue

            node_rewards = g.get("node_rewards", None)
            if not isinstance(node_rewards, list) or len(node_rewards) != len(node_refs):
                raise RuntimeError(
                    "[C3][FAIL-FAST] tree_groups require node_rewards aligned with node_refs "
                    f"(adv_group_id={g.get('adv_group_id')})."
                )

            n = int(len(node_refs))
            groups_seen += 1
            nodes_seen += n
            gsum += n
            gmin = n if gmin is None else min(gmin, n)
            gmax = n if gmax is None else max(gmax, n)

            depth = int(g.get("depth", -1))
            _check_group_size(
                n=n,
                depth=depth,
                adv_group_id=g.get("adv_group_id"),
                fanout_list=fanout_list,
                no_replay=no_replay,
            )

            # N==1 => advantage defined as 0.
            if n <= 1:
                adv = torch.zeros((n,), dtype=torch.float32)
            else:
                r = torch.tensor(node_rewards, dtype=torch.float32)

                if variant == "reward_only":
                    adv = r - baseline_fn(r)
                else:
                    assert q_scores is not None and q_slices is not None
                    s, e = q_slices[gi]
                    q = q_scores[s:e].view(-1)
                    if int(q.numel()) != n:
                        raise RuntimeError(
                            "[C3][FAIL-FAST] Q slice length mismatch with node_refs. "
                            f"gi={gi} slice=({s},{e}) q_len={q.numel()} n={n} adv_group_id={g.get('adv_group_id')}"
                        )

                    if variant == "value_only":
                        adv = q - baseline_fn(q)
                    else:
                        b = (1.0 - alpha) * baseline_fn(r) + alpha * baseline_fn(q)
                        adv = r - b

            # Route to per-experience rows.
            for idx, rr in enumerate(node_refs):
                exp_idx = int(_get(rr, "exp_idx"))
                row_idx = int(_get(rr, "row_idx"))
                per_exp_scalar[exp_idx][row_idx] = adv[idx]

            # Update streaming adv stats.
            adv_sum += float(adv.sum().item())
            adv_sumsq += float((adv * adv).sum().item())
            adv_n += int(adv.numel())

        # Fail-fast: every row must be filled.
        for ei, buf in enumerate(per_exp_scalar):
            if torch.isnan(buf).any():
                raise RuntimeError(
                    f"[C3][FAIL-FAST] Missing routed advantages for Experience index={ei}. "
                    "Bug in materialize_c3_tree_groups() grouping/routing or rollout annotations."
                )

        # Diagnostics (float-only).
        diag: Dict[str, float] = {
            "c3/rule_b_groups": float(groups_seen),
            "c3/rule_b_nodes": float(nodes_seen),
            "c3/no_replay": 1.0 if no_replay else 0.0,
            "c3/q_used": 1.0 if need_q else 0.0,
            "c3/va_alpha": float(alpha),
            "c3/variant_reward_only": 1.0 if variant == "reward_only" else 0.0,
            "c3/variant_value_assisted": 1.0 if variant == "value_assisted" else 0.0,
            "c3/variant_value_only": 1.0 if variant == "value_only" else 0.0,
            "c3/baseline_mode_is_loo": 1.0 if baseline_mode == "loo" else 0.0,
            "c3/baseline_mode_is_full_mean": 1.0 if baseline_mode == "full_mean" else 0.0,
        }

        if groups_seen > 0:
            diag["c3/rule_b_group_size_min"] = float(gmin if gmin is not None else 0)
            diag["c3/rule_b_group_size_max"] = float(gmax if gmax is not None else 0)
            diag["c3/rule_b_group_size_mean"] = float(gsum / float(groups_seen))

        if adv_n > 0:
            mean = adv_sum / float(adv_n)
            var = max(0.0, (adv_sumsq / float(adv_n)) - mean * mean)
            diag["c3/adv_mean"] = float(mean)
            diag["c3/adv_std"] = float(var**0.5)
        else:
            diag["c3/adv_mean"] = 0.0
            diag["c3/adv_std"] = 0.0

        return per_exp_scalar, diag


