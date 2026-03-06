from __future__ import annotations

from typing import List, Optional

from c3.rewards.base import RewardRequest, RewardResult


def _normalize_env_name(env_name: str) -> str:
    s = (env_name or "").strip().lower()
    if s in {"math", "mathenv", "c3_math"}:
        return "MathEnv"
    if s in {"code", "codeenv", "c3_code"}:
        return "CodeEnv"
    # Already normalized or unknown
    return env_name.strip() if isinstance(env_name, str) else str(env_name)


def _allow_empty_label(env_name: str) -> bool:
    """Whether env reward can be computed without label_text."""
    return _normalize_env_name(env_name) == "CodeEnv"

def _extract_label_from_meta(meta: dict) -> Optional[str]:
    """Try to recover gold label from req.meta for env reward."""
    if not isinstance(meta, dict):
        return None

    cand_keys = (
        "answer",
        "golden",
        "label",
        "solution",
        "target",
        "ground_truth",
        "gt",
        "final",
        "final_answer",
        "expected",
        "expected_output",
    )

    def _pick(d: dict) -> Optional[str]:
        for k in cand_keys:
            v = d.get(k, None)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return None

    dm = meta.get("dataset_meta", None)
    if isinstance(dm, dict):
        got = _pick(dm)
        if got:
            return got

    mj = meta.get("dataset_meta_json", None)
    if isinstance(mj, str) and mj.strip():
        try:
            import json as _json
            dm2 = _json.loads(mj)
            if isinstance(dm2, dict):
                got = _pick(dm2)
                if got:
                    return got
        except Exception:
            pass

    # last resort: top-level meta
    got = _pick(meta)
    return got


class RewardProvider:
    def compute(self, batch: List[RewardRequest]) -> List[Optional[RewardResult]]:
        raise NotImplementedError


class NoneRewardProvider(RewardProvider):
    def compute(self, batch: List[RewardRequest]) -> List[Optional[RewardResult]]:
        return [None for _ in batch]


class EnvRewardProvider(RewardProvider):
    def __init__(self, *, env_name: str):
        self.env_name = env_name
        try:
            from c3.envs import get_env_reward_fn

            self._fn = get_env_reward_fn(env_name)
        except Exception:
            self._fn = None

    def compute(self, batch: List[RewardRequest]) -> List[Optional[RewardResult]]:
        out: List[Optional[RewardResult]] = []
        for req in batch:
            if self._fn is None:
                out.append(None)
                continue

            # Prefer per-request env_name if present (meta injected by MAS rollout generator)
            req_env_name = str(req.meta.get("env_name", self.env_name) or self.env_name)

            label = req.label_text
            label_empty = (label is None) or (str(label).strip() == "")

            # For CodeEnv, allow label to be empty (judging comes from dataset_meta/tests).
            # For MathEnv, allow empty label ONLY if we can recover gold from dataset meta.
            if label_empty and not _allow_empty_label(req_env_name):
                recovered = _extract_label_from_meta(req.meta)
                if recovered is None or recovered.strip() == "":
                    out.append(None)
                    continue
                label = recovered
                label_empty = False


            # For CodeEnv, allow label to be empty (judging comes from dataset_meta/tests).
            if label_empty and not _allow_empty_label(req_env_name):
                out.append(None)
                continue

            pred = str(req.meta.get("answer_text", ""))

            # Keep label optional to support CodeEnv.
            label_arg = None if label_empty else str(label)

            try:
                r, info = self._fn(
                    prediction=pred,
                    label=label_arg,
                    meta=req.meta,
                )
                out.append(RewardResult(reward=float(r), score=float(r), source="env", info=info))
            except Exception as e:
                out.append(
                    RewardResult(
                        reward=0.0,
                        score=0.0,
                        source="env",
                        info={"error": f"{type(e).__name__}: {e}", "env_name": req_env_name},
                    )
                )
        return out


class RemoteRMRewardProvider(RewardProvider):
    def __init__(self, *, remote_reward_model):
        self.remote_reward_model = remote_reward_model

    def compute(self, batch: List[RewardRequest]) -> List[Optional[RewardResult]]:
        if self.remote_reward_model is None:
            return [None for _ in batch]

        import ray

        def _to_list(x):
            # Accept python list / torch tensor / numpy
            if x is None:
                return []
            if isinstance(x, list):
                return x
            if hasattr(x, "tolist"):
                return x.tolist()
            try:
                return list(x)
            except Exception:
                return [x]

        N = len(batch)
        queries = [r.query_text for r in batch]
        prompts = [r.prompt_text for r in batch]
        labels = [r.label_text for r in batch]
        metas = [r.meta for r in batch]

        # RemoteRewardModel returns a LIST of chunk results (dicts) for HTTP/custom reward_func.
        rewards_info = ray.get(self.remote_reward_model.get_rewards.remote(queries, prompts, labels, metas))

        # Normalize: allow single dict response
        if isinstance(rewards_info, dict):
            rewards_info = [rewards_info]

        # Case A: chunked/batched format like {"rewards":[...], "scores":[...], "extra_logs":{...}}
        if rewards_info and isinstance(rewards_info[0], dict) and ("rewards" in rewards_info[0]):
            rewards_flat: List[Any] = []
            scores_flat: List[Any] = []
            extra_logs_flat: dict = {}

            for chunk in rewards_info:
                r_list = _to_list(chunk.get("rewards", []))
                s_list = _to_list(chunk.get("scores", r_list))
                rewards_flat.extend(r_list)
                scores_flat.extend(s_list)

                ex = chunk.get("extra_logs", None)
                if isinstance(ex, dict):
                    for k, v in ex.items():
                        extra_logs_flat.setdefault(k, []).extend(_to_list(v))

            # Trim to N (safety)
            rewards_flat = rewards_flat[:N]
            scores_flat = scores_flat[:N]
            for k in list(extra_logs_flat.keys()):
                extra_logs_flat[k] = extra_logs_flat[k][:N]

            if len(rewards_flat) != N:
                raise RuntimeError(
                    f"Remote RM returned batched rewards with unexpected length: got={len(rewards_flat)} expected={N}. "
                    f"Check remote_rm server / custom reward_func output."
                )

            out: List[Optional[RewardResult]] = []
            for i in range(N):
                r = float(rewards_flat[i])
                s = float(scores_flat[i]) if i < len(scores_flat) else r
                info = {}
                # Keep per-sample extra logs if present
                for k, v in extra_logs_flat.items():
                    if i < len(v):
                        info[k] = v[i]
                out.append(RewardResult(reward=r, score=s, source="remote_rm", info=info if info else None))
            return out

        # Case B: per-sample list (each item is dict or number), length must match N
        if not isinstance(rewards_info, list) or len(rewards_info) != N:
            raise RuntimeError(
                f"Remote RM returned unexpected structure for MAS provider: "
                f"type={type(rewards_info)}, len={len(rewards_info) if isinstance(rewards_info, list) else 'NA'}, expected={N}."
            )

        out: List[Optional[RewardResult]] = []
        for ri in rewards_info:
            if isinstance(ri, dict):
                r = float(ri.get("reward", ri.get("score", 0.0)))
                s = float(ri.get("score", r))
                out.append(RewardResult(reward=r, score=s, source="remote_rm", info={k: v for k, v in ri.items()}))
            else:
                r = float(ri)
                out.append(RewardResult(reward=r, score=r, source="remote_rm", info=None))
        return out



class ChainedRewardProvider(RewardProvider):
    def __init__(self, *, env_name: str, remote_reward_model):
        self.env = EnvRewardProvider(env_name=env_name)
        self.rm = RemoteRMRewardProvider(remote_reward_model=remote_reward_model)

    def compute(self, batch: List[RewardRequest]) -> List[Optional[RewardResult]]:
        first = self.env.compute(batch)
        need = [i for i, r in enumerate(first) if (r is None)]
        if not need:
            return first

        second = self.rm.compute([batch[i] for i in need])
        for idx, rr in zip(need, second):
            first[idx] = rr
        return first


class AutoRewardProvider(RewardProvider):
    def __init__(self, *, env_name: str, remote_reward_model):
        # auto = chain, but if rm is None then env-only
        self.provider = ChainedRewardProvider(env_name=env_name, remote_reward_model=remote_reward_model)

    def compute(self, batch: List[RewardRequest]) -> List[Optional[RewardResult]]:
        return self.provider.compute(batch)
