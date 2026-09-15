"""c3.analysis.replay_batched

Cross bucket batched counterfactual replay.

`ReplayRunner.run_bucket` walks one bucket at a time, and for a workflow with
more than two roles it asks the engine for one sequence per downstream role per
replay. On a deep workflow that is hundreds of single sequence calls per
decision point, which no inference engine can batch. This module keeps the
semantics of `run_bucket` and only changes when the calls are issued: every
bucket that needs a generation at the same stage is handed to the policy in one
`sample_many` call.

Stages, in order:

  A. candidates: one prompt per bucket, n = the number of alternatives the
     config asks for, with the same de-duplication and the same retry bound as
     `run_bucket`. Forced actions, when the caller passes any, go in first and
     are not sampled, exactly as in `run_bucket`. The stage is skipped whole
     when the caller passes `reuse`: the alternatives then come from an earlier
     bucket file and only stages B and C run, which is the second continuation
     seed of E1b.
  B. replays: for replay index r and for each role after the target role in
     topological order, one prompt per (bucket, candidate), n = 1.
  C. returns: one evaluator call per (bucket, candidate, replay).

Seeds are derived from `batched_seed`, which is a pure function of the run seed,
the bucket, the candidate index, the replay index and the role name. Two runs of
this module over the same inputs therefore issue the same seeds in the same
places, which is what the replay fidelity checks need. The scheme differs from
the sequential one (that one derives a seed per bucket from the context hash),
so every bucket built here records `meta.seed_scheme`.

Nothing is written here: the caller receives the finished buckets, so a failure
anywhere in the cell leaves no partial file behind.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from c3.analysis.replay import (
    Bucket,
    CandidateReuse,
    CandidateReuseError,
    CandidateResult,
    ReplayConfig,
    ReplayRunner,
    RestartState,
    _unique_extend,
    reuse_meta,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

#: Written to `meta.seed_scheme` of every bucket this module builds.
SEED_SCHEME = "batched_v1"

#: Prompts per engine call. One stage of a large cell can hold 500 * 8 * 4
#: prompts, so the stage is cut into chunks of this size, run back to back.
DEFAULT_BATCH_PROMPTS = 4096

#: Seeds are reduced into [0, 2**31) so that every engine accepts them.
SEED_MODULUS = 2 ** 31

#: Hash domain tag. Kept separate from SEED_SCHEME so that renaming the label
#: written into meta cannot silently move the seeds.
_SEED_DOMAIN = "c3/replay_batched/batched_v1"

#: Stage A has no candidate yet, so it uses this in the candidate slot of the
#: seed and the resampling round in the replay slot.
_CANDIDATE_STAGE = -1


class BatchedReplayError(RuntimeError):
    """A generation or evaluation call failed inside a batched cell."""


# -----------------------------------------------------------------------------
# Seeds
# -----------------------------------------------------------------------------


def batched_seed(
    base_seed: Any,
    bucket_key: Any,
    candidate_index: int,
    replay_index: int,
    role_name: str,
) -> int:
    """Deterministic sampling seed for one generation slot.

    The five inputs identify the slot completely: the run seed, the bucket (its
    question id), which alternative, which replay and which role. The digest is
    blake2b, so the value does not depend on the process, unlike the built in
    hash of a string.

    Returns an integer in [0, 2**31).
    """
    payload = "|".join(
        (
            _SEED_DOMAIN,
            str(int(base_seed)),
            str(bucket_key),
            str(int(candidate_index)),
            str(int(replay_index)),
            str(role_name),
        )
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % SEED_MODULUS


# -----------------------------------------------------------------------------
# Policy access
# -----------------------------------------------------------------------------


def _policy_sample_many(
    runner: ReplayRunner,
    prompts: Sequence[str],
    *,
    n: int,
    seeds: Sequence[int] | None,
    decoding: Mapping[str, Any],
) -> list[list[str]]:
    """Batched sampling, with a loop for policies that predate `sample_many`."""
    policy = getattr(runner, "_policy_obj", None)
    batched = getattr(policy, "sample_many", None)
    if callable(batched):
        out = batched(list(prompts), n=int(n), seeds=(list(seeds) if seeds is not None else None), **dict(decoding))
        return [list(texts) for texts in out]

    fallback: list[list[str]] = []
    for i, prompt in enumerate(prompts):
        dec = dict(decoding)
        if seeds is not None:
            dec["seed"] = int(seeds[i])
        fallback.append(list(runner.sample_action(prompt, dec, n=int(n))))
    return fallback


def _sample_stage(
    runner: ReplayRunner,
    prompts: Sequence[str],
    *,
    n: int,
    seeds: Sequence[int],
    decoding: Mapping[str, Any],
    batch_prompts: int,
    where: str,
) -> list[list[str]]:
    """Run one stage, cut into chunks of at most `batch_prompts` prompts."""
    prompt_list = list(prompts)
    seed_list = list(seeds)
    if len(seed_list) != len(prompt_list):
        raise BatchedReplayError(f"{where}: {len(seed_list)} seeds for {len(prompt_list)} prompts")
    if not prompt_list:
        return []

    size = max(1, int(batch_prompts))
    out: list[list[str]] = []
    for start in range(0, len(prompt_list), size):
        chunk = prompt_list[start : start + size]
        chunk_seeds = seed_list[start : start + size]
        try:
            got = _policy_sample_many(runner, chunk, n=n, seeds=chunk_seeds, decoding=decoding)
        except Exception as exc:  # noqa: BLE001 - the cell fails, with the stage named
            raise BatchedReplayError(
                f"{where}: generation failed on prompts [{start}, {start + len(chunk)}) "
                f"of {len(prompt_list)} (n={n}): {type(exc).__name__}: {exc}"
            ) from exc
        if len(got) != len(chunk):
            raise BatchedReplayError(
                f"{where}: policy returned {len(got)} results for {len(chunk)} prompts"
            )
        out.extend(got)
    return out


# -----------------------------------------------------------------------------
# Bucket construction
# -----------------------------------------------------------------------------


def _roles_after(roles_topo: Sequence[str], target_role: str) -> list[str]:
    index = {r: i for i, r in enumerate(roles_topo)}
    return list(roles_topo[index[target_role] + 1 :])


def _render(
    runner: ReplayRunner,
    state: RestartState,
    role_outputs: Mapping[str, str],
    role: str,
    *,
    where: str,
) -> str:
    try:
        return runner.prompt_renderer.render_role_prompt(
            question=state.question,
            roles_topo=state.roles_topo,
            role_outputs=dict(role_outputs),
            target_role=role,
            meta=state.meta,
        )
    except Exception as exc:  # noqa: BLE001
        raise BatchedReplayError(f"{where}: prompt rendering failed: {type(exc).__name__}: {exc}") from exc


def _collect_candidates(
    runner: ReplayRunner,
    states: Sequence[RestartState],
    cfg: ReplayConfig,
    *,
    target_prompts: Sequence[str],
    decoding_base: Mapping[str, Any],
    base_seed: int,
    total_req: int,
    batch_prompts: int,
    forced: Sequence[str] = (),
) -> list[list[str]]:
    """Stage A: alternatives for the target role of every bucket.

    Mirrors `run_bucket`: forced actions first, then unique texts only, the
    optional first sample that puts a resampled action at j=0, and at most
    `min(30, max(5, 2 * total_req))` rounds of topping up.
    """
    candidates: list[list[str]] = [[] for _ in states]
    seen: list[set[str]] = [set() for _ in states]
    if total_req <= 0:
        return candidates

    keys = [str(s.question_id) for s in states]

    # The forced actions go in ahead of every sampled alternative, so the E3a
    # null arm is j=0 in every bucket. Same list for every bucket, so the
    # `not candidates` guard of `run_bucket` becomes `not forced` here.
    if forced:
        for b in range(len(states)):
            _unique_extend(candidates[b], seen[b], forced, total_req)

    if cfg.include_real_as_j0 and not forced:
        seeds = [batched_seed(base_seed, k, _CANDIDATE_STAGE, 0, cfg.target_role) for k in keys]
        texts = _sample_stage(
            runner,
            target_prompts,
            n=1,
            seeds=seeds,
            decoding=decoding_base,
            batch_prompts=batch_prompts,
            where="stage A (real action at j=0)",
        )
        for b, got in enumerate(texts):
            _unique_extend(candidates[b], seen[b], got, total_req)

    attempts = 0
    max_attempts = min(30, max(5, 2 * total_req))  # hard bound; don't burn GPU
    while attempts < max_attempts:
        short = [b for b in range(len(states)) if len(candidates[b]) < total_req]
        if not short:
            break

        # sample_many takes one n for the whole call, so buckets are grouped by
        # how many alternatives they still need. Without de-duplication losses
        # there is exactly one group, hence exactly one call.
        by_need: dict[int, list[int]] = {}
        for b in short:
            by_need.setdefault(total_req - len(candidates[b]), []).append(b)

        for need in sorted(by_need):
            group = by_need[need]
            seeds = [
                batched_seed(base_seed, keys[b], _CANDIDATE_STAGE, 1 + attempts, cfg.target_role) for b in group
            ]
            texts = _sample_stage(
                runner,
                [target_prompts[b] for b in group],
                n=need,
                seeds=seeds,
                decoding=decoding_base,
                batch_prompts=batch_prompts,
                where=f"stage A (round {attempts}, need {need}, {len(group)} buckets)",
            )
            for b, got in zip(group, texts):
                _unique_extend(candidates[b], seen[b], got, total_req)

        attempts += 1

    return [found[:total_req] for found in candidates]


def run_buckets_batched(
    runner: ReplayRunner,
    restart_states: Sequence[RestartState],
    cfg: ReplayConfig,
    *,
    target_role: str,
    next_role: str | None,
    batch_prompts: int = DEFAULT_BATCH_PROMPTS,
    forced_actions: Sequence[str] | None = None,
    reuse: CandidateReuse | None = None,
) -> list[Bucket]:
    """Build one bucket per restart state, batching every stage across buckets.

    Same result shape as calling `runner.run_bucket(state, cfg)` for each state:
    same de-duplication, same retry bound, same `include_real_as_j0` handling,
    same `record_next_teammate` handling, same meta keys plus `seed_scheme`. The
    sampled texts differ from the sequential path because the seeds are derived
    differently, which is the point: the sequential path gives every replay of a
    bucket the same seed.

    `forced_actions` is the `run_bucket` parameter of the same name, applied to
    every bucket of the call: the texts are placed ahead of the sampled
    alternatives, so the E3a null arm is j=0 everywhere. They count against
    `cfg.num_candidates` exactly as they do in `run_bucket`, so a caller that
    wants n sampled alternatives plus one injected arm asks for n + 1.

    `target_role` and `next_role` name the measured position. They must agree
    with `cfg`, which already carries them; a mismatch is refused rather than
    silently resolved.

    Seeds key off the question id, so two restart states that carry the same
    question id in one call draw from the same seeds. Their prompts still
    differ if their prefixes differ, so the texts still differ; identical
    prompts would mean identical contexts anyway.

    `reuse`, when given, is an index of the alternatives an earlier run
    recorded, and it replaces stage A: every bucket replays the texts the source
    holds for its decision point, in the recorded j order, while stages B and C
    run untouched and therefore follow this run's seed. A decision point the
    source does not cover raises CandidateReuseError, naming the first one, and
    no bucket is returned.
    """
    states = list(restart_states)
    if not states:
        return []

    if target_role != cfg.target_role:
        raise ValueError(f"target_role={target_role!r} disagrees with cfg.target_role={cfg.target_role!r}")
    if next_role != cfg.next_role:
        raise ValueError(f"next_role={next_role!r} disagrees with cfg.next_role={cfg.next_role!r}")

    roles_topo = list(states[0].roles_topo)
    for state in states:
        # The stages are organized around one role order, so a mixed batch would
        # silently mean different things per bucket.
        if list(state.roles_topo) != roles_topo:
            raise ValueError(
                f"batched replay needs one role order per call; question {state.question_id!r} "
                f"has {list(state.roles_topo)!r}, expected {roles_topo!r}"
            )
        runner._validate_cfg(state, cfg)

    roles_after = _roles_after(roles_topo, cfg.target_role)
    n_replays = int(cfg.num_completions_per_candidate)

    decoding_base = dict(cfg.decoding)
    base_seed = decoding_base.get("seed", None)
    if base_seed is None:
        base_seed = int(runner.runner_meta.get("seed") or 0)
        decoding_base["seed"] = base_seed
    else:
        base_seed = int(base_seed)

    credit_n_req = max(0, int(cfg.num_candidates))
    extra_n_req = max(0, int(cfg.num_extra_v_samples))
    total_req = max(0, credit_n_req + extra_n_req)

    ctx_hashes = [runner.build_context_hash(state, cfg.target_role) for state in states]
    target_prompts = [
        _render(runner, state, state.role_outputs_prefix, cfg.target_role, where=f"bucket {b} ({state.question_id!r})")
        for b, state in enumerate(states)
    ]

    if reuse is not None:
        if forced_actions:
            raise CandidateReuseError(
                "reused alternatives already fix what sits at every j, so they cannot be combined with "
                "a forced action, which claims j=0. The null arm of E3a is sampled, not reused."
            )
        # In order, so the refusal names the first decision point the source misses.
        candidates = [
            reuse.texts_for(state.question_id, ctx_hashes[b], total_req=total_req)
            for b, state in enumerate(states)
        ]
    else:
        candidates = _collect_candidates(
            runner,
            states,
            cfg,
            target_prompts=target_prompts,
            decoding_base=decoding_base,
            base_seed=base_seed,
            total_req=total_req,
            batch_prompts=batch_prompts,
            forced=[str(text) for text in (forced_actions or ())],
        )

    # Stage B. One dict of role outputs per (bucket, candidate, replay), grown
    # role by role in topological order so that each role sees what the
    # sequential path would have shown it.
    keys = [str(state.question_id) for state in states]
    role_outputs: dict[tuple[int, int, int], dict[str, str]] = {}
    captured_next: dict[tuple[int, int, int], str | None] = {}
    for b, state in enumerate(states):
        for j, action_text in enumerate(candidates[b]):
            base_out = dict(state.role_outputs_prefix)
            base_out[cfg.target_role] = action_text
            for r in range(n_replays):
                role_outputs[(b, j, r)] = dict(base_out)
                captured_next[(b, j, r)] = None

    for r in range(n_replays):
        for role in roles_after:
            owners: list[tuple[int, int]] = []
            prompts: list[str] = []
            seeds: list[int] = []
            for b, state in enumerate(states):
                for j in range(len(candidates[b])):
                    owners.append((b, j))
                    prompts.append(
                        _render(
                            runner,
                            state,
                            role_outputs[(b, j, r)],
                            role,
                            where=f"stage B (role {role}, replay {r}, bucket {b}, candidate {j})",
                        )
                    )
                    seeds.append(batched_seed(base_seed, keys[b], j, r, role))

            texts = _sample_stage(
                runner,
                prompts,
                n=1,
                seeds=seeds,
                decoding=decoding_base,
                batch_prompts=batch_prompts,
                where=f"stage B (role {role}, replay {r})",
            )

            for (b, j), got in zip(owners, texts):
                text = got[0] if got else ""
                role_outputs[(b, j, r)][role] = text
                if cfg.record_next_teammate and cfg.next_role == role and captured_next[(b, j, r)] is None:
                    captured_next[(b, j, r)] = text

    # Stage C.
    buckets: list[Bucket] = []
    for b, state in enumerate(states):
        results: list[CandidateResult] = []
        for j, action_text in enumerate(candidates[b]):
            returns: list[float] = []
            next_actions: list[str] = []
            for r in range(n_replays):
                try:
                    value = float(
                        runner.evaluator.evaluate(
                            restart=state,
                            role_outputs=role_outputs[(b, j, r)],
                            meta=runner.runner_meta,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    raise BatchedReplayError(
                        f"stage C: evaluation failed for bucket {b} ({state.question_id!r}), "
                        f"candidate {j}, replay {r}: {type(exc).__name__}: {exc}"
                    ) from exc
                returns.append(value)
                if captured_next[(b, j, r)] is not None:
                    next_actions.append(str(captured_next[(b, j, r)]))
            results.append(CandidateResult(action_text=action_text, returns=returns, next_actions=next_actions))

        credit_n = min(credit_n_req, len(candidates[b]))
        extra_n = max(0, len(candidates[b]) - credit_n)

        meta = dict(runner.runner_meta)
        meta.update(
            {
                "task_spec": runner._safe_task_id(),
                "decoding": dict(decoding_base),
                "seed": decoding_base.get("seed"),
                "record_next_teammate": cfg.record_next_teammate,
                "next_role": cfg.next_role,
                "include_real_as_j0": cfg.include_real_as_j0,
                "real_j": 0 if (cfg.include_real_as_j0 and candidates[b]) else None,
                "credit_n": credit_n,
                "v_extra_start": credit_n,
                "v_extra_n": extra_n,
                "candidate_total_req": total_req,
                "candidate_total_got": len(candidates[b]),
                "seed_scheme": SEED_SCHEME,
            }
        )
        if meta.get("real_j") is None:
            meta.pop("real_j", None)
        if reuse is not None:
            meta.update(reuse_meta(reuse, len(candidates[b])))

        buckets.append(
            Bucket(
                ctx_hash=ctx_hashes[b],
                restart=state,
                candidates=results,
                meta=meta,
                target_role=cfg.target_role,
            )
        )

    return buckets
