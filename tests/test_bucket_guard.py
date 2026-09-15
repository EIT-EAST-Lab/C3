# tests/test_bucket_guard.py
"""The context-key collision guard of the bucket validator.

Background (driver ruling C3, 2026-09-15): on the first real E1 run the very
first bucket ever written was reported as a context-key collision.
`ReplayRunner.build_context_hash` observes the key on the process-global guard
with a fingerprint of the context TEXT it hashed, while
`c3.analysis.buckets.validate_bucket` observes the same key with a fingerprint
of its own identity string, which is a different string by construction. Sharing
one guard therefore made every bucket look like a collision. The fix (commit
ff183d4) gives the validator its own guard; these tests are the reproduction the
ruling asked for, plus the check that a real collision is still refused.

The runner's context string is rebuilt here rather than imported, so the test
stays inside the light analysis environment.

    C:/Users/10350/.venvs/c3-light/Scripts/python.exe -m pytest tests/test_bucket_guard.py -q
"""

from __future__ import annotations

import json

import pytest

from c3.analysis import buckets as bk
from c3.utils.collision_guard import CollisionGuard, ContextKeyCollisionError, global_guard
from c3.utils.context_key import fingerprint, hash63

QUESTION = "A train leaves the station at 9am."
ROLES = ["Reasoner", "Actor"]
TARGET = "Actor"
PREFIX = {"Reasoner": "first work out the rate"}


def runner_context_text(question=QUESTION, roles_topo=ROLES, target_role=TARGET,
                        role_outputs_prefix=None):
    """The text `ReplayRunner.build_context_hash` hashes in its default branch.

    Copied from c3/analysis/replay.py (the payload and its stable JSON dump), so
    that this test does not import the replay module: the point is exactly that
    the two sides build different strings for the same decision point.
    """
    payload = {
        "question": question,
        "roles_topo": list(roles_topo),
        "target_role": target_role,
        "role_outputs_prefix": dict(sorted((role_outputs_prefix or PREFIX).items())),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_bucket(ctx_hash, *, question_id="q0", prefix=None, bucket_id="bkt_0"):
    """A minimal bucket in the schema of c3.analysis.buckets."""
    return {
        "bucket_id": bucket_id,
        "ctx_hash": int(ctx_hash),
        "target_role": TARGET,
        "question_id": question_id,
        "restart": {"roles_topo": list(ROLES),
                    "role_outputs_prefix": dict(PREFIX if prefix is None else prefix)},
        "candidates": [
            {"j": 0, "action_text": "one", "returns": [1.0], "next_actions": ["\\boxed{1}"]},
            {"j": 1, "action_text": "two", "returns": [0.0], "next_actions": ["\\boxed{2}"]},
        ],
        "meta": {"workflow": "a2", "model": "4b", "rule": "sweep_n2"},
    }


@pytest.fixture
def fresh_guards(monkeypatch):
    """A clean validator guard and a clean global guard around each test.

    The validator's guard is module state and the global guard is process state;
    without this a later test would inherit whatever an earlier one observed.
    """
    monkeypatch.setattr(bk, "_validation_guard", CollisionGuard())
    global_guard().reset()
    yield
    global_guard().reset()


def test_the_validator_ignores_what_the_replay_runner_observed(fresh_guards, monkeypatch):
    """The 2026-09-15 reproduction: the runner has already observed this ctx_hash
    with its own fingerprint, and the bucket still validates."""
    ctx_text = runner_context_text()
    ctx_hash = hash63(ctx_text)
    global_guard().observe(ctx_hash, fingerprint(ctx_text),
                           where="ReplayRunner/build_context_hash")

    bucket = make_bucket(ctx_hash)
    bk.validate_bucket(bucket)                       # must not raise

    # Why one shared guard could not work: the two sides fingerprint different
    # strings for the same decision point.
    identity = bk._context_identity_string(bucket)
    assert fingerprint(identity) != fingerprint(ctx_text)
    with pytest.raises(ContextKeyCollisionError):
        global_guard().observe(ctx_hash, fingerprint(identity), where="test")

    # And the failure as it happened: point the validator back at the global
    # guard, which is what the code did before ff183d4, and the same bucket is
    # refused on its first sighting.
    global_guard().reset()
    global_guard().observe(ctx_hash, fingerprint(ctx_text),
                           where="ReplayRunner/build_context_hash")
    monkeypatch.setattr(bk, "_validation_guard", global_guard())
    with pytest.raises(bk.BucketValidationError) as exc:
        bk.validate_bucket(make_bucket(ctx_hash, bucket_id="bkt_before_the_fix"))
    assert "collision" in str(exc.value).lower()


def test_the_validator_still_refuses_a_real_collision(fresh_guards):
    """Two buckets with different contexts and the same ctx_hash: the second one
    is refused, with the offending field named."""
    ctx_hash = hash63(runner_context_text())
    first = make_bucket(ctx_hash, question_id="q0", bucket_id="bkt_0")
    second = make_bucket(ctx_hash, question_id="q1", bucket_id="bkt_1")
    assert bk._context_identity_string(first) != bk._context_identity_string(second)

    bk.validate_bucket(first)
    with pytest.raises(bk.BucketValidationError) as exc:
        bk.validate_bucket(second)
    assert "$.ctx_hash" in str(exc.value)
    assert "collision" in str(exc.value).lower()

    # The same disagreement, one step further down: same question, different
    # upstream output, same key.
    other_prefix = make_bucket(ctx_hash, question_id="q0", bucket_id="bkt_2",
                               prefix={"Reasoner": "a different prefix"})
    with pytest.raises(bk.BucketValidationError):
        bk.validate_bucket(other_prefix)


def test_the_same_context_twice_is_not_a_collision(fresh_guards):
    """Re-reading one bucket, or two bucket files holding the same decision point,
    observes the same fingerprint and passes."""
    ctx_hash = hash63(runner_context_text())
    bucket = make_bucket(ctx_hash)
    twin = make_bucket(ctx_hash, bucket_id="bkt_from_another_file")
    bk.validate_bucket(bucket)
    bk.validate_bucket(bucket)
    bk.validate_bucket(twin)
    assert bk._validation_guard.size() == 1


def test_the_two_guards_are_not_the_same_object(fresh_guards):
    """The structural fact the fix rests on."""
    assert bk._validation_guard is not global_guard()
