"""Row chunking of the experience-phase log-probability forwards.

Two layers, because the two things worth pinning live at different depths.

The first layer reads the two `forward` methods as source. They sit in modules that
import deepspeed, transformers and ray at module scope, which no CPU-only machine has,
so the checks parse the files instead of importing them: both forwards must read
`--logprob_micro_batch_size`, both must leave the switch off for packed sequences, and
both must go through the shared helper.

The second layer runs the helper itself. `openrlhf/trainer/ray/logprob_chunking.py`
imports nothing but torch, so a machine with torch can check the claim that actually
matters: chunking returns the same tensor as one call, bit for bit, for a forward whose
rows are independent. It is loaded from its path rather than imported by name, because
importing `openrlhf.trainer.ray` runs that package's `__init__`, which pulls in ray.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PPO_ACTOR = REPO_ROOT / "openrlhf" / "trainer" / "ray" / "ppo_actor.py"
LAUNCHER = REPO_ROOT / "openrlhf" / "trainer" / "ray" / "launcher.py"
TOOLING = REPO_ROOT / "openrlhf" / "cli" / "train_ppo_ray_tooling.py"
CHUNKING = REPO_ROOT / "openrlhf" / "trainer" / "ray" / "logprob_chunking.py"


# ---------------------------------------------------------------------------
# Layer one: the two forwards, read as source
# ---------------------------------------------------------------------------


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.unparse(item)
            raise AssertionError(f"{path.name}: class {class_name} has no method {method_name}")
    raise AssertionError(f"{path.name}: no class {class_name}")


FORWARDS = (
    (PPO_ACTOR, "PolicyModelActor"),
    (LAUNCHER, "ReferenceModelActor"),
)


@pytest.mark.parametrize("path, class_name", FORWARDS, ids=("policy", "reference"))
def test_both_forwards_read_the_chunk_switch(path: Path, class_name: str) -> None:
    source = _method_source(path, class_name, "forward")
    assert "logprob_micro_batch_size" in source, f"{class_name}.forward does not read the switch"
    assert "chunked_forward" in source, f"{class_name}.forward does not go through the shared helper"


@pytest.mark.parametrize("path, class_name", FORWARDS, ids=("policy", "reference"))
def test_both_forwards_leave_packed_sequences_alone(path: Path, class_name: str) -> None:
    """Packed sequences carry their lengths in a parallel list, so rows cannot be sliced."""
    source = _method_source(path, class_name, "forward")
    assert "packed_seq_lens is None" in source, f"{class_name}.forward has no packed-sequence guard"
    guard = source.index("packed_seq_lens is None")
    switch = source.index("logprob_micro_batch_size")
    assert guard < switch, f"{class_name}.forward reads the switch before guarding packed sequences"


def test_the_switch_is_an_integer_that_defaults_to_off() -> None:
    tree = ast.parse(TOOLING.read_text(encoding="utf-8"), filename=str(TOOLING))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "--logprob_micro_batch_size":
            continue
        found.append({kw.arg: kw.value for kw in node.keywords})

    assert len(found) == 1, "expected exactly one --logprob_micro_batch_size argument"
    keywords = found[0]
    assert isinstance(keywords.get("type"), ast.Name) and keywords["type"].id == "int"
    assert isinstance(keywords.get("default"), ast.Constant) and keywords["default"].value == 0
    assert "help" in keywords, "the switch has no help text"


def test_the_reference_actor_moves_its_weights_at_the_batch_layer() -> None:
    """The offload round trip belongs to execute_batch, not to each item's forward."""
    source = _method_source(LAUNCHER, "ReferenceModelActor", "execute_batch")
    assert "_on_device" in source, "execute_batch does not mark the weights as resident"
    assert "empty_cache" in source, "execute_batch does not give the memory back"

    forward = _method_source(LAUNCHER, "ReferenceModelActor", "forward")
    assert "_on_device" in forward, "forward would move weights that execute_batch already moved"


# ---------------------------------------------------------------------------
# Layer two: the helper, run for real
# ---------------------------------------------------------------------------


def _load_chunking() -> ModuleType:
    pytest.importorskip("torch")
    name = "_c3_test_logprob_chunking"
    spec = importlib.util.spec_from_file_location(name, CHUNKING)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


class _RowForward:
    """A stand-in forward whose rows are independent, and which records what it saw.

    The output of row i depends on row i alone, which is the property that makes the
    chunking valid in the first place. Every column differs, so a join that reorders or
    drops a row cannot pass by accident.
    """

    def __init__(self, torch_module, actions: int = 3):
        self._torch = torch_module
        self._actions = actions
        self.row_counts: list[int] = []
        self.masks_seen: list[object] = []

    def __call__(self, sequences, action_mask, attention_mask):
        torch = self._torch
        self.row_counts.append(int(sequences.shape[0]))
        self.masks_seen.append(action_mask)
        base = sequences.float().sum(-1, keepdim=True)
        columns = torch.arange(self._actions, dtype=torch.float32).reshape(1, self._actions)
        out = base * 1.5 + columns
        if isinstance(action_mask, torch.Tensor):
            out = out * action_mask.float()
        return out


def _batch(torch, rows: int, length: int = 5, actions: int = 3):
    sequences = torch.arange(rows * length, dtype=torch.long).reshape(rows, length)
    action_mask = torch.ones(rows, actions, dtype=torch.long)
    attention_mask = torch.ones(rows, length, dtype=torch.long)
    return sequences, action_mask, attention_mask


def test_chunking_returns_the_same_tensor_as_one_call() -> None:
    module = _load_chunking()
    torch = module.torch
    sequences, action_mask, attention_mask = _batch(torch, rows=16)

    whole = module.chunked_forward(_RowForward(torch), sequences, action_mask, attention_mask, 0)
    for chunk in (1, 2, 3, 4, 5, 8, 15):
        split_fn = _RowForward(torch)
        split = module.chunked_forward(
            fn=split_fn,
            sequences=sequences,
            action_mask=action_mask,
            attention_mask=attention_mask,
            chunk=chunk,
        )
        assert torch.equal(whole, split), f"chunk={chunk} did not reproduce the single call"
        assert split.shape == whole.shape
        assert max(split_fn.row_counts) <= chunk, f"chunk={chunk} let a call see more rows than that"
        assert sum(split_fn.row_counts) == 16, f"chunk={chunk} did not cover the batch exactly once"


def test_the_last_chunk_is_short_when_the_batch_does_not_divide() -> None:
    module = _load_chunking()
    torch = module.torch
    sequences, action_mask, attention_mask = _batch(torch, rows=7)

    whole = module.chunked_forward(_RowForward(torch), sequences, action_mask, attention_mask, 0)
    split_fn = _RowForward(torch)
    split = module.chunked_forward(split_fn, sequences, action_mask, attention_mask, 3)

    assert split_fn.row_counts == [3, 3, 1]
    assert torch.equal(whole, split)


def test_a_batch_no_larger_than_one_chunk_takes_the_single_call() -> None:
    module = _load_chunking()
    torch = module.torch
    sequences, action_mask, attention_mask = _batch(torch, rows=4)

    for chunk in (4, 5, 99):
        fn = _RowForward(torch)
        module.chunked_forward(fn, sequences, action_mask, attention_mask, chunk)
        assert fn.row_counts == [4], f"chunk={chunk} split a batch it did not need to split"

    for chunk in (0, -1):
        fn = _RowForward(torch)
        module.chunked_forward(fn, sequences, action_mask, attention_mask, chunk)
        assert fn.row_counts == [4], f"chunk={chunk} is off and must not split"


def test_a_sequence_tensor_that_is_not_a_row_batch_takes_the_single_call() -> None:
    """One dimension or three is not a row batch, and is handed on as it arrived."""
    module = _load_chunking()
    torch = module.torch

    flat = torch.arange(30, dtype=torch.long)
    fn = _RowForward(torch)
    module.chunked_forward(fn, flat, None, None, 2)
    assert len(fn.row_counts) == 1

    stacked = torch.arange(6 * 5 * 2, dtype=torch.long).reshape(6, 5, 2)
    fn = _RowForward(torch)
    module.chunked_forward(fn, stacked, None, None, 2)
    assert len(fn.row_counts) == 1


def test_a_mask_that_is_not_a_tensor_reaches_every_chunk_unchanged() -> None:
    """`action_mask` is an action count rather than a tensor on some upstream paths."""
    module = _load_chunking()
    torch = module.torch
    sequences, _, attention_mask = _batch(torch, rows=6)

    fn = _RowForward(torch)
    module.chunked_forward(fn, sequences, 3, attention_mask, 2)
    assert fn.masks_seen == [3, 3, 3]

    fn = _RowForward(torch)
    module.chunked_forward(fn, sequences, None, attention_mask, 2)
    assert fn.masks_seen == [None, None, None]


def test_the_result_comes_back_on_the_cpu() -> None:
    module = _load_chunking()
    torch = module.torch
    sequences, action_mask, attention_mask = _batch(torch, rows=6)

    for chunk in (0, 2):
        out = module.chunked_forward(_RowForward(torch), sequences, action_mask, attention_mask, chunk)
        assert out.device.type == "cpu"
