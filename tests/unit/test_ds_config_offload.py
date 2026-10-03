"""The DeepSpeed training config when the optimizer is offloaded to the CPU.

`get_train_ds_config` only assembles a dictionary, so these checks need neither a GPU nor a
DeepSpeed install. The module it lives in imports `deepspeed` and `torch` at module scope,
though, and the CPU tier of this repository has neither, so the loader below stands in empty
modules for whichever of the two is missing and loads the file directly. On a machine that has
the real packages nothing is stood in, and the same assertions run against the real import.
"""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = REPO_ROOT / "openrlhf" / "utils" / "deepspeed" / "deepspeed_utils.py"

# What the module needs at import time: `import deepspeed`, `import torch` and
# `from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus`.
_STUB_PACKAGES = (
    "torch",
    "deepspeed",
    "deepspeed.runtime",
    "deepspeed.runtime.zero",
    "deepspeed.runtime.zero.partition_parameters",
)


def _install_stub_packages() -> list[str]:
    """Put empty modules in sys.modules for the imports the CPU tier cannot satisfy."""
    if importlib.util.find_spec("deepspeed") is not None and importlib.util.find_spec("torch") is not None:
        return []

    installed: list[str] = []
    for name in _STUB_PACKAGES:
        if name in sys.modules:
            continue
        module = ModuleType(name)
        if "." in name:
            parent_name, _, leaf = name.rpartition(".")
            setattr(sys.modules[parent_name], leaf, module)
        sys.modules[name] = module
        installed.append(name)

    partition = sys.modules["deepspeed.runtime.zero.partition_parameters"]
    if not hasattr(partition, "ZeroParamStatus"):
        partition.ZeroParamStatus = type("ZeroParamStatus", (), {"NOT_AVAILABLE": "not_available"})
    return installed


@contextmanager
def _loaded_module():
    installed = _install_stub_packages()
    name = "_c3_test_deepspeed_utils"
    try:
        spec = importlib.util.spec_from_file_location(name, TARGET)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {TARGET}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)
        for stub in reversed(installed):
            sys.modules.pop(stub, None)


@pytest.fixture(scope="module")
def ds_utils():
    with _loaded_module() as module:
        yield module


def test_optimizer_offload_turns_off_the_deepspeed_cpu_optimizer_check(ds_utils) -> None:
    config = ds_utils.get_train_ds_config(offload=False, adam_offload=True, stage=2)
    assert config["zero_force_ds_cpu_optimizer"] is False
    # The key is read at the top level of the config, not inside zero_optimization.
    assert "zero_force_ds_cpu_optimizer" not in config["zero_optimization"]


def test_the_check_is_left_alone_when_the_optimizer_stays_on_the_gpu(ds_utils) -> None:
    config = ds_utils.get_train_ds_config(offload=False, adam_offload=False, stage=2)
    assert "zero_force_ds_cpu_optimizer" not in config


def test_the_offload_switch_still_drives_the_optimizer_device(ds_utils) -> None:
    on = ds_utils.get_train_ds_config(offload=False, adam_offload=True, stage=2)
    off = ds_utils.get_train_ds_config(offload=False, adam_offload=False, stage=2)
    assert on["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert off["zero_optimization"]["offload_optimizer"]["device"] == "none"


def test_the_evaluation_config_never_carries_the_key(ds_utils) -> None:
    for offload in (True, False):
        config = ds_utils.get_eval_ds_config(offload=offload, stage=0)
        assert "zero_force_ds_cpu_optimizer" not in config
