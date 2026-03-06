from __future__ import annotations

import importlib
from typing import Any

from c3.rewards.providers import (
    AutoRewardProvider,
    ChainedRewardProvider,
    EnvRewardProvider,
    NoneRewardProvider,
    RemoteRMRewardProvider,
)


_BUILTIN = {
    "auto": AutoRewardProvider,
    "chain": ChainedRewardProvider,
    "env": EnvRewardProvider,
    "remote_rm": RemoteRMRewardProvider,
    "none": NoneRewardProvider,
}


def build_reward_provider(name: str, *, env_name: str, remote_reward_model) -> Any:
    """name can be builtin (auto/env/remote_rm/chain/none) or a full import path 'pkg.mod:Cls'."""
    name = (name or "auto").strip()

    if name in _BUILTIN:
        cls = _BUILTIN[name]
        if name == "env":
            return cls(env_name=env_name)
        if name == "remote_rm":
            return cls(remote_reward_model=remote_reward_model)
        if name in ("auto", "chain"):
            return cls(env_name=env_name, remote_reward_model=remote_reward_model)
        return cls()

    # custom class path: "a.b.c:MyProvider"
    if ":" not in name:
        raise ValueError(f"Unknown reward_provider_cls={name!r}. Use builtin: {sorted(_BUILTIN)} or 'pkg.mod:Cls'.")

    mod, cls_name = name.split(":", 1)
    m = importlib.import_module(mod)
    cls = getattr(m, cls_name)
    return cls(env_name=env_name, remote_reward_model=remote_reward_model)
