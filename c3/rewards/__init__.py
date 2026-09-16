"""Reward providers for C3 rollouts.

``base`` holds the request and result records, ``providers`` the provider
implementations, and ``registry`` the one builder the rollout generator calls.

This file exists so that the directory is a regular package, like every other
subpackage of ``c3``, and so that it declares its own public surface. It is not
a packaging fix: ``[tool.setuptools.packages.find]`` leaves ``namespaces`` on by
default, so the three modules were collected into the wheel without it too.

Keep this module dependency-light: it must stay importable without the
training stack.
"""

from __future__ import annotations

__all__ = ["base", "providers", "registry"]
