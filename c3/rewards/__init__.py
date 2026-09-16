"""Reward providers for C3 rollouts.

``base`` holds the request and result records, ``providers`` the provider
implementations, and ``registry`` the one builder the rollout generator calls.

This file exists so that the directory is a regular package. Without it
``[tool.setuptools.packages.find]`` in pyproject.toml does not collect the
directory, and an installed copy of C3 has no ``c3.rewards`` to import even
though a source checkout resolves it from the working directory.

Keep this module dependency-light: it must stay importable without the
training stack.
"""

from __future__ import annotations

__all__ = ["base", "providers", "registry"]
