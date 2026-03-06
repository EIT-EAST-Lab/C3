"""
C3 configs loader: roles/tasks -> runtime specs (single source of truth).

This module is the ONLY loader for task/roles assets in the "flattened" repo layout:

  repo_root/
    configs/
      tasks/
      roles/
    c3/
      integration/marl_specs.py   <-- (this file)

Design goals:
  - File-centric: tasks and roles are normal files under repo_root/configs/.
  - Robust path resolution:
      * supports relative paths (recommended)
      * supports legacy absolute paths from older layouts (e.g., <REPO_ROOT>/configs/roles/...)
  - Works for N=1 (single agent) and N>1 (multi-agent), with explicit topo order by depends_on.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml


# ---------------------------
# Repo/configs root discovery
# ---------------------------

_THIS_FILE = Path(__file__).resolve()
_PKG_ROOT = _THIS_FILE.parents[1]  # .../repo_root/c3


def _discover_repo_root(start: Path) -> Path:
    """Find repo root by walking up until configs/tasks exists.

    We intentionally do NOT depend on git. This works in a normal source checkout
    and also in many "copied directory" environments.
    """
    for p in [start] + list(start.parents):
        if (p / "configs" / "tasks").is_dir():
            # Optional sanity: also expect c3/ at repo root in our repo layout.
            if (p / "c3").exists():
                return p
            # If configs/tasks exists, it's still a very strong signal.
            return p

    # Fallback: assume layout repo_root/c3/integration/marl_specs.py
    # __file__ parents: integration -> c3 -> repo_root
    try:
        return start.parents[2]
    except IndexError:
        return start.parent


_REPO_ROOT = _discover_repo_root(_THIS_FILE)
_CONFIGS_ROOT = _REPO_ROOT / "configs"

# If someone packages configs inside c3/ (not recommended), allow fallback.
if not _CONFIGS_ROOT.exists():
    pkg_cfg = _PKG_ROOT / "configs"
    if pkg_cfg.exists():
        _CONFIGS_ROOT = pkg_cfg


# ---------------------------
# Data model
# ---------------------------

@dataclass(frozen=True)
class RoleSpec:
    """A single role/agent definition."""

    name: str
    prompt: str
    with_answer: bool
    depends_on: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    """A task definition loaded from C3 task yaml."""

    repo_root: str
    task_path: str
    experiment_name: str
    env_name: str
    roles_path: str
    environment: Mapping[str, Any]
    mas: Mapping[str, Any]
    roles: Tuple[RoleSpec, ...]
    train_datasets: Tuple[Mapping[str, Any], ...]
    eval_suites: Tuple[Mapping[str, Any], ...]

    def topo_role_names(self) -> List[str]:
        return [r.name for r in topo_sort_roles(self.roles)]


# ---------------------------
# Path resolution helpers
# ---------------------------

def _normalize_path(p: str) -> str:
    return p.replace("\\", "/")


def _expand_path(p: str) -> str:
    # Expand environment variables and user home.
    return os.path.expanduser(os.path.expandvars(p))


def _map_legacy_configs_path(path_like: str, *, subdir: str) -> Optional[str]:
    """Map legacy absolute paths into the current repo configs, if possible.

    We accept a few historical patterns:
      - <REPO_ROOT>/configs/<subdir>/...
      - .../C3/configs/<subdir>/...
    """
    s = _normalize_path(_expand_path(path_like))

    markers = [
        f"/C3/configs/{subdir}/",
        f"/configs/{subdir}/",  # generic absolute configs path
    ]
    for marker in markers:
        if marker in s:
            suffix = s.split(marker, 1)[1]
            cand_repo = (_CONFIGS_ROOT / subdir / suffix).resolve()
            if cand_repo.exists():
                return str(cand_repo)
            # fallback to package configs if present
            cand_pkg = (_PKG_ROOT / "configs" / subdir / suffix).resolve()
            if cand_pkg.exists():
                return str(cand_pkg)
            # If neither exists, still return the repo candidate (best-effort for clearer errors).
            return str(cand_repo)

    return None


def resolve_path(path_like: str, *, base_dir: Optional[Path] = None, subdir_hint: Optional[str] = None) -> str:
    """Resolve a path string with best-effort legacy mapping.

    Rules:
      0) Expand env vars and ~.
      1) If it looks like a legacy configs/<subdir_hint>/... absolute path, map it into current configs root.
      2) If the path exists as-is, use it.
      3) If relative and base_dir is given, resolve relative to base_dir.
      4) If relative and subdir_hint is given, try resolving relative to configs/<subdir_hint>/.
      5) Best-effort return resolved candidate for clearer error messages.
    """
    raw = _expand_path(path_like)

    if subdir_hint:
        mapped = _map_legacy_configs_path(raw, subdir=subdir_hint)
        if mapped is not None:
            # mapped may or may not exist; we prefer it because it points at current layout.
            if Path(mapped).exists():
                return str(Path(mapped).resolve())
            # If not exists, still allow later checks to raise with mapped path.
            return mapped

    p = Path(raw)
    if p.exists():
        return str(p.resolve())

    if not p.is_absolute() and base_dir is not None:
        cand = (base_dir / p).resolve()
        if cand.exists():
            return str(cand)

    if not p.is_absolute() and subdir_hint:
        cand = (_CONFIGS_ROOT / subdir_hint / p).resolve()
        if cand.exists():
            return str(cand)
        # also try package configs as fallback
        cand2 = (_PKG_ROOT / "configs" / subdir_hint / p).resolve()
        if cand2.exists():
            return str(cand2)

    # Last resort: return the most informative candidate
    if not p.is_absolute() and base_dir is not None:
        return str((base_dir / p).resolve())
    if not p.is_absolute() and subdir_hint:
        return str((_CONFIGS_ROOT / subdir_hint / p).resolve())
    return str(p)


# ---------------------------
# Role graph utilities
# ---------------------------

def _ensure_unique(names: Sequence[str]) -> None:
    seen = set()
    dup = []
    for n in names:
        if n in seen:
            dup.append(n)
        seen.add(n)
    if dup:
        raise ValueError(f"duplicate role names: {sorted(set(dup))}")


def topo_sort_roles(roles: Sequence[RoleSpec]) -> List[RoleSpec]:
    """Topologically sort roles by depends_on.

    Works for N=1.
    Raises ValueError on missing dependency or cycle.
    """
    if not roles:
        raise ValueError("roles list is empty")

    names = [r.name for r in roles]
    _ensure_unique(names)
    by_name = {r.name: r for r in roles}

    indeg: Dict[str, int] = {n: 0 for n in names}
    out_edges: Dict[str, List[str]] = {n: [] for n in names}

    for r in roles:
        for dep in r.depends_on:
            if dep not in by_name:
                raise ValueError(f"role {r.name} depends_on unknown role {dep}")
            out_edges[dep].append(r.name)
            indeg[r.name] += 1

    queue = [n for n in names if indeg[n] == 0]
    out: List[str] = []
    while queue:
        n = queue.pop(0)
        out.append(n)
        for nxt in out_edges[n]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)

    if len(out) != len(names):
        stuck = [n for n in names if indeg[n] > 0]
        raise ValueError(f"cycle detected in roles depends_on graph: {stuck}")

    return [by_name[n] for n in out]


# ---------------------------
# Public API: load roles/tasks
# ---------------------------

def load_roles(roles_path: str) -> List[RoleSpec]:
    """Load roles JSON.

    Expected C3 format (list of dict):
      - role (str) or name (str)
      - prompt (str)
      - with_answer (bool)
      - depends_on (list[str], optional)
    """
    # Prefer configs/roles as the default base.
    resolved = resolve_path(roles_path, base_dir=_CONFIGS_ROOT / "roles", subdir_hint="roles")
    p = Path(resolved)
    if not p.exists():
        raise FileNotFoundError(f"roles_path not found: {roles_path} -> {resolved}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"roles json must be a list, got {type(raw)}")

    roles: List[RoleSpec] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"roles[{i}] must be a dict, got {type(item)}")
        name = str(item.get("role") or item.get("name") or "").strip()
        if not name:
            raise ValueError(f"roles[{i}] missing role/name")
        prompt = str(item.get("prompt") or "")
        if not prompt:
            raise ValueError(f"roles[{i}] missing prompt")
        with_answer = bool(item.get("with_answer", False))
        depends = item.get("depends_on") or []
        if depends is None:
            depends = []
        if not isinstance(depends, list):
            raise ValueError(f"roles[{i}].depends_on must be list[str]")
        depends_on = tuple(str(d) for d in depends)
        roles.append(RoleSpec(name=name, prompt=prompt, with_answer=with_answer, depends_on=depends_on))

    topo_sort_roles(roles)
    return roles


def load_task(task_path: str) -> TaskSpec:
    """Load task YAML and embedded roles.

    Expected C3 task YAML:
      experiment_name: str (optional)
      environment: { env_name: str, ... }   # env selection
      mas: { roles_path: str, ... }         # role graph + MAS settings

    Path behavior:
      - task_path can be:
          * relative to configs/tasks/
          * absolute
      - roles_path inside YAML can be:
          * relative to the task YAML file directory (recommended, e.g., ../roles/math/roles_duo.json)
          * absolute
          * legacy absolute (will be mapped to current configs/roles when possible)
    """
    t_resolved = resolve_path(task_path, base_dir=_CONFIGS_ROOT / "tasks", subdir_hint="tasks")
    tpath = Path(t_resolved)
    if not tpath.exists():
        raise FileNotFoundError(f"task_path not found: {task_path} -> {tpath}")

    raw = yaml.safe_load(tpath.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"task yaml root must be a mapping, got {type(raw)}")

    experiment_name = str(raw.get("experiment_name") or raw.get("name") or tpath.stem)

    environment = raw.get("environment") or {}
    if not isinstance(environment, dict):
        raise ValueError("task.environment must be a mapping")
    env_name = str(environment.get("env_name") or environment.get("name") or raw.get("env_name") or "")
    if not env_name:
        raise ValueError("env_name missing in task yaml (environment.env_name)")

    mas = raw.get("mas") or {}
    if not isinstance(mas, dict):
        raise ValueError("task.mas must be a mapping")
    roles_path = mas.get("roles_path") or raw.get("roles_path")
    if not roles_path:
        raise ValueError("roles_path missing in task yaml (mas.roles_path)")

    # Resolve roles_path relative to task yaml directory first.
    roles_resolved = resolve_path(str(roles_path), base_dir=tpath.parent, subdir_hint="roles")
    roles = load_roles(roles_resolved)

    train_datasets = environment.get("train_datasets") or []
    eval_suites = environment.get("eval_suites") or []
    if not isinstance(train_datasets, list):
        raise ValueError("task.environment.train_datasets must be a list")
    if not isinstance(eval_suites, list):
        raise ValueError("task.environment.eval_suites must be a list")

    return TaskSpec(
        repo_root=str(_REPO_ROOT.resolve()),
        task_path=str(tpath.resolve()),
        experiment_name=experiment_name,
        env_name=env_name,
        roles_path=str(Path(roles_resolved).resolve()),
        environment=dict(environment),
        mas=dict(mas),
        roles=tuple(roles),
        train_datasets=tuple(dict(x) for x in train_datasets),
        eval_suites=tuple(dict(x) for x in eval_suites),
    )


# ---------------------------
# CLI (debug utility)
# ---------------------------

def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Dump C3 task/roles spec (repo-root configs loader)")
    parser.add_argument("--task", type=str, required=True, help="Path to C3 task yaml (relative or absolute)")
    args = parser.parse_args()

    spec = load_task(args.task)
    print("repo_root:", str(_REPO_ROOT))
    print("configs_root:", str(_CONFIGS_ROOT))
    print("task_path:", spec.task_path)
    print("experiment_name:", spec.experiment_name)
    print("env_name:", spec.env_name)
    print("roles_path:", spec.roles_path)
    print("roles_topo:", " -> ".join(spec.topo_role_names()))
    print("num_roles:", len(spec.roles))
    for r in topo_sort_roles(spec.roles):
        print(f"  - role={r.name} depends_on={list(r.depends_on)} with_answer={r.with_answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
