"""Contract tests for the public release surface.

These tests encode the mistakes that actually reached a release: preparation
scripts that were never committed, documentation links pointing at moved files,
unanchored ignore patterns that swallowed source directories, a workflow whose
YAML never parsed, and a vendored package that no import could resolve because
it was never committed.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

QUICKSTART_SCRIPTS = (
    "scripts/10_data/prepare_all.sh",
    "scripts/10_data/prepare_math.py",
    "scripts/10_data/prepare_code.py",
    "scripts/20_models/download_models.sh",
)

LINKED_DOCS = (
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "CHANGELOG.md",
)

# Directories that hold generated local output and must never be committed.
GENERATED_OUTPUT_DIRS = {"data", "ckpt", "runs", "wandb", "models", "artifacts"}

_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)")
_HTML_TARGET = re.compile(r"(?:href|src)=\"([^\"]+)\"")


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git is unavailable or this is not a work tree: {exc}")
    return out.stdout


def _tracked_modes() -> Dict[str, str]:
    modes: Dict[str, str] = {}
    for line in _git("ls-files", "-s").splitlines():
        if not line.strip():
            continue
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) >= 1 and path:
            modes[path] = parts[0]
    return modes


def _markdown_files() -> List[Path]:
    files = [REPO_ROOT / name for name in LINKED_DOCS]
    files.extend(sorted((REPO_ROOT / "docs").glob("*.md")))
    return [f for f in files if f.is_file()]


def _all_docs_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in _markdown_files())


def test_quickstart_scripts_are_tracked() -> None:
    modes = _tracked_modes()
    missing = [path for path in QUICKSTART_SCRIPTS if path not in modes]
    assert not missing, f"Quickstart scripts are not tracked by git: {missing}"


def test_shell_scripts_are_runnable_the_way_the_docs_say() -> None:
    """A non-executable script is fine only if the docs invoke it through bash."""
    modes = _tracked_modes()
    docs = _all_docs_text()

    offenders: List[str] = []
    for path, mode in sorted(modes.items()):
        if not path.startswith("scripts/") or not path.endswith(".sh"):
            continue
        if mode == "100755":
            continue
        if f"bash {path}" in docs:
            continue
        if path.startswith("scripts/_lib/"):
            # Library, sourced rather than executed.
            continue
        offenders.append(path)

    assert not offenders, (
        "These shell scripts are neither executable nor documented as "
        f"`bash <path>`: {offenders}"
    )


def test_every_relative_documentation_link_resolves() -> None:
    broken: List[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        targets: Set[str] = set(_MD_LINK.findall(text)) | set(_HTML_TARGET.findall(text))
        for target in targets:
            if target.startswith(("http://", "https://", "mailto:", "#", "data:")):
                continue
            cleaned = target.split("#", 1)[0].split("?", 1)[0].strip()
            if not cleaned:
                continue
            resolved = (path.parent / cleaned).resolve()
            if not resolved.exists():
                broken.append(f"{path.relative_to(REPO_ROOT).as_posix()} -> {target}")

    assert not broken, "Documentation links point at paths that do not exist:\n" + "\n".join(broken)


def test_prose_is_english_without_em_or_en_dashes() -> None:
    script = REPO_ROOT / "scripts" / "90_audit" / "scan_prose.py"
    assert script.is_file(), f"missing {script}"
    result = subprocess.run(
        [sys.executable, "-B", str(script), "--root", str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_generated_output_ignore_patterns_are_root_anchored() -> None:
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    unanchored: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!"):
            continue
        name = line.rstrip("/").lstrip("/")
        if "/" in name:
            continue
        if name in GENERATED_OUTPUT_DIRS and not line.startswith("/"):
            unanchored.append(raw)

    assert not unanchored, (
        "Generated-output ignore patterns must be anchored to the repository root, "
        f"otherwise they also hide nested source directories: {unanchored}"
    )


def test_workflows_parse_and_every_step_has_uses_or_run() -> None:
    workflow_dir = REPO_ROOT / ".github" / "workflows"
    workflows = sorted(list(workflow_dir.glob("*.yml")) + list(workflow_dir.glob("*.yaml")))
    assert workflows, "no workflow files found"

    problems: List[str] = []
    for path in workflows:
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            problems.append(f"{rel}: does not parse: {exc}")
            continue
        if not isinstance(doc, dict):
            problems.append(f"{rel}: top level is not a mapping")
            continue
        jobs = doc.get("jobs")
        if not isinstance(jobs, dict) or not jobs:
            problems.append(f"{rel}: has no jobs mapping")
            continue
        for job_name, job in jobs.items():
            if not isinstance(job, dict):
                problems.append(f"{rel}: job {job_name} is not a mapping")
                continue
            steps = job.get("steps")
            if steps is None:
                continue
            if not isinstance(steps, list):
                problems.append(f"{rel}: job {job_name} has a non-list steps value")
                continue
            for index, step in enumerate(steps):
                if not isinstance(step, dict):
                    problems.append(f"{rel}: job {job_name} step {index} is not a mapping")
                    continue
                if "uses" not in step and "run" not in step:
                    problems.append(
                        f"{rel}: job {job_name} step {index} "
                        f"({step.get('name', 'unnamed')}) has neither uses nor run"
                    )

    assert not problems, "Workflow problems:\n" + "\n".join(problems)


# ---------------------------------------------------------------------------
# Intra-repository import resolution
#
# `openrlhf/models/` was imported by nine modules of the vendored subtree and by
# the analysis CLI, yet it was never committed: an unanchored `models/` ignore
# pattern swallowed it. Every fresh clone therefore failed to import the
# training entry point. This test resolves import statements statically, so it
# needs neither the GPU stack nor any third-party package to be installed.
# ---------------------------------------------------------------------------

INTRA_REPO_PACKAGES = ("c3", "openrlhf")


def _tracked_files() -> Set[str]:
    return {line.strip() for line in _git("ls-files").splitlines() if line.strip()}


def _tracked_directories(tracked: Set[str]) -> Set[str]:
    directories: Set[str] = set()
    for path in tracked:
        parts = path.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            directories.add("/".join(parts[:depth]))
    return directories


def _module_name_for(rel_path: str) -> str:
    parts = rel_path.split("/")
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def _intra_repo_imports(rel_path: str, tree: ast.AST) -> Tuple[List[str], List[str]]:
    """Return (absolute module names imported, structural problems) for one file."""
    module = _module_name_for(rel_path)
    package = module if rel_path.endswith("/__init__.py") else module.rpartition(".")[0]

    names: List[str] = []
    problems: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
            continue

        if not isinstance(node, ast.ImportFrom):
            continue

        if not node.level:
            if node.module:
                names.append(node.module)
            continue

        base = package
        for _ in range(node.level - 1):
            base = base.rpartition(".")[0]
        if not base:
            problems.append(
                f"{rel_path}:{node.lineno}: relative import climbs above the "
                f"top-level package (level={node.level}, module={node.module!r})"
            )
            continue

        head = f"{base}.{node.module}" if node.module else base
        names.append(head)
        if node.module is None:
            # `from . import x` always names submodules, never symbols.
            names.extend(f"{head}.{alias.name}" for alias in node.names if alias.name != "*")

    return names, problems


def _resolves_to_tracked_module(name: str, tracked: Set[str], directories: Set[str]) -> bool:
    stem = name.replace(".", "/")
    return f"{stem}.py" in tracked or f"{stem}/__init__.py" in tracked or stem in directories


def test_every_intra_repository_import_resolves_to_a_tracked_file() -> None:
    """Every `c3.*` and `openrlhf.*` import must resolve inside the release.

    Only the module being imported from is checked. `from pkg import name`
    cannot tell a submodule from a symbol without importing, and importing
    would require the GPU training stack; the module head is enough to catch a
    package that is missing from the release.
    """
    tracked = _tracked_files()
    directories = _tracked_directories(tracked)
    sources = sorted(
        path
        for path in tracked
        if path.endswith(".py") and path.split("/", 1)[0] in INTRA_REPO_PACKAGES
    )
    assert sources, "no tracked python sources found under c3/ or openrlhf/"

    problems: List[str] = []
    unresolved: List[str] = []

    for rel_path in sources:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=rel_path)
        except SyntaxError as exc:
            problems.append(f"{rel_path}: does not parse: {exc}")
            continue

        names, file_problems = _intra_repo_imports(rel_path, tree)
        problems.extend(file_problems)
        for name in names:
            if name.split(".", 1)[0] not in INTRA_REPO_PACKAGES:
                continue
            if not _resolves_to_tracked_module(name, tracked, directories):
                unresolved.append(f"{rel_path} -> {name}")

    assert not problems, "Import statements could not be analyzed:\n" + "\n".join(problems)
    assert not unresolved, (
        "These imports name a module of this repository that git does not "
        "track, so a fresh clone cannot import them:\n" + "\n".join(sorted(set(unresolved)))
    )
