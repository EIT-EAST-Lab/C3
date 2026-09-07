"""Contract tests for the public release surface.

These tests encode the mistakes that actually reached a release: preparation
scripts that were never committed, documentation links pointing at moved files,
unanchored ignore patterns that swallowed source directories, and a workflow
whose YAML never parsed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

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
