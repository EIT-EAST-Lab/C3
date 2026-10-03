# tests/unit/test_eval_method_labels.py
"""What a method LABEL is allowed to be, along the evaluation path of record.

The path is `scripts/70_rebuild/final_eval.py`
-> `scripts/50_eval/paper_main_results.sh one`, and the label travels the whole
way: the caller passes the method name as `--method`,
the driver passes it through, and the runner turns it into the trainer's
`--marl_algorithm`.

A supplementary row names a CHECKPOINT of a method rather than a run's final
model (`magrpo_s250`, MAGRPO's step 250 exported under its own method name), so
the label gains a suffix the algorithm dispatch never saw before. What is pinned
here is that the suffix changes the RECORD and not the evaluation: the same
weights under `MAGRPO` and under `MAGRPO_S250` are evaluated with the same
algorithm setting, and a label that is not a method at all still falls through
to `auto` rather than being read as one.

    python -m pytest tests/unit/test_eval_method_labels.py -q
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "50_eval" / "paper_main_results.sh"
SCRIPT_DIR = REPO_ROOT / "scripts" / "70_rebuild"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import final_eval  # noqa: E402

FUNCTION = "_infer_alg_from_method"


def extract_bash_function(path: Path, name: str) -> str:
    """The text of one shell function, from its header to the `}` in column one.

    The runner executes a command at the end of the file, so it cannot be
    sourced to get at one function; the function is lifted out and sourced on
    its own instead.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("%s()" % name):
            start = index
            break
    assert start is not None, "%s does not define %s" % (path, name)
    for end in range(start + 1, len(lines)):
        if lines[end] == "}":
            return "\n".join(lines[start:end + 1]) + "\n"
    raise AssertionError("%s in %s is not closed in column one" % (name, path))


def infer_alg(label: str, tmp_path) -> str:
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("no bash on this machine")
    script = tmp_path / "fn.sh"
    script.write_text(extract_bash_function(RUNNER, FUNCTION)
                      + '%s "$1"\n' % FUNCTION, encoding="utf-8")
    out = subprocess.run([bash, str(script), label], capture_output=True, text=True,
                         check=True)
    return out.stdout.strip()


# -----------------------------------------------------------------------------
# the label -> algorithm dispatch
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("label,expected", [
    ("SFT", "none"),
    ("MAPPO", "mappo"),
    ("MAGRPO", "magrpo"),
    ("C3", "c3"),
    ("c3", "c3"),
])
def test_the_four_method_labels_of_this_round_dispatch_as_they_did(label, expected, tmp_path):
    assert infer_alg(label, tmp_path) == expected


@pytest.mark.parametrize("label", ["MAGRPO_S250", "magrpo_s250", "MAGRPO_S1", "magrpo_s312"])
def test_a_checkpoint_label_is_evaluated_as_the_method_it_is_a_checkpoint_of(label, tmp_path):
    """The whole point of the change: `magrpo_s250` is MAGRPO's weights, so it is
    evaluated under MAGRPO's algorithm setting. Before this it fell through to
    `auto`, which is a different setting than the label claims."""
    assert infer_alg(label, tmp_path) == "magrpo"


@pytest.mark.parametrize("label,expected", [
    ("C3_S250", "c3"),
    ("MAPPO_S100", "mappo"),
    ("SFT_S0", "none"),
])
def test_the_suffix_is_stripped_for_every_method_and_not_just_magrpo(label, expected, tmp_path):
    assert infer_alg(label, tmp_path) == expected


@pytest.mark.parametrize("label", ["CCPO", "PPO1A", "NONSENSE", "MAGRPO_STEP250",
                                   "MAGRPO_S", "MAGRPOS250"])
def test_a_label_that_is_not_a_method_still_falls_through(label, tmp_path):
    """The other half. Only `_s<digits>` is a checkpoint suffix, so a label that
    merely looks like one is not quietly read as a method it is not."""
    assert infer_alg(label, tmp_path) == "auto"


# -----------------------------------------------------------------------------
# the driver in the middle passes the label through untouched
# -----------------------------------------------------------------------------


def test_the_driver_has_no_whitelist_of_method_labels():
    """`final_eval.py` records the label and hands it to the runner; a new method
    name needs no change there, and this is what says so."""
    parser = final_eval.build_parser() if hasattr(final_eval, "build_parser") else None
    if parser is None:
        pytest.skip("final_eval.py builds its parser inside main()")
    action = [a for a in parser._actions if "--method" in a.option_strings]
    assert action and action[0].choices is None


def test_the_default_method_is_still_the_untrained_start():
    assert final_eval.DEFAULT_METHOD == "SFT"


def test_the_label_reaches_the_runner_and_the_record_unchanged():
    """Both places the label lands: the `--method` argument of the runner command
    and the `method` field of the record. Read off the source, because the two
    are the contract between this driver and the aggregation."""
    text = (SCRIPT_DIR / "final_eval.py").read_text(encoding="utf-8")
    assert '"--method",\n            args.method,' in text
    assert '"method": args.method,' in text
