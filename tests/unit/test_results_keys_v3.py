"""The key table of the paper, configs/results_keys_v3.json.

It lists the result keys arXiv:2603.06859v3 prints, each with its unit and the
tables, figures or sections it appears in, and no value. It has the manifest's
shape, so every aggregation accepts it as `--manifest`, and their help says so.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from c3.analysis.rebuild import aggregate_e1, aggregate_e3a, aggregate_e5
from c3.analysis.rebuild.summary import MANIFEST_HELP, RESULTS_KEYS_V3, key_refusal, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = REPO_ROOT / RESULTS_KEYS_V3
PLACE = re.compile(r"^(Abstract|Section [0-9A-Z]+(\.[0-9]+)?|Table [0-9]+|Figure [0-9]+)$")


def test_the_key_table_carries_names_units_and_places_only() -> None:
    table = load_manifest(str(SCHEMA))
    assert len(table) > 200
    for key, entry in table.items():
        assert set(entry) == {"unit", "paper"}, key
        assert isinstance(entry["unit"], str) and entry["unit"], key
        assert entry["paper"] and all(PLACE.match(place) for place in entry["paper"]), (key, entry["paper"])


def test_the_key_table_covers_the_trained_tables() -> None:
    table = load_manifest(str(SCHEMA))
    for arm, paper in (("a2_4b", "Table 2"), ("a3_4b", "Table 8")):
        for bench in ("math500", "aime25", "cmath", "gsm8k", "avg3"):
            for method in ("mappo", "magrpo", "c3"):
                for stat in ("mean", "std"):
                    key = "E2main.%s.%s.%s.%s" % (arm, bench, method, stat)
                    assert paper in table[key]["paper"], key
    for bench in ("math500", "aime25", "cmath", "gsm8k"):
        for stat in ("diff", "ci_lo", "ci_hi", "p", "p_holm"):
            assert "Table 10" in table["E2main.a2_4b.%s.c3_minus_best.%s" % (bench, stat)]["paper"]
    assert any("Table 7" in entry["paper"] for key, entry in table.items() if key.startswith("E2.a2_4b."))
    assert any("Table 1" in entry["paper"] for key, entry in table.items() if key.startswith("E1."))


def test_the_key_table_has_no_training_token_key_for_table_8() -> None:
    table = load_manifest(str(SCHEMA))
    assert not [key for key, entry in table.items()
                if "Table 8" in entry["paper"] and "token" in key + entry["unit"]]
    assert not [key for key in table if key.startswith("E2main.a3_4b.tokens")]
    assert not [key for key in table if "jan" in key.lower()]


def test_the_key_table_is_a_manifest_the_aggregations_accept() -> None:
    table = load_manifest(str(SCHEMA))
    assert key_refusal("E2main.a2_4b.math500.c3.mean", table) is None
    assert key_refusal("E2main.a2_4b.math500.c3.not_a_key", table) == "not a manifest key"
    text = SCHEMA.read_text(encoding="utf-8")
    assert "\r" not in text
    json.loads(text)


def test_every_manifest_help_points_at_the_key_table(capsys) -> None:
    assert RESULTS_KEYS_V3 in MANIFEST_HELP
    for module in (aggregate_e1, aggregate_e3a, aggregate_e5):
        try:
            module.main(["--help"])
        except SystemExit:
            pass
        out = " ".join(capsys.readouterr().out.split())
        assert RESULTS_KEYS_V3 in out, module.__name__
