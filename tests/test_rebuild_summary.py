# tests/test_rebuild_summary.py
"""One summary.json implementation for the whole rebuild package.

Driver ruling B13 (2026-09-15) merged the two writers this package used to have:
`c3.analysis.rebuild.summary` holds the document, and the E1 family's
`SummaryBuilder` is a collecting wrapper around it. These tests check that the
two paths agree on the top-level shape and on the three conventions that used to
differ (git sha null, timestamp ending in Z, empty note left out), and that both
refuse an unknown key and a verdict key without stopping.

    python -m pytest tests/test_rebuild_summary.py -q
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess

import pytest

from c3.analysis.rebuild import aggregate_e1, influence, summary

KEY = "E1.sweep_n.a2.4b.n4.rel"
OTHER = "E1.sweep_n.a2.4b.n8.rel"
VERDICT = "E1.reliability.budget_not_depth"


def manifest():
    """Two ordinary keys and one verdict key, shaped like the paper's manifest."""
    return {
        KEY: {"value": None, "fmt": "{:.2f}", "unit": "spearman", "verdict": None,
              "source": None, "n": None, "experiment": "E1", "note": ""},
        OTHER: {"value": None, "fmt": "{:.2f}", "unit": "spearman", "verdict": None,
                "source": None, "n": None, "experiment": "E1", "note": ""},
        VERDICT: {"value": None, "fmt": "{}", "unit": "verdict", "verdict": None,
                  "source": None, "n": None, "experiment": "E1", "note": ""},
    }


def entries():
    """The same two keys as the collecting path stores them."""
    return {
        KEY: {"value": 0.87, "n": 96, "source": ["20_data/results/E1/a2/4b/sweep_n4/"
                                                 "buckets.jsonl"],
              "note": "excluded 4/100 groups"},
        OTHER: {"value": 0.91, "n": 98, "source": ["20_data/results/E1/a2/4b/sweep_n8/"
                                                   "buckets.jsonl"],
                "note": ""},
    }


def both_documents(stream=None):
    """The same content through both paths: (collected, direct)."""
    man = manifest()
    builder = aggregate_e1.SummaryBuilder("E1", "script", man, stream=stream)
    for key, entry in entries().items():
        assert builder.add(key, entry["value"], n=entry["n"], source=entry["source"],
                           note=entry["note"])
    direct = summary.build_summary("E1", "script", entries(), man, stream=stream)
    return builder.payload(), direct


# -----------------------------------------------------------------------------
# 1. one implementation
# -----------------------------------------------------------------------------


def test_the_moved_names_are_the_same_objects():
    """The old names still work because they are imported, not copied."""
    for name in ("build_summary", "write_summary", "load_manifest", "source_path",
                 "git_sha", "format_p"):
        assert getattr(influence, name) is getattr(summary, name)
    assert aggregate_e1.load_manifest is summary.load_manifest
    assert aggregate_e1.git_sha is summary.git_sha


def test_both_paths_write_the_same_document():
    collected, direct = both_documents()
    assert collected["keys"] == direct["keys"]
    assert collected["experiment"] == direct["experiment"] == "E1"
    assert set(collected["generated_by"]) == set(direct["generated_by"]) == {
        "script", "git_sha", "when"}
    assert list(collected["keys"]) == sorted(entries())
    assert collected["keys"][KEY] == {
        "value": 0.87, "n": 96, "unit": "spearman",
        "source": ["20_data/results/E1/a2/4b/sweep_n4/buckets.jsonl"],
        "note": "excluded 4/100 groups"}


# -----------------------------------------------------------------------------
# 2. the three conventions the ruling fixed
# -----------------------------------------------------------------------------


def test_a_git_sha_that_cannot_be_read_is_null(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no git here")

    monkeypatch.setattr(subprocess, "run", boom)
    assert summary.git_sha() is None
    collected, direct = both_documents()
    assert collected["generated_by"]["git_sha"] is None
    assert direct["generated_by"]["git_sha"] is None


def test_a_git_sha_that_can_be_read_is_the_commit():
    sha = summary.git_sha()
    if sha is None:
        pytest.skip("no git available in this environment")
    assert len(sha) == 40 and all(ch in "0123456789abcdef" for ch in sha)
    collected, direct = both_documents()
    assert collected["generated_by"]["git_sha"] == direct["generated_by"]["git_sha"] == sha


def test_the_timestamp_ends_in_z():
    collected, direct = both_documents()
    for doc in (collected, direct):
        when = doc["generated_by"]["when"]
        assert when.endswith("Z")
        assert "+00:00" not in when
        parsed = datetime.datetime.strptime(when, "%Y-%m-%dT%H:%M:%SZ")
        assert parsed.year >= 2026


def test_an_empty_note_is_left_out():
    collected, direct = both_documents()
    for doc in (collected, direct):
        assert "note" not in doc["keys"][OTHER]
        assert doc["keys"][KEY]["note"] == "excluded 4/100 groups"


# -----------------------------------------------------------------------------
# 3. the refusals (contract revision 2, ruling B9)
# -----------------------------------------------------------------------------


def test_both_paths_refuse_an_unknown_key(capsys):
    man = manifest()
    builder = aggregate_e1.SummaryBuilder("E1", "script", man)
    assert builder.add("E1.not.a.key", 0.5, n=3, source="x") is False
    assert builder.refused == ["E1.not.a.key"]
    assert "E1.not.a.key" not in builder.payload()["keys"]

    direct = summary.build_summary("E1", "script",
                                   {"E1.not.a.key": {"value": 0.5, "n": 3, "source": ["x"]},
                                    KEY: {"value": 0.87, "n": 96, "source": ["x"]}}, man)
    assert set(direct["keys"]) == {KEY}
    err = capsys.readouterr().err
    assert err.count("skip E1.not.a.key: not a manifest key") == 2


def test_both_paths_refuse_a_verdict_key(capsys):
    man = manifest()
    builder = aggregate_e1.SummaryBuilder("E1", "script", man)
    assert builder.add(VERDICT, "confirmed", n=5, source="x") is False
    assert VERDICT not in builder.payload()["keys"]

    direct = summary.build_summary("E1", "script",
                                   {VERDICT: {"value": "confirmed", "n": 5, "source": ["x"]},
                                    KEY: {"value": 0.87, "n": 96, "source": ["x"]}}, man)
    assert set(direct["keys"]) == {KEY}
    err = capsys.readouterr().err
    assert err.count("skip %s: verdict key, the driver decides it" % VERDICT) == 2


def test_key_refusal_is_the_one_rule():
    man = manifest()
    assert summary.key_refusal(KEY, man) is None
    assert summary.key_refusal("E1.not.a.key", man) == "not a manifest key"
    assert summary.key_refusal(VERDICT, man) == "verdict key, the driver decides it"


def test_a_refusal_does_not_stop_the_aggregation(tmp_path, capsys):
    """A refused key costs one line on stderr; the file is still written and the
    caller still returns zero."""
    man = manifest()
    builder = aggregate_e1.SummaryBuilder("E1", "script", man)
    assert not builder.add("E1.not.a.key", 1.0, n=1, source="x")
    assert builder.add(KEY, 0.87, n=96, source="x", note="kept")
    path = builder.write(os.path.join(str(tmp_path), "deep", "summary.json"))
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert set(doc["keys"]) == {KEY}
    assert capsys.readouterr().err.count("skip ") == 1


# -----------------------------------------------------------------------------
# 4. the written file
# -----------------------------------------------------------------------------


def test_the_two_writers_produce_the_same_file_shape(tmp_path):
    man = manifest()
    builder = aggregate_e1.SummaryBuilder("E1", "script", man)
    for key, entry in entries().items():
        builder.add(key, entry["value"], n=entry["n"], source=entry["source"],
                    note=entry["note"])
    collected_path = os.path.join(str(tmp_path), "collected", "summary.json")
    builder.write(collected_path)
    direct_path = os.path.join(str(tmp_path), "direct", "summary.json")
    summary.write_summary(direct_path, "E1", "script", entries(), man)

    raw = [open(p, "rb").read() for p in (collected_path, direct_path)]
    for blob in raw:
        assert b"\r" not in blob                 # LF on every platform
        assert blob.endswith(b"\n")
    docs = [json.loads(b.decode("utf-8")) for b in raw]
    assert docs[0]["keys"] == docs[1]["keys"]
    assert docs[0]["experiment"] == docs[1]["experiment"]


def test_load_manifest_reads_the_flat_file(tmp_path):
    path = os.path.join(str(tmp_path), "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest(), fh)
    assert summary.load_manifest(path)[KEY]["unit"] == "spearman"
    broken = os.path.join(str(tmp_path), "broken.json")
    with open(broken, "w", encoding="utf-8") as fh:
        json.dump([KEY], fh)
    with pytest.raises(ValueError):
        summary.load_manifest(broken)


def test_source_path_cuts_at_the_data_root(tmp_path):
    inside = os.path.join(str(tmp_path), "20_data", "results", "E1", "buckets.jsonl")
    assert summary.source_path(inside) == "20_data/results/E1/buckets.jsonl"
    assert summary.source_path("some/other/place.jsonl") == "some/other/place.jsonl"


# -----------------------------------------------------------------------------
# 5. the source shape, on both paths (WP-R14 item 5)
#
# The E1 family joins a results root with a cell's sub-path; the other family
# passes a whole path. Before this the E1 join kept the root as the caller
# spelled it, so aggregating by absolute path wrote the operator's own machine
# path into the summary of every cell.
# -----------------------------------------------------------------------------


def test_the_two_source_functions_agree_on_absolute_and_relative_input(tmp_path):
    parts = ("E1", "a2", "4b", "sweep_n4", "buckets.jsonl")
    want = "20_data/results/" + "/".join(parts)

    absolute = os.path.join(str(tmp_path), "20_data", "results")
    assert aggregate_e1.source_path(absolute, *parts) == want
    assert aggregate_e1.source_path(absolute.replace("\\", "/"), *parts) == want
    assert aggregate_e1.source_path("20_data/results", *parts) == want
    assert aggregate_e1.source_path("./20_data/results", *parts) == want
    # the whole-path function, handed the same file, says the same thing
    assert summary.source_path(os.path.join(absolute, *parts)) == want


def test_the_join_keeps_a_root_that_has_no_data_segment(tmp_path):
    """Outside a 20_data tree there is nothing to cut at, so the path stays as
    spelled. The unit tests of the aggregation live in such a tree."""
    root = os.path.join(str(tmp_path), "elsewhere")
    joined = aggregate_e1.source_path(root, "E1", "buckets.jsonl")
    assert joined == root.replace("\\", "/") + "/E1/buckets.jsonl"


def test_the_last_data_segment_is_the_one_cut_at(tmp_path):
    nested = os.path.join(str(tmp_path), "20_data", "mirror", "20_data", "results")
    assert aggregate_e1.source_path(nested, "E1", "buckets.jsonl") == \
        "20_data/results/E1/buckets.jsonl"


# -----------------------------------------------------------------------------
# 6. the extra top-level fields (WP-R14 item 4)
# -----------------------------------------------------------------------------


def test_extra_fields_sit_beside_experiment_and_cannot_overwrite_the_shape():
    man = manifest()
    doc = summary.build_summary("E1", "script", entries(), man,
                                extra={"results_root": "20_data/results/E1",
                                       "key_prefix": "E1"})
    assert list(doc) == ["experiment", "results_root", "key_prefix", "generated_by", "keys"]
    assert doc["results_root"] == "20_data/results/E1"

    for reserved in summary.RESERVED_TOP_LEVEL:
        with pytest.raises(ValueError):
            summary.build_summary("E1", "script", entries(), man, extra={reserved: "x"})
