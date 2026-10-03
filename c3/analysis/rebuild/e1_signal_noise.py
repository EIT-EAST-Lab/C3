# c3/analysis/rebuild/e1_signal_noise.py
"""Signal and noise behind split-half reliability, and the reliability they imply.

At every decision point of a sweep cell (n alternatives, c continuations each),
the estimator has to order the alternatives by the mean return of their
continuations. The variance of those means decomposes into the variance of the
alternatives' true values (the signal the ordering rests on) and the sampling
variance of a c-continuation mean (the noise):

    var_across_alternatives(mean_j)  =  signal  +  noise,     noise = mean_j( s2_j / c_j ),

with s2_j the sample variance of alternative j's continuation returns and c_j
its continuation count. Both terms are averaged over the decision points of a
workflow; a decision point with fewer than two alternatives that carry at least
two continuations is skipped. The signal is the difference of the two averages;
it can be slightly negative when the true spread is at the noise floor, and it
is reported as computed.

The split-half reliability the decomposition implies follows with nothing
fitted. Each half of a split averages c/2 of the c continuations, so its noise
variance is 2N, and the expected correlation between the two halves is

    implied reliability  =  S / (S + 2N).

In the paper (arXiv:2603.06859v3) these are the "Implied" column of Table 1
(S / (S + 2N) for the six workflows) and the lower block of Table 3 (S and N).

    python -m c3.analysis.rebuild.e1_signal_noise \\
        --results 20_data/results/E1 --out signal_noise_summary.json \\
        [--e1-summary 20_data/results/E1/summary.json]

`--results` is the E1 tree `e1_cells.py` writes
(`<workflow>/<model>/sweep_n<n>/buckets.jsonl`). With `--e1-summary`, the
summary `aggregate_e1` wrote, the largest gap between the implied and the
measured per-decision reliability is written as well.

Only the standard library is imported here.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .summary import git_sha, source_path

__all__ = [
    "WORKFLOWS",
    "decompose",
    "implied_reliability",
    "bucket_path",
    "build_keys",
    "main",
]

#: The six workflows of Table 1, in its row order.
WORKFLOWS = ("a2", "a3", "mt4", "branch", "c5", "c10")
DEFAULT_MODEL = "4b"
DEFAULT_N = 4
SCRIPT = "c3.analysis.rebuild.e1_signal_noise"


def _rows(path: str) -> Iterable[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield json.loads(line)


def decompose(rows: Iterable[Mapping[str, Any]]) -> Tuple[float, float, int]:
    """(signal, noise, decision points used) over the bucket rows of one cell.

    Raises ValueError when no decision point has two usable alternatives.
    """
    total = 0.0
    noise = 0.0
    n = 0
    for row in rows:
        alts = [
            [float(r) for r in (cand.get("returns") or [])]
            for cand in (row.get("candidates") or [])
        ]
        alts = [r for r in alts if len(r) >= 2]
        if len(alts) < 2:
            continue
        means = [statistics.mean(r) for r in alts]
        total += statistics.variance(means)
        noise += statistics.mean(statistics.variance(r) / len(r) for r in alts)
        n += 1
    if n == 0:
        raise ValueError("no decision point with two alternatives of two continuations or more")
    return total / n - noise / n, noise / n, n


def implied_reliability(signal: float, noise: float) -> float:
    """S / (S + 2N): the split-half correlation the decomposition predicts."""
    denom = float(signal) + 2.0 * float(noise)
    if not denom > 0.0:
        raise ValueError("S + 2N is not positive (%r)" % denom)
    return float(signal) / denom


def bucket_path(results: str, workflow: str, model: str = DEFAULT_MODEL, n: int = DEFAULT_N) -> str:
    return os.path.join(results, workflow, model, "sweep_n%d" % int(n), "buckets.jsonl")


def _measured_rel(e1_summary: Mapping[str, Any], workflow: str, model: str, n: int) -> Optional[float]:
    keys = e1_summary.get("keys") or {}
    entry = keys.get("E1.per_decision_n%d.%s.%s.rel" % (int(n), workflow, model))
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def build_keys(
    results: str,
    *,
    workflows: Sequence[str] = WORKFLOWS,
    model: str = DEFAULT_MODEL,
    n: int = DEFAULT_N,
    e1_summary: Optional[Mapping[str, Any]] = None,
    stream=None,
) -> Dict[str, Dict[str, Any]]:
    """The summary keys of the decomposition, one block per workflow found.

    A workflow whose cell is missing is skipped with one line on stderr, so a
    partly collected tree still produces the keys it can support.
    """
    out = sys.stderr if stream is None else stream
    keys: Dict[str, Dict[str, Any]] = {}
    devs: List[Tuple[float, str, float, float]] = []
    rel_sources: List[str] = []
    for wf in workflows:
        path = bucket_path(results, wf, model, n)
        if not os.path.isfile(path):
            print("skip %s: no %s" % (wf, source_path(path)), file=out)
            continue
        signal, noise, used = decompose(_rows(path))
        src = [source_path(path)]
        prefix = "E1.signal_noise.%s.%s" % (wf, model)
        keys[prefix + ".signal_var"] = {
            "value": signal, "n": used, "unit": "return^2", "source": src,
            "note": ("variance across the %d alternatives of their continuation-mean returns minus the "
                     "sampling variance of such a mean, averaged over %d decision points" % (int(n), used)),
        }
        keys[prefix + ".noise_var"] = {
            "value": noise, "n": used, "unit": "return^2", "source": src,
            "note": ("sampling variance of a continuation-mean return (within-alternative variance over "
                     "the continuation count), averaged over alternatives and over %d decision points" % used),
        }
        implied = implied_reliability(signal, noise)
        keys["E1.predicted_rel.%s.%s" % (wf, model)] = {
            "value": implied, "n": used, "unit": "correlation", "source": src,
            "note": "S / (S + 2N) with S = %.6f and N = %.6f" % (signal, noise),
        }
        if e1_summary is not None:
            measured = _measured_rel(e1_summary, wf, model, n)
            if measured is None:
                print("skip the deviation of %s: the E1 summary has no per-decision reliability" % wf,
                      file=out)
            else:
                devs.append((abs(implied - measured), wf, implied, measured))
                rel_sources.extend(src)
    if devs and len(devs) == len(workflows):
        worst = max(devs)
        keys["E1.predicted_rel.max_abs_dev"] = {
            "value": worst[0], "n": len(devs), "unit": "correlation", "source": rel_sources,
            "note": ("max over %s of |implied - measured per-decision reliability|; attained at %s: "
                     "|%.4f - %.4f| = %.4f" % (", ".join(w for _, w, _, _ in devs), worst[1], worst[2],
                                                worst[3], worst[0])),
        }
    elif e1_summary is not None:
        print("E1.predicted_rel.max_abs_dev not written: %d of %d workflows have both readings"
              % (len(devs), len(workflows)), file=out)
    return keys


def write_document(out_path: str, keys: Mapping[str, Any], *, results: str) -> Dict[str, Any]:
    doc = {
        "experiment": "E1",
        "results_root": source_path(results),
        "generated_by": {
            "script": SCRIPT,
            "git_sha": git_sha(),
            "when": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "keys": {k: keys[k] for k in sorted(keys)},
    }
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    return doc


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m c3.analysis.rebuild.e1_signal_noise",
        description="Signal, noise and implied split-half reliability of the E1 sweep cells.",
    )
    ap.add_argument("--results", required=True, help="the E1 results directory")
    ap.add_argument("--out", required=True, help="where the summary JSON is written")
    ap.add_argument("--workflows", default=",".join(WORKFLOWS), help="comma separated workflow names")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="model directory name of the cells")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="branching factor of the cells (sweep_n<n>)")
    ap.add_argument("--e1-summary", default="",
                    help="the summary aggregate_e1 wrote; adds E1.predicted_rel.max_abs_dev")
    args = ap.parse_args(argv)

    workflows = [w.strip() for w in args.workflows.split(",") if w.strip()]
    e1_summary = None
    if args.e1_summary:
        with open(args.e1_summary, "r", encoding="utf-8") as fh:
            e1_summary = json.load(fh)
    try:
        keys = build_keys(args.results, workflows=workflows, model=args.model, n=args.n,
                          e1_summary=e1_summary)
    except (OSError, ValueError) as exc:
        print("[e1_signal_noise] %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2
    if not keys:
        print("[e1_signal_noise] no cell found under %s" % args.results, file=sys.stderr)
        return 2
    write_document(args.out, keys, results=args.results)
    for key in sorted(keys):
        print("  %-40s %9.4f  (n=%s)" % (key, keys[key]["value"], keys[key]["n"]))
    print("[e1_signal_noise] wrote %d keys to %s" % (len(keys), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
