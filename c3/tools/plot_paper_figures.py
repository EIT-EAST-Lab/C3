#!/usr/bin/env python3
# c3/tools/plot_paper_figures.py
#
# Single-column, paper-ready *compact row* bar plots for C3/C3 analysis.
#
# This revision (responding to feedback):
#  1) No method names on x-axis; use a compact legend in the upper-right (figure-level).
#  2) (a)(b)(c) panel labels OFF by default.
#  3) Increase inter-panel spacing to avoid any overlap in 1x3 row.
#  4) Metric units/meaning stay on the y-axis; each panel keeps a centered title.
#  5) Fonts slightly smaller but still crisp at single-column size.
#
# Usage:
#   # Paper Fig2 (official): mechanism diagnostics
#   python plot_paper_figures.py mechanism --out_dir /tmp/figs --use_dummy
#
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected dict json")
    return obj


# -----------------------------------------------------------------------------
# Palettes (CVD-friendly + paper-friendly)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    name: str
    mapping: Dict[str, str]     # fixed label -> hex
    cycle: List[str]            # fallback colors
    mono_fill: str = "#D9D9D9"
    mono_edge: str = "#222222"
    mono_hatches: Tuple[str, ...] = ("", "///", "xx", "\\\\", "oo", "..", "--")


def _palette_okabe_ito() -> Palette:
    colors = {
        "orange": "#E69F00",
        "sky": "#56B4E9",
        "green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "purple": "#CC79A7",
        "grey": "#999999",
    }
    mapping = {
        "SFT": colors["grey"],
        "MAPPO": colors["blue"],
        "MAGRPO": colors["orange"],
        "C3": colors["green"],
        "No replay": colors["vermillion"],
        "No LOO": colors["purple"],
    }
    cycle = [
        colors["blue"], colors["orange"], colors["green"],
        colors["vermillion"], colors["purple"], colors["sky"], colors["yellow"], colors["grey"],
    ]
    return Palette(name="okabe-ito", mapping=mapping, cycle=cycle)


def _palette_tol_light() -> Palette:
    # Paul Tol "Light"
    cycle = ["#77AADD", "#99DDFF", "#44BB99", "#BBCC33", "#AAAA00", "#EEDD88", "#EE8866", "#FFAABB", "#DDDDDD"]
    mapping = {
        "SFT": "#888888",
        "MAPPO": "#77AADD",
        "MAGRPO": "#EE8866",
        "C3": "#44BB99",
        "No replay": "#EE8866",
        "No LOO": "#FFAABB",
    }
    return Palette(name="tol-light", mapping=mapping, cycle=cycle)


def _palette_tol_muted() -> Palette:
    # Paul Tol "Muted"
    cycle = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499", "#DDDDDD"]
    mapping = {
        "SFT": "#777777",
        "MAPPO": "#332288",
        "MAGRPO": "#DDCC77",
        "C3": "#117733",
        "No replay": "#CC6677",
        "No LOO": "#AA4499",
    }
    return Palette(name="tol-muted", mapping=mapping, cycle=cycle)


def _palette_brewer_set2() -> Palette:
    cycle = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"]
    mapping = {
        "SFT": "#777777",
        "MAPPO": "#8DA0CB",
        "MAGRPO": "#FC8D62",
        "C3": "#66C2A5",
        "No replay": "#FC8D62",
        "No LOO": "#E78AC3",
    }
    return Palette(name="brewer-set2", mapping=mapping, cycle=cycle)


def _get_palette(name: str) -> Palette:
    key = (name or "tol-light").lower()
    if key in {"tol-light", "tol_light", "light"}:
        return _palette_tol_light()
    if key in {"tol-muted", "tol_muted", "muted", "tol"}:
        return _palette_tol_muted()
    if key in {"brewer-set2", "brewer_set2", "set2"}:
        return _palette_brewer_set2()
    if key in {"okabe-ito", "okabe_ito", "okabe"}:
        return _palette_okabe_ito()
    raise ValueError(f"Unknown palette '{name}'. Choose: tol-light, tol-muted, brewer-set2, okabe-ito")


# -----------------------------------------------------------------------------
# Styling helpers
# -----------------------------------------------------------------------------


def _mix_with_white(hex_color: str, amount: float) -> str:
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(hex_color)
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return mcolors.to_hex((r, g, b), keep_alpha=False)


def _mix_with_black(hex_color: str, amount: float) -> str:
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(hex_color)
    r = r * (1.0 - amount)
    g = g * (1.0 - amount)
    b = b * (1.0 - amount)
    return mcolors.to_hex((r, g, b), keep_alpha=False)


def _panel_letters(i: int) -> str:
    return f"({chr(ord('a') + i)})"


def _preferred_method_order() -> List[str]:
    return ["SFT", "MAPPO", "MAGRPO", "C3", "No replay", "No LOO"]


def _unique_preserve(seq: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _build_color_map(all_labels: Sequence[str], *, pal: Palette) -> Dict[str, str]:
    color_map: Dict[str, str] = dict(pal.mapping)
    unknown = [l for l in all_labels if l not in color_map]
    unknown = _unique_preserve(unknown)
    for i, lbl in enumerate(unknown):
        if pal.cycle:
            color_map[lbl] = pal.cycle[i % len(pal.cycle)]
        else:
            color_map[lbl] = "#777777"
    return color_map


@dataclass(frozen=True)
class PaperStyle:
    medium: str = "paper"     # paper|slides
    theme: str = "color"      # color|mono

    palette: str = "tol-light"
    fill_lighten: float = 0.06
    edge_darken: float = 0.30
    best_edge_darken: float = 0.58

    grid_alpha: float = 0.10
    grid_linewidth: float = 0.55

    fig_width_in: float = 3.25
    fig_height_in: float = 1.72

    layout: str = "row"       # row|stack
    wspace: float = 0.70
    hspace: float = 0.78

    grid_left: float = 0.10
    grid_right: float = 0.99
    grid_bottom: float = 0.22
    grid_top: float = 0.50

    # Typography
    base_fontsize: float = 7.9
    tick_fontsize: float = 7.2
    title_fontsize: float = 7.9
    label_fontsize: float = 7.6
    value_fontsize: float = 4.8

    panel_titles: bool = True
    title_pad: float = 2.1

    # Panel labels (off by default)
    panel_labels: bool = False
    panel_label_fontsize: float = 7.6
    panel_label_weight: str = "bold"
    panel_label_x: float = -0.18
    panel_label_y: float = 1.08

    # Legend (figure-level)
    show_legend: bool = True
    legend_fontsize: float = 7.2
    legend_frame: bool = False
    legend_ncol: int = 2
    legend_loc: str = "upper center"
    legend_bbox_x: float = 0.5
    legend_bbox_y: float = 0.9
    legend_handlelength: float = 1.1
    legend_handletextpad: float = 0.45
    legend_columnspacing: float = 0.85
    legend_labelspacing: float = 0.25

    axis_linewidth: float = 0.8
    tick_width: float = 0.8
    tick_size: float = 3.0
    x_tick_pad: float = 1.1
    y_tick_pad: float = 1.1

    category_step: float = 0.80
    bar_width: float = 0.46
    bar_edge_lw: float = 0.85
    best_edge_lw: float = 1.15
    x_pad_steps: float = 0.55

    show_values: str = "off"        # off|auto|on
    show_best_star: bool = False
    save_pad_inches: float = 0.020


def _apply_mpl_style(*, mpl: Any, style: PaperStyle, use_tex: bool, corefonts: bool) -> None:
    serif_stack = ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "STIXGeneral", "DejaVu Serif"]
    rc = {
        "font.family": "serif",
        "font.serif": serif_stack,
        "font.size": style.base_fontsize,
        "axes.titlesize": style.title_fontsize,
        "axes.labelsize": style.label_fontsize,
        "xtick.labelsize": style.tick_fontsize,
        "ytick.labelsize": style.tick_fontsize,
        "legend.fontsize": style.legend_fontsize,
        "mathtext.fontset": "stix",

        "axes.linewidth": style.axis_linewidth,
        "axes.labelpad": 1.4,
        "axes.titlepad": style.title_pad,
        "xtick.major.width": style.tick_width,
        "ytick.major.width": style.tick_width,
        "xtick.major.size": style.tick_size,
        "ytick.major.size": style.tick_size,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "savefig.bbox": "tight",
        "savefig.pad_inches": style.save_pad_inches,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    if corefonts:
        rc.update({"pdf.use14corefonts": True, "ps.useafm": True})
    if use_tex:
        rc.update({"text.usetex": True, "text.latex.preamble": r"\\usepackage{newtxtext}\\usepackage{newtxmath}"})
    mpl.rcParams.update(rc)
    if style.theme == "mono":
        mpl.rcParams.update({"hatch.linewidth": 0.6})


def _style_axes(ax: Any, *, style: PaperStyle) -> None:
    ax.grid(axis="y", linestyle="-", linewidth=style.grid_linewidth, alpha=style.grid_alpha)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.92)
    ax.spines["bottom"].set_alpha(0.92)

    ax.tick_params(axis="x", pad=style.x_tick_pad)
    ax.tick_params(axis="y", pad=style.y_tick_pad)


def _nice_ylim(values: Sequence[float], *, lower: float = 0.0, headroom: float = 0.12) -> Tuple[float, float]:
    import math
    vmax = max(values) if values else 1.0
    if vmax <= 0:
        return lower, 1.0
    raw = vmax * (1.0 + headroom)
    exp = math.floor(math.log10(raw)) if raw > 0 else 0
    for nice in (1, 2, 2.5, 5, 10):
        upper = nice * (10**exp)
        if upper >= raw:
            return lower, upper
    return lower, 10 * (10**exp)


def _effective_yerr(errors: Optional[List[float]]) -> Optional[List[float]]:
    if not errors:
        return None
    if all(abs(e) <= 1e-12 for e in errors):
        return None
    return errors


def _should_show_values(style: PaperStyle, n: int) -> bool:
    s = style.show_values.lower()
    if s == "on":
        return True
    if s == "off":
        return False
    return n <= 4


def _adaptive_bar_width(style: PaperStyle, n: int) -> float:
    if n <= 3:
        return style.bar_width
    return max(0.40, min(style.bar_width, style.bar_width - 0.03 * (n - 3)))


def _category_positions(style: PaperStyle, n: int) -> List[float]:
    step = style.category_step if n <= 6 else 1.0
    return [i * step for i in range(n)]


def _set_compact_xlim(ax: Any, xs: List[float], *, style: PaperStyle) -> None:
    if not xs:
        return
    step = xs[1] - xs[0] if len(xs) >= 2 else 1.0
    pad = style.x_pad_steps * step
    ax.set_xlim(xs[0] - pad, xs[-1] + pad)


def _add_panel_label(ax: Any, i: int, *, style: PaperStyle) -> None:
    if not style.panel_labels:
        return
    ax.text(
        style.panel_label_x, style.panel_label_y, _panel_letters(i),
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=style.panel_label_fontsize,
        fontweight=style.panel_label_weight,
        color="#111111",
    )


def _annotate_best(ax: Any, x: float, y: float, *, style: PaperStyle) -> None:
    if not style.show_best_star:
        return
    ax.text(x, y, "★", ha="center", va="bottom", fontsize=style.value_fontsize + 0.6, color="#111111")


def _plot_panel(
    ax: Any,
    *,
    panel_index: int,
    title: str,
    labels: List[str],
    values: List[float],
    errors: Optional[List[float]],
    ylabel: str,
    best_mode: str,          # max|min
    fmt_value: str,
    style: PaperStyle,
    pal: Palette,
    color_map: Dict[str, str],
) -> None:
    import matplotlib.ticker as mticker

    _style_axes(ax, style=style)
    _add_panel_label(ax, panel_index, style=style)

    xs = _category_positions(style, len(labels))
    yerr = _effective_yerr(errors)

    # Colors
    if style.theme == "mono":
        base = [pal.mono_fill] * len(labels)
        fill = base
        edge = [pal.mono_edge] * len(labels)
    else:
        base = [color_map[lbl] for lbl in labels]
        fill = [_mix_with_white(c, style.fill_lighten) for c in base]
        edge = [_mix_with_black(c, style.edge_darken) for c in base]

    # Best index
    best_idx = None
    if values:
        if best_mode == "max":
            best_idx = int(max(range(len(values)), key=lambda k: values[k]))
        else:
            best_idx = int(min(range(len(values)), key=lambda k: values[k]))

    # Title (centered)
    if style.panel_titles:
        ax.set_title(title, loc="center", pad=style.title_pad, fontweight="regular", color="#111111")

    # Metric meaning on y-axis (requested)
    ax.set_ylabel(ylabel)

    # No x-axis method names: legend handles decoding
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)

    # Y ticks: few, clean, no offset text clutter
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, min_n_ticks=3))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.yaxis.get_major_formatter().set_scientific(False)  # type: ignore

    # X limits: compact and symmetric padding
    _set_compact_xlim(ax, xs, style=style)

    # Y limits
    headroom = 0.18 if _should_show_values(style, len(labels)) else 0.12
    y0, y1 = _nice_ylim(values, lower=0.0, headroom=headroom)
    ax.set_ylim(y0, y1)

    # Bars
    bw = _adaptive_bar_width(style, len(labels))
    bar_kwargs = dict(
        width=bw,
        color=fill,
        edgecolor=edge if style.theme != "mono" else pal.mono_edge,
        linewidth=style.bar_edge_lw,
        zorder=2,
    )
    if yerr is not None:
        bar_kwargs.update(
            dict(
                yerr=yerr,
                capsize=2.0,
                error_kw={
                    "ecolor": "#222222",
                    "elinewidth": 0.8,
                    "capthick": 0.8,
                },
            )
        )
    bars = ax.bar(xs, values, **bar_kwargs)

    if style.theme == "mono":
        for i, b in enumerate(bars):
            b.set_hatch(pal.mono_hatches[i % len(pal.mono_hatches)])

    # Emphasize best
    if best_idx is not None:
        bars[best_idx].set_linewidth(style.best_edge_lw)
        if style.theme != "mono":
            bars[best_idx].set_edgecolor(_mix_with_black(base[best_idx], style.best_edge_darken))
        bars[best_idx].set_zorder(3)
        _annotate_best(ax, float(xs[best_idx]), float(values[best_idx]), style=style)

    # Optional value labels
    if _should_show_values(style, len(labels)):
        try:
            ax.bar_label(
                bars,
                fmt=fmt_value,
                padding=1.4,
                fontsize=style.value_fontsize,
                fontweight="bold",
            )
        except Exception:
            dy = (y1 - y0) * 0.02
            for b, yi in zip(bars, values):
                ax.text(b.get_x() + b.get_width() / 2, yi + dy, fmt_value % yi,
                        ha="center", va="bottom", fontsize=style.value_fontsize)

    for line in ax.get_lines():
        line.set_linewidth(max(line.get_linewidth(), 0.8))


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BarSeries:
    labels: List[str]
    values: List[float]
    errors: Optional[List[float]] = None

def _reorder_series(series: "BarSeries", order: Sequence[str]) -> "BarSeries":
    """Return a new BarSeries whose bars follow a stable method order.

    This matters more once x-axis labels are removed: consistent left-to-right ordering
    reduces cognitive load even when colors are the primary key (legend).
    """
    if not series.labels:
        return series
    idx = {lbl: i for i, lbl in enumerate(series.labels)}
    ordered_labels: List[str] = []
    ordered_values: List[float] = []
    ordered_errors: Optional[List[float]] = [] if series.errors is not None else None

    # First: in preferred order
    for lbl in order:
        if lbl in idx:
            i = idx[lbl]
            ordered_labels.append(lbl)
            ordered_values.append(series.values[i])
            if ordered_errors is not None and series.errors is not None:
                ordered_errors.append(series.errors[i])

    # Then: any remaining labels, preserving original appearance order
    for lbl in series.labels:
        if lbl in ordered_labels:
            continue
        i = idx[lbl]
        ordered_labels.append(lbl)
        ordered_values.append(series.values[i])
        if ordered_errors is not None and series.errors is not None:
            ordered_errors.append(series.errors[i])

    return BarSeries(labels=ordered_labels, values=ordered_values, errors=ordered_errors)


@dataclass(frozen=True)
class MechanismFigureData:
    fidelity: BarSeries
    variance: BarSeries
    influence: BarSeries
    suite_title: str


@dataclass(frozen=True)
class CalibrationCurve:
    method: str
    x: List[float]
    y: List[float]
    y_low: Optional[List[float]] = None
    y_high: Optional[List[float]] = None


def _dummy_mechanism_data() -> MechanismFigureData:
    return MechanismFigureData(
        suite_title="Qwen3 math (dummy)",
        fidelity=BarSeries(labels=["MAPPO", "MAGRPO", "C3"], values=[0.41, 0.44, 0.63], errors=[0.0, 0.0, 0.0]),
        variance=BarSeries(labels=["MAPPO", "MAGRPO", "C3"], values=[0.192, 0.165, 0.082], errors=[0.0, 0.0, 0.0]),
        influence=BarSeries(labels=["SFT", "MAPPO", "MAGRPO", "C3"], values=[0.021, 0.026, 0.028, 0.041], errors=[0.0, 0.0, 0.0, 0.0]),
    )


# -----------------------------------------------------------------------------
# JSON adapters
# -----------------------------------------------------------------------------


def _series_from_summary(
    summary: Mapping[str, Any],
    *,
    metric_group: str,
    metric_key: str,
    methods: Sequence[str],
) -> BarSeries:
    suite = summary.get("suite", {})
    group = suite.get(metric_group, {})
    labels: List[str] = []
    vals: List[float] = []
    errs: List[float] = []

    for m in methods:
        rec = group.get(m, {}).get(metric_key)
        if not isinstance(rec, dict):
            raise KeyError(f"summary missing {metric_group}.{m}.{metric_key}")
        labels.append(m)
        vals.append(float(rec.get("mean")))
        errs.append(float(rec.get("std")))
    return BarSeries(labels=labels, values=vals, errors=errs)


def load_mechanism_from_summary(summary_json: Path) -> MechanismFigureData:
    summary = _load_json(summary_json)
    meta = summary.get("meta", {})
    suite_name = str(meta.get("suite", "suite"))

    method_order = list(meta.get("method_order") or [])
    method_set = set(method_order)
    credit_methods = [m for m in ["MAPPO", "MAGRPO", "C3"] if m in method_set]
    infl_methods = [m for m in ["SFT", "MAPPO", "MAGRPO", "C3"] if m in method_set]

    fidelity = _series_from_summary(
        summary,
        metric_group="credit",
        metric_key=str(meta.get("fidelity_variant", "fidelity_real")),
        methods=credit_methods,
    )
    variance = _series_from_summary(summary, metric_group="credit", metric_key="var", methods=credit_methods)
    influence = _series_from_summary(summary, metric_group="influence", metric_key="influence", methods=infl_methods)

    return MechanismFigureData(
        suite_title=f"{suite_name} (from summary)",
        fidelity=fidelity,
        variance=variance,
        influence=influence,
    )


def load_mechanism_from_json(mechanism_json: Path) -> MechanismFigureData:
    obj = _load_json(mechanism_json)
    title = str(obj.get("suite_title", "Mechanism"))

    def _bs(k: str) -> BarSeries:
        rec = obj.get(k)
        if not isinstance(rec, dict):
            raise KeyError(f"mechanism json missing '{k}'")
        labels = list(rec.get("labels") or [])
        values = [float(x) for x in (rec.get("values") or [])]
        errors_raw = rec.get("errors")
        errors = [float(x) for x in errors_raw] if isinstance(errors_raw, list) else None
        if len(labels) != len(values):
            raise ValueError(f"{k}: labels/values length mismatch")
        if errors is not None and len(errors) != len(values):
            raise ValueError(f"{k}: errors length mismatch")
        return BarSeries(labels=labels, values=values, errors=errors)

    return MechanismFigureData(
        suite_title=title,
        fidelity=_bs("fidelity"),
        variance=_bs("variance"),
        influence=_bs("influence"),
    )


# -----------------------------------------------------------------------------
# Plotters
# -----------------------------------------------------------------------------


def _make_gridspec(fig: Any, style: PaperStyle, *, n_panels: int = 3):
    if style.layout == "row":
        return fig.add_gridspec(
            1, int(max(1, n_panels)),
            left=style.grid_left, right=style.grid_right,
            bottom=style.grid_bottom, top=style.grid_top,
            wspace=style.wspace,
        )
    return fig.add_gridspec(
        int(max(1, n_panels)), 1,
        left=style.grid_left, right=style.grid_right,
        bottom=style.grid_bottom, top=style.grid_top,
        hspace=style.hspace,
    )


def _mechanism_panels() -> List[Tuple[str, str, str, str]]:
    return [
        ("Fidelity", "max", "%.2f", "Spearman $\\rho$"),
        ("Variance", "min", "%.3f", "Within-context var"),
        ("Influence", "max", "%.3f", "Mutual info (nats)"),
    ]


def _canonical_method_label(s: str) -> str:
    k = str(s).strip()
    lk = k.lower()
    if lk == "c3":
        return "C3"
    if lk == "magrpo":
        return "MAGRPO"
    if lk == "mappo":
        return "MAPPO"
    if lk == "sft":
        return "SFT"
    return k


def _load_calibration_curves_from_csv(
    path: Path,
    *,
    methods: Optional[Sequence[str]] = None,
) -> List[CalibrationCurve]:
    if not path.exists():
        raise FileNotFoundError(f"calibration csv not found: {path}")

    want = None
    if methods:
        want = {_canonical_method_label(x) for x in methods}

    rows_by_method: Dict[str, List[Dict[str, float]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = _canonical_method_label(str(row.get("method", "")).strip())
            if not m:
                continue
            if want is not None and m not in want:
                continue

            try:
                x = float(row.get("a_hat_bin_center", "nan"))
                y = float(row.get("delta_mean", "nan"))
            except Exception:
                continue
            if not (x == x and y == y):
                continue

            rec: Dict[str, float] = {"x": x, "y": y}
            try:
                rec["y_low"] = float(row.get("delta_ci_low", "nan"))
                rec["y_high"] = float(row.get("delta_ci_high", "nan"))
            except Exception:
                pass
            try:
                rec["bin_id"] = float(row.get("bin_id", "nan"))
            except Exception:
                rec["bin_id"] = float("nan")
            rows_by_method.setdefault(m, []).append(rec)

    out: List[CalibrationCurve] = []
    for m, rows in rows_by_method.items():
        rows2 = sorted(rows, key=lambda r: (r.get("bin_id", float("nan")), r["x"]))
        x = [float(r["x"]) for r in rows2]
        y = [float(r["y"]) for r in rows2]
        y_low: List[float] = []
        y_high: List[float] = []
        has_ci = True
        for r in rows2:
            lo = r.get("y_low", float("nan"))
            hi = r.get("y_high", float("nan"))
            if not (lo == lo and hi == hi):
                has_ci = False
                break
            y_low.append(float(lo))
            y_high.append(float(hi))
        out.append(
            CalibrationCurve(
                method=m,
                x=x,
                y=y,
                y_low=y_low if has_ci else None,
                y_high=y_high if has_ci else None,
            )
        )
    return out


def _plot_calibration_panel(
    ax: Any,
    *,
    curves: Sequence[CalibrationCurve],
    style: PaperStyle,
    color_map: Dict[str, str],
) -> None:
    if not curves:
        return

    _style_axes(ax, style=style)
    ax.set_title("Calibration", loc="center", pad=style.title_pad, fontweight="regular", color="#111111")

    all_x: List[float] = []
    all_y: List[float] = []
    for c in curves:
        all_x.extend(c.x)
        all_y.extend(c.y)
        if c.y_low:
            all_y.extend(c.y_low)
        if c.y_high:
            all_y.extend(c.y_high)
    if not all_x or not all_y:
        return
    lo = min(min(all_x), min(all_y))
    hi = max(max(all_x), max(all_y))
    if hi <= lo:
        hi = lo + 1.0

    # Slight symmetric padding keeps points away from borders.
    pad = 0.08 * (hi - lo)
    lo2 = lo - pad
    hi2 = hi + pad

    ax.plot([lo2, hi2], [lo2, hi2], linestyle="--", linewidth=0.9, color="#666666", alpha=0.9, zorder=1)
    for c in curves:
        col = color_map.get(c.method, "#444444")
        if c.y_low is not None and c.y_high is not None and len(c.y_low) == len(c.x) and len(c.y_high) == len(c.x):
            ax.fill_between(c.x, c.y_low, c.y_high, color=col, alpha=0.14, linewidth=0.0, zorder=1.5)
        ax.plot(c.x, c.y, marker="o", markersize=2.8, linewidth=1.2, color=col, label=c.method, zorder=2.0)

    ax.set_xlabel("Pred. A_hat")
    ax.set_ylabel("Empirical DeltaR")
    ax.set_xlim(lo2, hi2)
    ax.set_ylim(lo2, hi2)
    ax.grid(True, alpha=0.15, linewidth=0.45)
    # Use the figure-level legend only to avoid duplicate legends.


def _figure_legend(fig: Any, *, style: PaperStyle, pal: Palette, color_map: Dict[str, str], labels_union: List[str]) -> None:
    if not style.show_legend:
        return
    import matplotlib.patches as mpatches

    preferred = _preferred_method_order()
    ordered = [m for m in preferred if m in labels_union] + [m for m in labels_union if m not in preferred]
    ordered = _unique_preserve(ordered)

    handles = []
    for i, m in enumerate(ordered):
        if style.theme == "mono":
            patch = mpatches.Patch(
                facecolor=pal.mono_fill, edgecolor=pal.mono_edge,
                linewidth=style.bar_edge_lw, hatch=pal.mono_hatches[i % len(pal.mono_hatches)],
                label=m,
            )
        else:
            base = color_map[m]
            patch = mpatches.Patch(
                facecolor=_mix_with_white(base, style.fill_lighten),
                edgecolor=_mix_with_black(base, style.edge_darken),
                linewidth=style.bar_edge_lw,
                label=m,
            )
        handles.append(patch)

    n = len(handles)
    ncol = n  # 强制一行排开（n 个条目 -> n 列）


    fig.legend(
        handles=handles,
        loc=style.legend_loc,
        bbox_to_anchor=(style.legend_bbox_x, style.legend_bbox_y),
        frameon=style.legend_frame,
        ncol=ncol,
        borderaxespad=0.0,
        handlelength=style.legend_handlelength,
        handletextpad=style.legend_handletextpad,
        columnspacing=style.legend_columnspacing,
        labelspacing=style.legend_labelspacing,
    )


def plot_mechanism_figure(
    data: MechanismFigureData,
    *,
    out_dir: Path,
    style: PaperStyle,
    calibration_curves: Optional[Sequence[CalibrationCurve]],
    fmt: Sequence[str],
    dpi: int,
    use_tex: bool,
    corefonts: bool,
) -> List[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    _apply_mpl_style(mpl=mpl, style=style, use_tex=use_tex, corefonts=corefonts)
    pal = _get_palette(style.palette)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []

    has_calib = bool(calibration_curves)
    n_panels = 4 if has_calib else 3
    fig_w = style.fig_width_in * (4.0 / 3.0) if (has_calib and style.layout == "row") else style.fig_width_in
    fig_h = style.fig_height_in * (1.05 if has_calib else 1.0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = _make_gridspec(fig, style, n_panels=n_panels)

    panels = _mechanism_panels()
    series_list = [data.fidelity, data.variance, data.influence]
    pref = _preferred_method_order()
    series_list = [_reorder_series(s, pref) for s in series_list]

    labels_union = _unique_preserve([l for s in series_list for l in s.labels])
    color_map = _build_color_map(labels_union, pal=pal)

    for i, ((title, best_mode, fmt_value, ylabel), series) in enumerate(zip(panels, series_list)):
        ax = fig.add_subplot(gs[0, i] if style.layout == "row" else gs[i, 0])
        _plot_panel(
            ax,
            panel_index=i,
            title=title,
            labels=series.labels,
            values=series.values,
            errors=series.errors,
            ylabel=ylabel,
            best_mode=best_mode,
            fmt_value=fmt_value,
            style=style,
            pal=pal,
            color_map=color_map,
        )
    if has_calib and calibration_curves:
        axc = fig.add_subplot(gs[0, 3] if style.layout == "row" else gs[3, 0])
        _plot_calibration_panel(
            axc,
            curves=calibration_curves,
            style=style,
            color_map=color_map,
        )

    _figure_legend(fig, style=style, pal=pal, color_map=color_map, labels_union=labels_union)

    if style.medium == "slides":
        fig.suptitle(data.suite_title, y=0.995, fontsize=style.title_fontsize)

    # Paper naming: mechanism diagnostics corresponds to Fig2.
    stem = out_dir / "fig2_mechanism"
    for ext in fmt:
        p = stem.with_suffix("." + ext)
        if ext.lower() == "png":
            fig.savefig(p, dpi=dpi, facecolor="white")
        else:
            fig.savefig(p, facecolor="white")
        out_paths.append(p)

    plt.close(fig)
    return out_paths


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="plot_paper_figures",
        description="Single-column, paper-ready bar figures for C3/C3 analysis (dummy or JSON-backed).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_style_args(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--theme", type=str, default="color", choices=["color", "mono"])
        pp.add_argument("--palette", type=str, default="tol-light",
                        choices=["tol-light", "tol-muted", "brewer-set2", "okabe-ito"])
        pp.add_argument("--medium", type=str, default="paper", choices=["paper", "slides"])
        pp.add_argument("--layout", type=str, default="row", choices=["row", "stack"],
                        help="single-column default: row (1x3). stack is 3x1.")
        pp.add_argument("--show_values", type=str, default="auto", choices=["off", "auto", "on"])
        pp.add_argument("--best_star", action="store_true", help="mark best with a small star (usually off for paper).")

        pp.add_argument("--panel_titles", action="store_true", help="force show panel titles.")
        pp.add_argument("--no_panel_titles", action="store_true", help="hide panel titles for ultra-clean figs.")
        pp.add_argument("--panel_labels", action="store_true", help="enable (a)(b)(c) labels (default off).")

        pp.add_argument("--no_legend", action="store_true", help="disable figure legend (not recommended).")

        pp.add_argument("--fig_width_in", type=float, default=3.25)
        pp.add_argument("--fig_height_in", type=float, default=1.72)

        pp.add_argument("--hspace", type=float, default=0.78)
        pp.add_argument("--wspace", type=float, default=0.65)
        pp.add_argument("--grid_left", type=float, default=0.10)
        pp.add_argument("--grid_right", type=float, default=0.99)
        pp.add_argument("--grid_bottom", type=float, default=0.22)
        pp.add_argument("--grid_top", type=float, default=0.70)

        pp.add_argument("--bar_width", type=float, default=0.46)
        pp.add_argument("--category_step", type=float, default=0.80,
                        help="distance between category centers (<1 tightens gaps).")

        pp.add_argument("--usetex", action="store_true", help="render text via LaTeX (verify fonts with pdffonts).")
        pp.add_argument("--corefonts", action="store_true", help="use PDF/PS core fonts knobs (may still be overridden by math).")

    p_mech = sub.add_parser(
        "mechanism",
        aliases=["fig2"],
        help="Paper Fig2: mechanism diagnostics (fidelity/var/influence).",
    )
    p_mech.add_argument("--out_dir", type=str, required=True)
    p_mech.add_argument("--summary_json", type=str, default="", help="analysis_results.summary.json")
    p_mech.add_argument("--mechanism_json", type=str, default="", help="optional simple JSON (overrides summary_json)")
    p_mech.add_argument("--calibration_csv", type=str, default="", help="optional calibration bins csv.")
    p_mech.add_argument("--calibration_methods", type=str, default="C3,MAGRPO", help="comma-separated methods shown in calibration inset.")
    p_mech.add_argument("--fmt", type=str, default="pdf,png", help="comma-separated: pdf,png")
    p_mech.add_argument("--dpi", type=int, default=300)
    p_mech.add_argument("--use_dummy", action="store_true", help="ignore inputs and use dummy data")
    add_style_args(p_mech)

    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    fmt = _parse_list(args.fmt)

    panel_titles = True
    if getattr(args, "no_panel_titles", False):
        panel_titles = False
    if getattr(args, "panel_titles", False):
        panel_titles = True

    style = PaperStyle(
        medium=args.medium,
        theme=args.theme,
        palette=args.palette,
        layout=args.layout,
        show_values=args.show_values,
        show_best_star=bool(args.best_star),
        panel_titles=panel_titles,
        panel_labels=bool(getattr(args, "panel_labels", False)),
        show_legend=not bool(getattr(args, "no_legend", False)),
        fig_width_in=float(args.fig_width_in),
        fig_height_in=float(args.fig_height_in),
        hspace=float(args.hspace),
        wspace=float(args.wspace),
        grid_left=float(args.grid_left),
        grid_right=float(args.grid_right),
        grid_bottom=float(args.grid_bottom),
        grid_top=float(args.grid_top),
        bar_width=float(args.bar_width),
        category_step=float(args.category_step),
    )

    if args.cmd == "mechanism":
        calibration_curves: Optional[List[CalibrationCurve]] = None
        if args.calibration_csv:
            calibration_curves = _load_calibration_curves_from_csv(
                Path(args.calibration_csv),
                methods=_parse_list(args.calibration_methods),
            )
        if args.use_dummy:
            data = _dummy_mechanism_data()
        elif args.mechanism_json:
            data = load_mechanism_from_json(Path(args.mechanism_json))
        elif args.summary_json:
            data = load_mechanism_from_summary(Path(args.summary_json))
        else:
            data = _dummy_mechanism_data()

        paths = plot_mechanism_figure(
            data,
            out_dir=out_dir,
            style=style,
            calibration_curves=calibration_curves,
            fmt=fmt,
            dpi=args.dpi,
            use_tex=bool(args.usetex),
            corefonts=bool(args.corefonts),
        )
        print("\\n".join(str(p) for p in paths))
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
