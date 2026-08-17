"""Grid of per-metric boxplots across DE (or EN) sources, one panel per
displayed metric, with adjacent-step significance brackets.

Reuses `make_metrics_table.compute_sig_cells` for the brackets, so the
figure can never show a significance pattern that disagrees with the
published table: the correction family is the canonical per-language
metric set (`make_metrics_table.family_metrics`), independent of which
subset this figure or that table displays, so Holm sees the same m in
both. Same hard-zero-on-wrong-direction rule. Each bracket spans exactly
one adjacent step (e.g.
Gesund.Bund -> MSD Consumer) and carries a single asterisk if that one
step is right-signed and significant -- unlike a single bracket spanning
the whole source range labelled with a star count, which misrepresents a
multi-step family of tests as one pairwise comparison.

Metrics with a wrong-direction significant flip, or zero significant
right-direction steps, get no brackets at all (mirrors the table's
"no marker" case).

No title/caption on the figure or panels, per project convention: the
metric name is the y-axis label (as in plot_metric_boxplots_slides.py)
and the source names are the legend/x-tick labels.

Usage (from repo root):
    PYTHONPATH=readability python readability/reporting/plot_metrics_grid.py --language de
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from lib.metrics_config import metrics_by_key, plot_title
from make_metrics_table import (
    DIR_KEYS, SOURCES_DE, SOURCES_EN, _displayed_metrics, compute_sig_cells,
)
from lib.stats_direction import VALID_LABELS, VALID_JUDGES

# Okabe-Ito colourblind-safe palette, same family as bleed_through_figure.py,
# one hue per source position in display (easy -> hard) order.
SOURCE_COLORS = ["#009E73", "#E69F00", "#D55E00", "#0072B2"]

# Spelled-out source labels for the legend/x-ticks (compact, single line).
SOURCE_LABELS = {
    "apoum": "Apotheken Umschau",
    "nhs": "NHS",
    "gesund": "Gesund.Bund",
    "msd_short": "MSD Short",
    "msd_lay": "MSD Cons.",
    "msd_expert": "MSD Prof.",
}


def load_column(language: str, source: str, metric) -> pd.Series:
    path = DIR_KEYS[metric.dir_key] / f"{language}_{source}.tsv"
    col = pd.read_csv(path, sep="\t", usecols=[metric.key])[metric.key]
    return col[col != -1].dropna() * metric.scale


WRONG_DIR_COLOR = "#D55E00"  # Okabe-Ito vermillion: flags an unexpected flip


def draw_significance_brackets(ax, x_index: dict[str, int], cells: list[dict],
                               fontsize: float = 11) -> None:
    """One bracket per significant adjacent step, stacked upward:
      - right-signed & significant -> black bracket, single asterisk.
      - WRONG-direction & significant -> vermillion bracket, dagger. This is
        an unexpected flip (the metric moved opposite the expected
        readability direction on that step) -- it's what hard-zeros the
        metric's star count in the table, so it's surfaced here rather than
        silently dropped.
    Non-significant steps get no bracket."""
    sig_steps = [c for c in cells if c["sig"]]
    if not sig_steps:
        return
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    base = ymax + 0.06 * span
    step_gap = 0.10 * span
    for i, cell in enumerate(sig_steps):
        a, b = cell["step"]
        xa, xb = x_index[a], x_index[b]
        lo, hi = (xa, xb) if xa < xb else (xb, xa)
        y = base + i * step_gap
        color = "black" if cell["right"] else WRONG_DIR_COLOR
        mark = "*" if cell["right"] else "†"  # dagger
        ax.plot([lo, lo, hi, hi], [y, y + 0.02 * span, y + 0.02 * span, y],
                color=color, linewidth=1.0)
        ax.text((lo + hi) / 2, y + 0.025 * span, mark,
                ha="center", va="bottom", fontsize=fontsize, color=color)
    ax.set_ylim(ymin, base + len(sig_steps) * step_gap + 0.05 * span)


def plot_grid(
    language: str, leftovers: bool, labels: set[str], judges: set[str],
    alpha: float, correction: str, ncols: int, output: Path,
    metric_keys: list[str] | None = None,
    fig_width: float | None = None, fig_height: float | None = None,
) -> None:
    sources = SOURCES_DE if language == "de" else SOURCES_EN
    if metric_keys is not None:
        by_key = metrics_by_key()
        missing = [k for k in metric_keys if k not in by_key]
        if missing:
            raise KeyError(f"unknown metric keys: {missing}")
        metrics = [by_key[k] for k in metric_keys]
    else:
        # Default figure set is the language's *full* metric list, not the
        # space-constrained main-table subset -- `--leftovers` still gives
        # the DE table's leftover split for anyone who wants that figure.
        metrics = _displayed_metrics(language, leftovers=leftovers,
                                     grid=not leftovers)
    sig_cells = compute_sig_cells(
        language, labels, judges, alpha=alpha, leftovers=leftovers,
        correction=correction, metrics=metrics,
    )
    x_index = {s: i for i, s in enumerate(sources)}

    # Fewer metrics than columns would otherwise leave switched-off axes as
    # dead space on the right (which `bbox_inches="tight"` can't crop, since
    # the legend spans the full figure width) and stretch the legend across
    # it. Shrink the grid to the metrics we actually have.
    ncols = min(ncols, len(metrics))
    nrows = math.ceil(len(metrics) / ncols)
    width = fig_width if fig_width is not None else 3.4 * ncols
    # Panels narrower than the 3.4in/panel default (e.g. a single-column
    # LaTeX figure) need smaller type or labels collide; scale down (never
    # up) relative to that baseline, floored so text stays legible.
    scale = min(1.0, (width / ncols) / 3.4)
    # The legend spans the figure's full width, so it needs its own scale
    # tied to *total* width (not per-panel width like `scale` above) --
    # otherwise a narrow 2-panel figure gets the same absolute legend size
    # as a wide 4-panel row and the legend dwarfs the plots below it.
    legend_scale = min(1.0, width / (3.4 * 4))
    # A narrow figure wraps the legend into a 2-column block instead of one
    # wide row; text stays the same size as the panel titles (`scale`), so
    # the block just gets taller, not smaller.
    legend_ncol = len(sources) if legend_scale >= 1.0 else 2
    legend_rows = math.ceil(len(sources) / legend_ncol)
    # Reserve a fixed inch band at the top for the source legend, on top of
    # per-panel titles -- a *fraction* of height (as tight_layout's `rect`
    # normally wants) would shrink relative to the legend's fixed point
    # size as nrows grows, so compute the fraction from an absolute margin.
    # ~0.35in per legend row is what a `legend_fontsize`-pt row with this
    # handleheight actually occupies; more than that shows up as a visible
    # blank strip between the legend and the panel titles.
    legend_margin_in = 0.35 * scale * legend_rows + 0.1
    height = fig_height if fig_height is not None else 3.0 * nrows + legend_margin_in
    top_frac = 1 - legend_margin_in / height
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, height), squeeze=False,
    )
    flat_axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = flat_axes[i]
        data = [load_column(language, s, metric) for s in sources]
        bp = ax.boxplot(
            data, positions=range(len(sources)), widths=0.6,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.3),
        )
        for patch, color in zip(bp["boxes"], SOURCE_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor(color)

        ax.set_title(plot_title(metric), fontsize=max(10, 13 * scale),
                     fontweight="bold")
        ax.set_xlim(-0.6, len(sources) - 0.4)
        ax.set_xticks(range(len(sources)))
        # Source identity is carried by the legend + fixed left-to-right
        # box order (consistent across every panel), so tick labels are
        # redundant in a multi-panel grid. Keep them only for a
        # single-metric render, where the panel may get cropped out of
        # the figure (e.g. for a slide) and lose its legend.
        single_metric = len(metrics) == 1
        if single_metric:
            ax.set_xticklabels([SOURCE_LABELS.get(s, s) for s in sources],
                               fontsize=max(9, 12 * scale), rotation=40,
                               ha="right")
        else:
            ax.set_xticklabels([])
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.tick_params(axis="y", labelsize=max(9, 12 * scale))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        info = sig_cells.get(metric.key, {})
        if not info.get("skip") and not info.get("missing") and "cells" in info:
            draw_significance_brackets(ax, x_index, info["cells"],
                                       fontsize=max(11, 15 * scale))

    for j in range(len(metrics), len(flat_axes)):
        flat_axes[j].axis("off")

    # Match the per-panel title size so the legend doesn't read as a
    # different (smaller) type scale from the titles sitting right below
    # it. Only the wrap-to-2-columns decision (and the margin reserved for
    # it) depends on the figure's total width via `legend_scale`.
    legend_fontsize = max(10, 13 * scale)

    handles = [
        Patch(facecolor=color, alpha=0.75, edgecolor=color,
              label=SOURCE_LABELS.get(s, s))
        for s, color in zip(sources, SOURCE_COLORS)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=legend_ncol,
               frameon=False, fontsize=legend_fontsize,
               handlelength=1.8, handleheight=1.4,
               handletextpad=0.6, columnspacing=1.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, top_frac))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", choices=["de", "en"], default="de")
    ap.add_argument("--leftovers", action="store_true",
                    help="DE only: the appendix leftover metric set "
                         "instead of the main-table set.")
    ap.add_argument("--alignment-labels", default="richtig")
    ap.add_argument("--include-spez-gen", action="store_true")
    ap.add_argument("--alignment-judges", default="human,llm,site")
    ap.add_argument("--correction", choices=["none", "bonferroni", "holm"],
                    default="holm")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--metrics", nargs="+", default=None,
                    help="Explicit metric keys to plot instead of the "
                         "language's display set (e.g. --metrics "
                         "word_freq_filtered numbers_ratio noun_ratio). "
                         "Bypasses --leftovers.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Defaults to figures/metrics_grid_<lang>.png "
                         "(and a .pdf alongside it).")
    ap.add_argument("--fig-width", type=float, default=None,
                    help="Total figure width in inches, overriding the "
                         "default 3.4*ncols. E.g. 3.3 for an ACL single-"
                         "column figure (\\columnwidth).")
    ap.add_argument("--fig-height", type=float, default=None,
                    help="Total figure height in inches, overriding the "
                         "default 3.0*nrows.")
    args = ap.parse_args()

    if args.leftovers and args.language != "de":
        ap.error("--leftovers is only valid with --language de")

    labels = {s.strip() for s in args.alignment_labels.split(",") if s.strip()}
    bad = labels - VALID_LABELS
    if bad:
        ap.error(f"unknown --alignment-labels values: {sorted(bad)}; "
                 f"allowed: {sorted(VALID_LABELS)}")
    if args.include_spez_gen:
        labels |= {"spezialisierung", "generalisierung"}
    judges = {s.strip() for s in args.alignment_judges.split(",") if s.strip()}
    bad = judges - VALID_JUDGES
    if bad:
        ap.error(f"unknown --alignment-judges values: {sorted(bad)}; "
                 f"allowed: {sorted(VALID_JUDGES)}")

    suffix = "_leftovers" if args.leftovers else ""
    if args.metrics:
        suffix = "_" + "_".join(args.metrics)
    output = args.output or Path(f"figures/metrics_grid_{args.language}{suffix}.png")
    plot_grid(args.language, args.leftovers, labels, judges, args.alpha,
              args.correction, args.ncols, output, metric_keys=args.metrics,
              fig_width=args.fig_width, fig_height=args.fig_height)


if __name__ == "__main__":
    main()
