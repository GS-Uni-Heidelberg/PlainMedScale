"""
Grid figure showing original vs simplified per-tier medians, one panel per
metric. Visualises bleed-through: both lines slope downward across the
4-tier readability chain, but the simplified-side line is dramatically
compressed in absolute terms even though BT% (its slope relative to the
original line) is non-trivial.

Default --metrics is the paper's DISPLAY_KEYS_<language> table column set
from make_metrics_table.py, so this stays in sync with what's reported.

Usage (from repo root):
    PYTHONPATH=readability:simplification python \\
        simplification/plot_bleed_through.py \\
        --run-dir simplification/results/qwen3-30b \\
        --language de
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.stats_direction import load_values_by_id
from lib.metrics_config import metrics_by_key, plot_title
from stats_simplification_source_dependence import (
    SIMP_FAMILY_FOR_DIRKEY,
    load_simplified_values,
)
from make_simplification_table import SOURCE_LABELS
from reporting.make_metrics_table import (
    DISPLAY_KEYS_DE, DISPLAY_KEYS_EN, SOURCES_DE, SOURCES_EN,
)

DEFAULT_METRICS = {"de": DISPLAY_KEYS_DE, "en": DISPLAY_KEYS_EN}

# Abbreviations for --short-labels, used when many panels share one row and
# the rotated x tick labels would otherwise collide. Only sources whose full
# SOURCE_LABELS entry is too long need an entry here; the rest fall through.
SHORT_SOURCE_LABELS = {
    "apoum": "ApoUm",
}


def per_tier_medians(
    run_dir: Path, language: str, key: str, short_labels: bool = False,
) -> tuple[list[str], list[float], list[float]]:
    """Return (tier_labels, orig_medians, simp_medians) across the chain
    in order easiest -> hardest (so the line reads left-to-right as the
    canonical readability chain). Values use the metric's `scale`."""
    by_key = metrics_by_key()
    m = by_key[key]
    family = SIMP_FAMILY_FOR_DIRKEY[m.dir_key]

    # SOURCES_DE/_EN (from make_metrics_table) are easiest -> hardest
    # (apoum/nhs first, msd_expert last), matching the column order used
    # in the metrics table and plot_metrics_grid.py.
    chain = SOURCES_DE if language == "de" else SOURCES_EN

    orig_meds: list[float] = []
    simp_meds: list[float] = []
    labels: list[str] = []
    for src in chain:
        orig = load_values_by_id(language, src, key, m.dir_key)
        simp = load_simplified_values(run_dir, language, src, key, family)
        full = SOURCE_LABELS.get(src, src)
        labels.append(
            SHORT_SOURCE_LABELS.get(src, full) if short_labels else full
        )
        orig_meds.append(
            float(pd.Series(list(orig.values())).median()) * m.scale
            if orig else float("nan")
        )
        simp_meds.append(
            float(pd.Series(list(simp.values())).median()) * m.scale
            if simp else float("nan")
        )
    return labels, orig_meds, simp_meds


def plot(
    run_dir: Path, language: str, metrics: list[str], output: Path,
    ncols: int = 3, panel_width: float = 4.0, panel_height: float = 3.2,
    short_labels: bool = False,
) -> None:
    by_key = metrics_by_key()
    n = len(metrics)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_width * ncols, panel_height * nrows), squeeze=False,
    )
    flat_axes = axes.flatten()

    for ax, key in zip(flat_axes, metrics):
        m = by_key[key]
        labels, orig_meds, simp_meds = per_tier_medians(
            run_dir, language, key, short_labels=short_labels,
        )
        xs = np.arange(len(labels))
        ax.plot(
            xs, orig_meds, marker="o", linewidth=2,
            color="#0b5394", label="Originals",
        )
        ax.plot(
            xs, simp_meds, marker="s", linewidth=2,
            color="#e69138", label="Simplifications",
        )
        # Apoum (DE only) is the human-written easy-language endpoint of
        # the chain; draw its median as a horizontal reference spanning
        # the panel so it's comparable to the simplified line at every
        # tier, not just at the first x-tick.
        if language == "de" and not np.isnan(orig_meds[0]):
            ax.axhline(
                orig_meds[0], linestyle=":", linewidth=1.2,
                color="#0b5394", alpha=0.6,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(plot_title(m))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="best")
    for ax in flat_axes[n:]:
        ax.set_visible(False)

    plt.subplots_adjust(hspace=0.2, wspace=0.1)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Simplification run dir.",
    )
    parser.add_argument("--language", choices=["de", "en"], default="de")
    parser.add_argument(
        "--metrics", nargs="+", default=None,
        help="Metric keys to plot (one panel per metric). Defaults to the "
             "paper's DISPLAY_KEYS_<language> table columns from "
             "make_metrics_table.py.",
    )
    parser.add_argument(
        "--ncols", type=int, default=3,
        help="Panels per row when laying out multiple metrics. Default 3.",
    )
    parser.add_argument(
        "--panel-width", type=float, default=4.0,
        help="Width in inches per panel. Default 4.0.",
    )
    parser.add_argument(
        "--panel-height", type=float, default=3.2,
        help="Height in inches per panel. Default 3.2.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Defaults to figures/bt_panels_<language>.pdf.",
    )
    parser.add_argument(
        "--short-labels", action="store_true",
        help="Abbreviate long source names on the x axis (Apotheken "
             "Umschau -> ApoUm). For wide single-row layouts where the "
             "full labels collide.",
    )
    args = parser.parse_args()

    metrics = args.metrics or DEFAULT_METRICS[args.language]
    output = args.output or (
        Path("figures") / f"bt_panels_{args.language}.pdf"
    )
    plot(
        args.run_dir, args.language, metrics, output, ncols=args.ncols,
        panel_width=args.panel_width, panel_height=args.panel_height,
        short_labels=args.short_labels,
    )


if __name__ == "__main__":
    main()
