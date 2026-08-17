"""Diagnostic: does an "off"-looking box in plot_metrics_grid.py come from
including articles with no cross-source alignment partner?

plot_metrics_grid.py boxes every article in a source's TSV, aligned or not.
This script restricts two adjacent sources to the subset of articles that
are actually aligned to each other (via data/alignment/
alignment_judgments.json, same pairing logic as lib/stats_direction.py) and
draws full-distribution vs aligned-only boxes side by side for an eye test.

No title/caption, per project convention.

Usage (from repo root):
    PYTHONPATH=readability python readability/reporting/plot_diag_aligned_boxplot.py \\
        --language de --metric lexical_chain_lens --sources gesund msd_lay
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lib.metrics_config import metrics_by_key, plot_title
from lib.stats_direction import (
    DIR_KEYS, SOURCE_LABELS, load_pairs, pairs_for_step, load_values_by_id,
)

COLORS = {"all": "#0072B2", "aligned": "#D55E00"}  # Okabe-Ito


def plot(
    language: str, metric_key: str, source_a: str, source_b: str,
    labels: set[str], judges: set[str], output: Path,
    dump_pairs: Path | None = None,
) -> None:
    metric = metrics_by_key()[metric_key]
    values_a = load_values_by_id(language, source_a, metric_key, metric.dir_key)
    values_b = load_values_by_id(language, source_b, metric_key, metric.dir_key)

    _, pairs = load_pairs(labels, judges)
    step_pairs = pairs_for_step(pairs, language, source_a, source_b)

    aligned_a, aligned_b = [], []
    rows = []
    for a_id, b_id in step_pairs:
        va, vb = values_a.get(a_id), values_b.get(b_id)
        if va is None or vb is None:
            continue
        aligned_a.append(va)
        aligned_b.append(vb)
        rows.append({
            f"{source_a}_id": a_id, f"{source_b}_id": b_id,
            f"{source_a}_{metric_key}": va, f"{source_b}_{metric_key}": vb,
        })

    print(f"{source_a}: n_all={len(values_a)} n_aligned={len(aligned_a)}")
    print(f"{source_b}: n_all={len(values_b)} n_aligned={len(aligned_b)}")

    if dump_pairs is not None:
        dump_pairs.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(dump_pairs, sep="\t", index=False)
        print(f"wrote {dump_pairs}")

    fig, ax = plt.subplots(figsize=(5, 4))
    positions_all = [0, 2]
    positions_aligned = [0.6, 2.6]
    data_all = [list(values_a.values()), list(values_b.values())]
    data_aligned = [aligned_a, aligned_b]

    bp_all = ax.boxplot(
        data_all, positions=positions_all, widths=0.5, patch_artist=True,
        showfliers=False, medianprops=dict(color="black", linewidth=1.3),
    )
    bp_aligned = ax.boxplot(
        data_aligned, positions=positions_aligned, widths=0.5,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.3),
    )
    for patch in bp_all["boxes"]:
        patch.set_facecolor(COLORS["all"])
        patch.set_alpha(0.75)
        patch.set_edgecolor(COLORS["all"])
    for patch in bp_aligned["boxes"]:
        patch.set_facecolor(COLORS["aligned"])
        patch.set_alpha(0.75)
        patch.set_edgecolor(COLORS["aligned"])

    ax.set_xticks([0.3, 2.3])
    ax.set_xticklabels([
        SOURCE_LABELS.get(source_a, source_a),
        SOURCE_LABELS.get(source_b, source_b),
    ])
    ax.set_ylabel(plot_title(metric))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["all"], alpha=0.75,
                      edgecolor=COLORS["all"], label="all"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["aligned"], alpha=0.75,
                      edgecolor=COLORS["aligned"], label="aligned-only"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", choices=["de", "en"], default="de")
    ap.add_argument("--metric", default="lexical_chain_lens")
    ap.add_argument("--sources", nargs=2, default=["gesund", "msd_lay"],
                    metavar=("SOURCE_A", "SOURCE_B"),
                    help="TSV source keys, e.g. gesund msd_lay.")
    ap.add_argument("--alignment-labels", default="richtig")
    ap.add_argument("--alignment-judges", default="human,llm,site")
    ap.add_argument("--output", type=Path, default=None,
                    help="Defaults to figures/diag_aligned_boxplot_"
                         "<lang>_<metric>_<a>_<b>.png")
    ap.add_argument("--dump-pairs", type=Path, default=None,
                    help="Write the aligned (id, id, value, value) rows "
                         "used in the aligned-only boxes to this TSV.")
    args = ap.parse_args()

    labels = {s.strip() for s in args.alignment_labels.split(",") if s.strip()}
    judges = {s.strip() for s in args.alignment_judges.split(",") if s.strip()}

    a, b = args.sources
    output = args.output or Path(
        f"figures/diag_aligned_boxplot_{args.language}_{args.metric}_{a}_{b}.png"
    )
    plot(args.language, args.metric, a, b, labels, judges, output,
         dump_pairs=args.dump_pairs)


if __name__ == "__main__":
    main()
