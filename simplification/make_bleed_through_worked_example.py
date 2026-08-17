"""
Emit a worked-example LaTeX table that justifies the bleed-through (BT%)
calculation used in `make_simplification_table.py`. Default: jargon density
DE, the headline result.

For one (metric, language), prints/emits a per-step breakdown:
    orig_med, simp_med, per-step ratio (simp/orig as %)
and the aggregate
    BT% = mean(simp_meds) / mean(orig_meds)  (algebraically = sum/sum)

Then prints the three rejected aggregation schemes on the same numbers so
the choice can be defended in the prose:
    median of per-step ratios, pooled (concat then ratio), end-to-end.

Usage (from repo root):
    PYTHONPATH=readability python simplification/make_bleed_through_worked_example.py \\
        --run-dir simplification/results/qwen3-30b \\
        --metric jargon_per_100 --language de
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from lib.stats_direction import (
    SOURCES_DE,
    SOURCES_EN,
    VALID_LABELS,
    VALID_JUDGES,
    adjacent_pairs,
    load_pairs,
    load_values_by_id,
    pairs_for_step,
)
from lib.metrics_config import metrics_by_key
from stats_simplification_source_dependence import (
    SIMP_FAMILY_FOR_DIRKEY,
    load_simplified_values,
)
from make_simplification_table import SOURCE_LABELS


def collect_paired_deltas(
    run_dir: Path, language: str, key: str,
    labels: set[str], judges: set[str],
):
    """Return list[(step_label, orig_deltas, simp_deltas, n)] for the metric's
    adjacent steps in the canonical chain."""
    by_key = metrics_by_key()
    m = by_key[key]
    family = SIMP_FAMILY_FOR_DIRKEY[m.dir_key]
    sign = -m.expected_sign

    sources = SOURCES_DE if language == "de" else SOURCES_EN
    steps = adjacent_pairs(sources)

    _, pairs = load_pairs(labels, judges)
    step_pairs = {(a, b): pairs_for_step(pairs, language, a, b)
                  for a, b in steps}

    simp_values = {s: load_simplified_values(run_dir, language, s, key, family)
                   for s in sources}
    orig_values = {s: load_values_by_id(language, s, key, m.dir_key)
                   for s in sources}

    out = []
    for a, b in steps:
        label = (f"{SOURCE_LABELS.get(a, a)}$\\to$"
                 f"{SOURCE_LABELS.get(b, b)}")
        orig_deltas: list[float] = []
        simp_deltas: list[float] = []
        for a_id, b_id in step_pairs[(a, b)]:
            sva = simp_values[a].get(a_id)
            svb = simp_values[b].get(b_id)
            ova = orig_values[a].get(a_id)
            ovb = orig_values[b].get(b_id)
            if sva is None or svb is None or ova is None or ovb is None:
                continue
            orig_deltas.append(sign * (ova - ovb))
            simp_deltas.append(sign * (sva - svb))
        out.append((label, orig_deltas, simp_deltas, len(orig_deltas)))
    return out


def _med(xs: list[float]) -> float:
    return float(pd.Series(xs).median()) if xs else float("nan")


def aggregations(per_step):
    """Compute the chosen BT% and the three rejected alternatives.

    per_step: list[(label, orig_deltas, simp_deltas, n)]
    """
    step_orig_meds = [_med(o) for _, o, _, _ in per_step]
    step_simp_meds = [_med(s) for _, _, s, _ in per_step]
    step_ratios = [
        100.0 * sm / om if om not in (0, float("nan")) else float("nan")
        for sm, om in zip(step_simp_meds, step_orig_meds)
    ]

    # Chosen: mean of step medians, ratio on the means.
    mean_orig = float(pd.Series(step_orig_meds).mean())
    mean_simp = float(pd.Series(step_simp_meds).mean())
    chosen = 100.0 * mean_simp / mean_orig if mean_orig != 0 else float("nan")

    # Rejected: median of per-step ratios.
    med_ratio = _med(step_ratios)

    # Rejected: pooled across all pairs (concat, ratio of medians).
    all_orig = [v for _, o, _, _ in per_step for v in o]
    all_simp = [v for _, _, s, _ in per_step for v in s]
    pooled = (100.0 * _med(all_simp) / _med(all_orig)
              if _med(all_orig) not in (0,) else float("nan"))

    # Rejected: end-to-end only (first and last step's union? — actually the
    # convention is "hardest vs easiest pair set only", i.e. only the
    # last step's pair set... no — the end-to-end set means align the
    # extremes directly. Approximate: use only the final step (the hardest
    # vs easiest tier transition that the corpus chain expresses). For the
    # 4-tier chain that's MSD Prof. -> ApoUm or MSD Prof. -> NHS, but we
    # don't have direct alignment for the extremes; the closest stand-in
    # is the LAST step (Gesund -> ApoUm / NHS) which uses the
    # easiest-side pairs. Document this caveat in prose; for the worked
    # example we report the last step's ratio under this label.
    last_orig_med = step_orig_meds[-1] if step_orig_meds else float("nan")
    last_simp_med = step_simp_meds[-1] if step_simp_meds else float("nan")
    end_to_end = (100.0 * last_simp_med / last_orig_med
                  if last_orig_med != 0 else float("nan"))

    return {
        "step_orig_meds": step_orig_meds,
        "step_simp_meds": step_simp_meds,
        "step_ratios": step_ratios,
        "chosen": chosen,
        "median_of_ratios": med_ratio,
        "pooled": pooled,
        "end_to_end_proxy": end_to_end,
        "mean_orig": mean_orig,
        "mean_simp": mean_simp,
    }


def build_table(
    run_dir: Path, language: str, key: str,
    labels: set[str], judges: set[str],
) -> str:
    by_key = metrics_by_key()
    m = by_key[key]
    title = m.title_en
    scale = m.scale
    # Use one more decimal than the published metrics table since the
    # per-step medians are typically smaller than the cross-source means.
    dec = max(0, m.decimals)
    per_step = collect_paired_deltas(run_dir, language, key, labels, judges)
    agg = aggregations(per_step)

    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Step & $\\tilde\\Delta_{\\text{orig}}$ & "
        "$\\tilde\\Delta_{\\text{simp}}$ & "
        "$\\tilde\\Delta_{\\text{simp}}/\\tilde\\Delta_{\\text{orig}}$ & "
        "$n$ \\\\"
    )
    lines.append("\\midrule")
    for (label, _, _, n), om, sm, r in zip(
        per_step, agg["step_orig_meds"], agg["step_simp_meds"],
        agg["step_ratios"],
    ):
        lines.append(
            f"{label} & {om * scale:,.{dec}f} & {sm * scale:,.{dec}f} & "
            f"{r:+.0f}\\% & {n} \\\\"
        )
    lines.append("\\midrule")
    lines.append(
        f"Mean across steps & {agg['mean_orig'] * scale:,.{dec}f} & "
        f"{agg['mean_simp'] * scale:,.{dec}f} & "
        f"\\textbf{{{agg['chosen']:+.0f}\\%}} & -- \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lang_label = "German" if language == "de" else "English"
    caption = (
        f"Worked example for the bleed-through (BT\\%) metric, {title}, "
        f"{lang_label}. For each adjacent harder$\\to$easier source step we "
        "form per-cluster paired deltas (sign-aligned with the metric's "
        "harder-is-higher direction) on the original and simplified texts, "
        "then take their medians $\\tilde\\Delta_{\\text{orig}}$ and "
        "$\\tilde\\Delta_{\\text{simp}}$. The cell value in "
        "Tab.~\\ref{tab:simp-bleedthrough} is the across-step mean of "
        "$\\tilde\\Delta_{\\text{simp}}$ divided by the across-step mean of "
        "$\\tilde\\Delta_{\\text{orig}}$ "
        "(equivalently $\\sum\\tilde\\Delta_{\\text{simp}}/"
        "\\sum\\tilde\\Delta_{\\text{orig}}$): a residual simplified gap "
        "expressed as a percentage of the input-side gap."
    )
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{tab:bt-worked-{language}-{key.replace('_','-')}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Simplification run dir (e.g. simplification/results/"
             "qwen3-30b).",
    )
    parser.add_argument("--language", choices=["de", "en"], default="de")
    parser.add_argument("--metric", default="jargon_per_100",
                        help="Metric key (TOML key, default jargon_per_100).")
    parser.add_argument("--alignment-labels", default="richtig")
    parser.add_argument("--alignment-judges", default="human,llm,site")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    labels = {s.strip() for s in args.alignment_labels.split(",") if s.strip()}
    bad = labels - VALID_LABELS
    if bad:
        parser.error(f"unknown --alignment-labels: {sorted(bad)}")
    judges = {s.strip() for s in args.alignment_judges.split(",") if s.strip()}
    bad = judges - VALID_JUDGES
    if bad:
        parser.error(f"unknown --alignment-judges: {sorted(bad)}")

    # Print the alternative aggregations to stdout for use in prose footnotes.
    per_step = collect_paired_deltas(
        args.run_dir, args.language, args.metric, labels, judges,
    )
    agg = aggregations(per_step)
    print(f"# {args.metric} / {args.language}")
    print(f"# Step orig medians : "
          f"{[round(x, 2) for x in agg['step_orig_meds']]}")
    print(f"# Step simp medians : "
          f"{[round(x, 2) for x in agg['step_simp_meds']]}")
    print(f"# Per-step ratios   : "
          f"{[round(x, 1) for x in agg['step_ratios']]}")
    print()
    print(f"  Chosen (mean-of-step-medians)  BT% = "
          f"{agg['chosen']:+.1f}%")
    print(f"  Median of per-step ratios       = "
          f"{agg['median_of_ratios']:+.1f}%")
    print(f"  Pooled (concat, ratio of medians)= "
          f"{agg['pooled']:+.1f}%")
    print(f"  Last-step only (end-to-end proxy)= "
          f"{agg['end_to_end_proxy']:+.1f}%")
    print()

    table = build_table(
        args.run_dir, args.language, args.metric, labels, judges,
    )
    if args.stdout:
        print(table)
    else:
        suffix = f"{args.language}_{args.metric}"
        output = args.output or (
            Path("latex_tables") / f"bt_worked_example_{suffix}.tex"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table, encoding="utf-8")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
