"""
Emit a half-page LaTeX table summarizing input-complexity bleed-through in the
simplification experiment.

For each metric confirmed monotonic on the human originals (3/3 right-signed
significant steps in `make_metrics_table.compute_sig_markers()` — i.e. the
`***` tier), test whether the SIMPLIFIED outputs preserve the same monotonic
ordering across source pairs. Cells show
\\textsuperscript{*/**/***} counts of right-direction-significant adjacent
steps on the one-sided Wilcoxon over per-cluster deltas of simplified texts.

Layout: rows = metrics, columns = DE / EN. Where a metric is NOT confirmed
monotonic for that language on the originals, the cell is '--' (testing it
on the simplifications would be meaningless: the input-side ordering it's
supposed to bleed through doesn't hold).

The main table is restricted to metrics that survived into the human-side
main table (Scholz-significant + correct-direction for German; see
`KEPT_FROM_HUMAN_TABLE`). The `--appendix` mode emits the per-step detail
tables across all monotonic metrics.

Usage (from repo root):
    # main table -> latex_tables/simp_bleedthrough.tex
    PYTHONPATH=readability python simplification/make_simplification_table.py \\
        --run-dir simplification/results/qwen3-30b
    # appendix per-step tables -> latex_tables/simp_bleedthrough_appendix.tex
    PYTHONPATH=readability python simplification/make_simplification_table.py \\
        --run-dir simplification/results/qwen3-30b --appendix
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
    wilcoxon_p,
)
from lib.metrics_config import load_metrics, metrics_by_key
from stats_simplification_source_dependence import (
    MONOTONIC_DE,
    MONOTONIC_EN,
    SIMP_FAMILY_FOR_DIRKEY,
    load_simplified_values,
)


SOURCE_LABELS = {
    "apoum": "Apotheken Umschau",
    "nhs": "NHS",
    "gesund": "Gesund.Bund",
    "msd_short": "MSD Short",
    "msd_lay": "MSD Cons.",
    "msd_expert": "MSD Prof.",
}


# Metric keys that survived into the human-side main table (image 1 in the
# paper draft): Scholz-significant AND correct-direction-on-German. Used to
# split the bleed-through table into a main (kept) view and a dropped-metric
# companion for the appendix. Keep this in sync with the rows actually
# emitted into the human-side main metrics table.
KEPT_FROM_HUMAN_TABLE: set[str] = {
    "wstf",
    "noun_ratio",
    "adjective_ratio",
    "function_word_ratio",
    "numbers_ratio",
    "negations_ratio",
    "grammar_frequency",
    "word_freq_filtered",
    "jargon_per_100",
    "lexical_chain_lens",
    "crossing_lexical_chains",
}


def bleed_through_cell(
    run_dir: Path,
    language: str,
    col: str,
    family: str,
    dir_key: str,
    expected: int,
    step_pairs: dict[tuple[str, str], list[tuple[str, str]]],
    sources: list[str],
    alpha: float,
) -> tuple[str, float, list[tuple[str, float, int]]]:
    """Return (stars, bleed_pct, per_step) for one metric in one language.

    `stars` counts adjacent (harder->easier) steps that are right-signed
    AND significant on the one-sided Wilcoxon over per-cluster simplified
    deltas (mirrors make_metrics_table.compute_sig_markers exactly):

        '***' = all N steps right-signed AND p<alpha
        '**'  = N-1 right-signed-significant
        '*'   = N-2 right-signed-significant (only one, with N=3)
        ''    = zero right-signed-significant, OR any wrong-direction
                significant flip, OR missing data

    `bleed_pct` is 100 * mean across steps of median(simp_delta) divided
    by mean across steps of median(orig_delta), both in the harder->easier
    direction. Each step contributes one (orig_med, simp_med) pair so the
    across-step average is stratified by construction (equal weight per
    step regardless of N), and combining at the totals layer prevents a
    small orig_med on a single step from inflating the cell. 100% =
    simplification preserved the full input-side gap; 0% = gap closed;
    negative = direction inverted in the simplified outputs. NaN if no
    step contributed.

    `per_step` is one entry per adjacent step in source order, each
    (step_stars, step_pct, step_n): step_stars = "*" if this step was
    right-signed-significant else ""; step_pct = 100 * simp_med /
    orig_med for the step (NaN if no data or orig_med == 0); step_n =
    number of cluster pairs contributing.
    """
    sign = -expected
    simp_values = {
        s: load_simplified_values(run_dir, language, s, col, family)
        for s in sources
    }
    orig_values = {
        s: load_values_by_id(language, s, col, dir_key)
        for s in sources
    }
    right_sig = 0
    wrong_sig = False
    any_missing = False
    step_orig_meds: list[float] = []
    step_simp_meds: list[float] = []
    per_step: list[tuple[str, float, int]] = []
    for a, b in step_pairs:
        simp_deltas: list[float] = []
        orig_deltas: list[float] = []
        for a_id, b_id in step_pairs[(a, b)]:
            sva = simp_values[a].get(a_id)
            svb = simp_values[b].get(b_id)
            ova = orig_values[a].get(a_id)
            ovb = orig_values[b].get(b_id)
            if sva is None or svb is None or ova is None or ovb is None:
                continue
            simp_deltas.append(sign * (sva - svb))
            orig_deltas.append(sign * (ova - ovb))
        n_step = len(simp_deltas)
        if not simp_deltas:
            any_missing = True
            per_step.append(("", float("nan"), 0))
            continue
        simp_med, p = wilcoxon_p(simp_deltas, alternative="greater")
        if pd.isna(p):
            any_missing = True
            per_step.append(("", float("nan"), n_step))
            continue
        orig_med = float(pd.Series(orig_deltas).median())
        step_orig_meds.append(orig_med)
        step_simp_meds.append(simp_med)
        right = simp_med > 0
        step_stars = ""
        if right and p < alpha:
            right_sig += 1
            step_stars = "*"
        elif not right:
            if (1.0 - p) < alpha:
                wrong_sig = True
        step_pct = (
            100.0 * simp_med / orig_med if orig_med != 0 else float("nan")
        )
        per_step.append((step_stars, step_pct, n_step))
    if any_missing or wrong_sig or right_sig == 0:
        stars = ""
    else:
        stars = "*" * right_sig
    if step_orig_meds:
        mean_orig = float(pd.Series(step_orig_meds).mean())
        mean_simp = float(pd.Series(step_simp_meds).mean())
        bleed_pct = (
            100.0 * mean_simp / mean_orig if mean_orig != 0 else float("nan")
        )
    else:
        bleed_pct = float("nan")
    return stars, bleed_pct, per_step


def _compute(
    run_dir: Path,
    alpha: float,
    labels: set[str],
    judges: set[str],
):
    """Shared computation for the main and appendix tables.

    Returns (cells, per_lang_steps, keys_union, by_key) where:
      cells[key][lang] is None (metric not monotonic in that lang) or
        the bleed_through_cell tuple (stars, bleed_pct, per_step).
      per_lang_steps[lang] = (sources, steps, step_pair_map).
      keys_union = metric keys in TOML order that are monotonic in DE or EN.
    """
    by_key = metrics_by_key()
    keys_union = [m.key for m in load_metrics()
                  if m.key in (MONOTONIC_DE | MONOTONIC_EN)]

    _, pairs = load_pairs(labels, judges)

    per_lang_steps = {}
    for lang, sources in [("de", SOURCES_DE), ("en", SOURCES_EN)]:
        steps = adjacent_pairs(sources)
        per_lang_steps[lang] = (
            sources,
            steps,
            {(a, b): pairs_for_step(pairs, lang, a, b) for a, b in steps},
        )

    cells: dict[str, dict[str, tuple | None]] = {}
    for key in keys_union:
        metric = by_key[key]
        family = SIMP_FAMILY_FOR_DIRKEY[metric.dir_key]
        cells[key] = {}
        for lang, confirmed in (("de", MONOTONIC_DE), ("en", MONOTONIC_EN)):
            if key not in confirmed:
                cells[key][lang] = None
                continue
            sources, _, step_pairs = per_lang_steps[lang]
            cells[key][lang] = bleed_through_cell(
                run_dir, lang, key, family, metric.dir_key,
                metric.expected_sign,
                step_pairs, sources, alpha,
            )
    return cells, per_lang_steps, keys_union, by_key


def _fmt_pct_stars(pct: float, stars: str) -> str:
    if pd.isna(pct):
        return "--"
    suffix = f"\\textsuperscript{{{stars}}}" if stars else ""
    return f"{pct:+.0f}\\%{suffix}"


def _build_compact_table(
    cells: dict[str, dict[str, tuple | None]],
    keys: list[str],
    by_key: dict,
) -> str:
    def fmt(c) -> str:
        if c is None:
            return "--"
        stars, pct, _ = c
        return _fmt_pct_stars(pct, stars)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Metric & DE & EN \\\\")
    lines.append("\\midrule")
    for key in keys:
        metric = by_key[key]
        lines.append(
            f"{metric.title_en} & {fmt(cells[key]['de'])} "
            f"& {fmt(cells[key]['en'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    # stars = # of 3 adjacent harder->easier steps right-signed-significant
    # (one-sided Wilcoxon, alpha=0.05); -- = not monotonic on originals
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_table(
    run_dir: Path,
    alpha: float,
    labels: set[str],
    judges: set[str],
) -> str:
    cells, _, keys_union, by_key = _compute(run_dir, alpha, labels, judges)
    keys = [k for k in keys_union if k in KEPT_FROM_HUMAN_TABLE]
    return _build_compact_table(cells, keys, by_key)


def build_appendix_table(
    run_dir: Path,
    alpha: float,
    labels: set[str],
    judges: set[str],
) -> str:
    cells, per_lang_steps, keys_union, by_key = _compute(
        run_dir, alpha, labels, judges,
    )

    lines: list[str] = []
    for lang_idx, (lang, confirmed) in enumerate([
        ("de", MONOTONIC_DE),
        ("en", MONOTONIC_EN),
    ]):
        _, steps, _ = per_lang_steps[lang]
        step_headers = [
            f"{SOURCE_LABELS.get(a, a)}$\\to${SOURCE_LABELS.get(b, b)}"
            for a, b in steps
        ]
        rows = [k for k in keys_union if k in confirmed]
        ncols = len(steps) + 1
        col_spec = "l" + "c" * ncols

        if lang_idx > 0:
            lines.append("")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\setlength{\\tabcolsep}{4pt}")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        lines.append(
            "Metric & " + " & ".join(step_headers) + " & Mean \\\\"
        )
        lines.append("\\midrule")
        for key in rows:
            metric = by_key[key]
            stars_total, mean_pct, per_step = cells[key][lang]
            row = [metric.title_en]
            for s_stars, s_pct, _ in per_step:
                row.append(_fmt_pct_stars(s_pct, s_stars))
            row.append(_fmt_pct_stars(mean_pct, stars_total))
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        # step cell = 100 * median paired simplified gap / median paired
        # original gap on that step's cluster pairs; Mean = across-step
        # mean of medians; stars = right-signed-significant (Wilcoxon,
        # alpha=0.05), on Mean = count of such steps
        lines.append("\\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Run dir holding simplified metrics, e.g. "
             "simplification/results/qwen3-30b",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to write the .tex table. Defaults to "
             "latex_tables/simp_bleedthrough[_appendix].tex. "
             "Use --stdout to print instead.",
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="Print to stdout instead of writing to --output.",
    )
    parser.add_argument(
        "--appendix", action="store_true",
        help="Emit the per-step appendix tables (one per language) "
             "instead of the compact main table.",
    )
    parser.add_argument(
        "--alignment-labels", default="richtig",
        help="Comma-separated label set used to define 'aligned' pairs. "
             f"Allowed: {sorted(VALID_LABELS)}. Default: 'richtig'.",
    )
    parser.add_argument(
        "--include-spez-gen", action="store_true",
        help="Convenience: add 'spezialisierung' and 'generalisierung' "
             "to --alignment-labels.",
    )
    parser.add_argument(
        "--alignment-judges", default="human,llm,site",
        help="Comma-separated judges to trust. "
             f"Allowed: {sorted(VALID_JUDGES)}. Default: 'human,llm,site'.",
    )
    args = parser.parse_args()

    labels = {s.strip() for s in args.alignment_labels.split(",") if s.strip()}
    bad = labels - VALID_LABELS
    if bad:
        parser.error(f"unknown --alignment-labels values: {sorted(bad)}; "
                     f"allowed: {sorted(VALID_LABELS)}")
    if args.include_spez_gen:
        labels |= {"spezialisierung", "generalisierung"}
    judges = {s.strip() for s in args.alignment_judges.split(",") if s.strip()}
    bad = judges - VALID_JUDGES
    if bad:
        parser.error(f"unknown --alignment-judges values: {sorted(bad)}; "
                     f"allowed: {sorted(VALID_JUDGES)}")

    if args.appendix:
        builder = build_appendix_table
        default_name = "simp_bleedthrough_appendix.tex"
    else:
        builder = build_table
        default_name = "simp_bleedthrough.tex"
    table = builder(args.run_dir, args.alpha, labels, judges)
    if args.stdout:
        print(table)
    else:
        output = args.output or (Path("latex_tables") / default_name)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table, encoding="utf-8")
        print(f"wrote {output}")
