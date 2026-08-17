"""
Emit an ACL-style LaTeX table of readability metrics per source for one language.

Rows = metrics, columns = sources. Each cell is "mean +/- std". Metric labels
are annotated with monotonicity markers (\\textsuperscript{*/**/***}) based
on the one-sided Wilcoxon over adjacent low->high-readability source steps.
The marker counts how many of the N adjacent steps are right-signed AND
significant (p<alpha), with a hard zero for any significant wrong-direction
flip:

    *** = all N steps right-signed AND significant
    **  = N-1 steps right-signed AND significant
    *   = N-2 steps right-signed AND significant (only one)
    (no marker) = no right-signed-significant step, OR at least one
                  significant wrong-direction flip, OR no expected sign,
                  OR missing data

Comment out metrics or sources in the lists below to exclude them.

Usage:
    PYTHONPATH=readability python readability/reporting/make_metrics_table.py --language de
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lib.stats_direction import (
    SOURCES_DE as STATS_SOURCES_DE,
    SOURCES_EN as STATS_SOURCES_EN,
    VALID_LABELS, VALID_JUDGES,
    adjacent_pairs, load_pairs, load_values_by_id, pairs_for_step,
    wilcoxon_p,
)
from lib.metrics_config import load_metrics, metrics_by_key


# ---- sources (same order as in the notebook; comment lines to drop) --------
SOURCES_DE = [
    "apoum",
    "gesund",
    # "msd_short",
    "msd_lay",
    "msd_expert",
]

SOURCES_EN = [
    "nhs",
    "gesund",
    # "msd_short",
    "msd_lay",
    "msd_expert",
]

# Pretty column headers per source
SOURCE_LABELS = {
    "apoum": "Apotheken Umschau",
    "nhs": "NHS",
    "gesund": "Gesund.Bund",
    "msd_short": "MSD Short",
    "msd_lay": "MSD Cons.",
    "msd_expert": "MSD Prof.",
}

# ---- rows to display in the paper table (comment out lines to drop) --------
# Order = display order. Keys must exist in readability/metrics_config.toml,
# which is the single source of truth for title / dir_key / expected_sign /
# decimals (edit signs and labels there, not here).
#
# DE goes in the main body (space-constrained), so DISPLAY_KEYS_DE is the
# Scholz-significant+correct-direction subset plus WSTF + jargon. EN goes in
# the appendix (no space constraint), so DISPLAY_KEYS_EN is the full Scholz
# metric set plus jargon. WSTF is a German formula and is omitted from EN.
DISPLAY_KEYS_DE: list[str] = [
    "wstf",
    "noun_ratio",
    "adjective_ratio",
    "function_word_ratio",
    "numbers_ratio",
    "negations_ratio",
    "grammar_frequency",
    "word_freq_filtered",
    "lexical_chain_lens",
    "crossing_lexical_chains",
    "jargon_per_100",
]

DISPLAY_KEYS_EN: list[str] = [
    # statistical
    "avg_sentence_length",
    "avg_syllable_length",
    "fkgl",
    # POS-based
    "noun_ratio",
    "adjective_ratio",
    "function_word_ratio",
    "numbers_ratio",
    "negations_ratio",
    # syntactic
    "grammar_frequency",
    "adjacent_sent_edit_distance",
    "dep_tree_depth",
    "noun_phrase_complexity",
    # semantic
    "word_freq_filtered",
    "lmfm",
    # fluency
    "appl_filtered_mean",  # TODO: confirm which APPL variant matches Scholz
    "lexical_chain_lens",
    "crossing_lexical_chains",
    "ansp",
    # jargon (our addition)
    "jargon_per_100",
]

DISPLAY_KEYS_PER_LANG: dict[str, list[str]] = {
    "de": DISPLAY_KEYS_DE,
    "en": DISPLAY_KEYS_EN,
}

# Appendix-only leftover set for DE: the Scholz metrics not retained in
# the space-constrained main-body table DISPLAY_KEYS_DE. Selected with
# --leftovers; only valid for --language de.
DISPLAY_KEYS_DE_LEFTOVERS: list[str] = [
    "avg_sentence_length",
    "avg_syllable_length",
    "fkgl",
    "adjacent_sent_edit_distance",
    "dep_tree_depth",
    "noun_phrase_complexity",
    "lmfm",
    "appl_filtered_mean",
    "ansp",
]

# Metric sets for the *figures* (plot_metrics_grid.py), which are not
# space-constrained the way the main-body DE table is: both languages show
# every metric in one grid, so DE is not split into a leftovers figure.
# GRID_KEYS_DE is DISPLAY_KEYS_DE + DISPLAY_KEYS_DE_LEFTOVERS reordered into
# the same thematic grouping DISPLAY_KEYS_EN uses (statistical, POS-based,
# syntactic, semantic, fluency, jargon) so the DE and EN grids read the same
# way panel-for-panel. EN's grid set is just DISPLAY_KEYS_EN: it is already
# the full set, WSTF being a German formula that does not apply to English.
GRID_KEYS_DE: list[str] = [
    # statistical
    "avg_sentence_length",
    "avg_syllable_length",
    "fkgl",
    "wstf",
    # POS-based
    "noun_ratio",
    "adjective_ratio",
    "function_word_ratio",
    "numbers_ratio",
    "negations_ratio",
    # syntactic
    "grammar_frequency",
    "adjacent_sent_edit_distance",
    "dep_tree_depth",
    "noun_phrase_complexity",
    # semantic
    "word_freq_filtered",
    "lmfm",
    # fluency
    "appl_filtered_mean",
    "lexical_chain_lens",
    "crossing_lexical_chains",
    "ansp",
    # jargon (our addition)
    "jargon_per_100",
]

GRID_KEYS_PER_LANG: dict[str, list[str]] = {
    "de": GRID_KEYS_DE,
    "en": DISPLAY_KEYS_EN,
}


DIR_KEYS = {
    "std":  Path("data/metrics_strip"),
    "llm":  Path("data/metrics_llm_strip"),
    "jarg": Path("data/jargon_density"),
}


def _check_grid_covers_table_sets() -> None:
    """GRID_KEYS_DE must be a reordering of the two DE table sets.

    The figure set is maintained separately from the table sets (different
    ordering), so an edit to one can silently drop a metric from the other.
    Fail loudly at import instead.
    """
    table_side = set(DISPLAY_KEYS_DE) | set(DISPLAY_KEYS_DE_LEFTOVERS)
    grid_side = set(GRID_KEYS_DE)
    if table_side != grid_side:
        raise ValueError(
            "GRID_KEYS_DE out of sync with DISPLAY_KEYS_DE + "
            f"DISPLAY_KEYS_DE_LEFTOVERS: only in tables "
            f"{sorted(table_side - grid_side)}, only in grid "
            f"{sorted(grid_side - table_side)}"
        )


_check_grid_covers_table_sets()


def _displayed_metrics(language: str, leftovers: bool = False,
                       grid: bool = False):
    """Return the Metric records for the language's display key list in
    order, validating that every requested key exists in
    metrics_config.toml. `leftovers=True` selects DE's appendix-only
    leftover set; `grid=True` selects the full-set figure ordering used by
    plot_metrics_grid.py (and makes `leftovers` meaningless, since the
    figure set is not split)."""
    if grid:
        if leftovers:
            raise ValueError("grid=True and leftovers=True are exclusive")
        keys = GRID_KEYS_PER_LANG[language]
        name = f"GRID_KEYS_{language.upper()}"
    elif leftovers:
        if language != "de":
            raise ValueError("leftovers=True only valid for language='de'")
        keys = DISPLAY_KEYS_DE_LEFTOVERS
        name = "DISPLAY_KEYS_DE_LEFTOVERS"
    else:
        keys = DISPLAY_KEYS_PER_LANG[language]
        name = f"DISPLAY_KEYS_{language.upper()}"
    by_key = metrics_by_key()
    out = []
    missing = []
    for k in keys:
        m = by_key.get(k)
        if m is None:
            missing.append(k)
        else:
            out.append(m)
    if missing:
        raise KeyError(
            f"{name} entries missing from metrics_config.toml: {missing}"
        )
    return out


def load_cache(language: str, sources: list[str]) -> dict:
    cache = {"std": {}, "llm": {}, "jarg": {}}
    for key, d in DIR_KEYS.items():
        for s in sources:
            fp = d / f"{language}_{s}.tsv"
            if fp.exists():
                cache[key][s] = pd.read_csv(fp, sep="\t")
            else:
                cache[key][s] = None
    return cache


def fmt(mean: float, std: float, nd: int = 2) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}"


def cell_value(df: pd.DataFrame | None, col: str) -> tuple[float, float]:
    if df is None or col not in df.columns:
        return float("nan"), float("nan")
    s = df[col].replace(-1, pd.NA).dropna()
    if s.empty:
        return float("nan"), float("nan")
    return float(s.mean()), float(s.std())


def reject_family(pvals: np.ndarray, method: str, alpha: float) -> np.ndarray:
    """Boolean reject vector for a family of p-values under `method`.

    'none'       -> per-test alpha (no multiple-comparison correction).
    'bonferroni' -> alpha / m per test.
    'holm'       -> Holm-Bonferroni step-down (same FWER control as
                    Bonferroni, uniformly more powerful).
    """
    m = len(pvals)
    if m == 0:
        return np.zeros(0, dtype=bool)
    if method == "none":
        return pvals < alpha
    if method == "bonferroni":
        return pvals < (alpha / m)
    if method == "holm":
        order = np.argsort(pvals)
        reject = np.zeros(m, dtype=bool)
        for rank, idx in enumerate(order):
            if pvals[idx] < alpha / (m - rank):
                reject[idx] = True
            else:
                break  # Holm stops at the first non-rejection
        return reject
    raise ValueError(f"unknown correction method: {method}")


def family_metrics(language: str, extra: list | None = None) -> list:
    """The canonical multiple-comparison family for a language: every
    metric computed for that language, i.e. GRID_KEYS_PER_LANG (for DE the
    union of the main-body table set and the appendix leftovers; for EN the
    full set).

    The family deliberately does NOT depend on which subset a given table
    or figure happens to display. If it did, the main-body table, the
    appendix leftovers table and the metrics grid would report different
    significance for the same step purely because Holm saw a different m.

    `extra` adds Metric records that are outside the canonical set (e.g. an
    explicit --metrics selection naming a key not in the language's grid
    list), so they still get a corrected p rather than none at all.
    """
    fam = _displayed_metrics(language, grid=True)
    if extra:
        seen = {m.key for m in fam}
        fam = fam + [m for m in extra if m.key not in seen]
    return fam


def compute_sig_cells(
    language: str, labels: set[str], judges: set[str], alpha: float = 0.05,
    leftovers: bool = False, correction: str = "holm",
    metrics: list | None = None,
) -> dict[str, dict]:
    """Per-metric, per-adjacent-step significance, corrected across the
    canonical per-language family (`family_metrics`): every metric x
    adjacent-step cell with an expected sign and non-missing p.

    Returns {metric_key: info} where info is one of:
      {"skip": True}                       -- expected_sign == 0
      {"skip": False, "missing": True, "cells": [...]}  -- some step had
        no data / undefined p (cells present for the steps that DID
        resolve; a fully-missing metric has cells == [])
      {"skip": False, "missing": False,
       "cells": [{"step": (a, b), "right": bool, "sig": bool,
                  "delta": float}, ...]}

    The returned dict covers the whole family, which is a superset of any
    one table's or figure's displayed metrics -- callers look up only the
    keys they show. `correction` is one of 'none' | 'bonferroni' | 'holm'
    (default 'holm'). Shared by `compute_sig_markers` (table stars) and
    `plot_metrics_grid.py` (bracket asterisks) so the two can never
    disagree.

    `metrics`, if given, only *extends* the family with keys outside the
    canonical set (see `family_metrics`); it never narrows it. `leftovers`
    is likewise not a family selector -- it is kept only for signature
    compatibility with the display-set callers.
    """
    sources = STATS_SOURCES_DE if language == "de" else STATS_SOURCES_EN
    steps = adjacent_pairs(sources)
    _, pairs = load_pairs(labels, judges)
    step_pairs = {(a, b): pairs_for_step(pairs, language, a, b)
                  for a, b in steps}

    metrics = family_metrics(language, extra=metrics)

    # Pass 1: gather per-metric cells and a flat family of p-values (each
    # cell's p is oriented TOWARD its expected direction: right-direction
    # cells keep p, wrong-direction cells contribute 1 - p).
    per_metric: dict[str, dict] = {}
    family_p: list[float] = []
    family_ref: list[tuple[str, int]] = []  # (metric_key, cell index)
    for metric in metrics:
        col = metric.key
        expected = metric.expected_sign
        if expected == 0:
            per_metric[col] = {"skip": True}
            continue
        alternative = "greater" if expected > 0 else "less"
        values = {s: load_values_by_id(language, s, col, metric.dir_key)
                  for s in sources}
        info: dict = {"skip": False, "missing": False, "cells": []}
        for a, b in steps:
            deltas: list[float] = []
            for a_id, b_id in step_pairs[(a, b)]:
                va = values[a].get(a_id)
                vb = values[b].get(b_id)
                if va is None or vb is None:
                    continue
                deltas.append(vb - va)
            if not deltas:
                info["missing"] = True
                continue
            delta_med, p = wilcoxon_p(deltas, alternative)
            if pd.isna(p):
                info["missing"] = True
                continue
            right = ((expected > 0 and delta_med > 0)
                     or (expected < 0 and delta_med < 0))
            # One-sided p toward the expected direction: ~1 when the
            # observed shift is in the wrong direction, so its wrong-side
            # p is 1 - p.
            p_toward = p if right else (1.0 - p)
            family_ref.append((col, len(info["cells"])))
            family_p.append(p_toward)
            info["cells"].append({
                "step": (a, b), "right": right, "sig": False,
                "delta": delta_med,
            })
        per_metric[col] = info

    # Pass 2: correct across the whole family, write significance back.
    reject = reject_family(np.asarray(family_p, dtype=float), correction, alpha)
    for (col, idx), r in zip(family_ref, reject):
        per_metric[col]["cells"][idx]["sig"] = bool(r)

    return per_metric


def compute_label_emphasis(
    language: str, per_metric: dict[str, dict],
) -> dict[str, str]:
    """Per-metric label emphasis derived from the same per-step cells used
    for the row markers:

        'bold'      = all N adjacent steps are right-signed and significant
        'underline' = at least one but not all N steps are right-signed and
                      significant, AND no significant wrong-direction flip
        ''          = zero right-signed-significant steps, OR any
                      significant wrong-direction flip, OR no expected
                      sign, OR missing data

    N is the number of adjacent low->high-readability steps in the
    canonical source chain (STATS_SOURCES_{DE,EN}).
    """
    sources = STATS_SOURCES_DE if language == "de" else STATS_SOURCES_EN
    n_steps = len(adjacent_pairs(sources))
    emphasis: dict[str, str] = {}
    for col, info in per_metric.items():
        if info.get("skip") or info.get("missing"):
            emphasis[col] = ""
            continue
        cells = info["cells"]
        wrong_sig = any((not c["right"]) and c["sig"] for c in cells)
        right_sig = sum(1 for c in cells if c["right"] and c["sig"])
        if wrong_sig or right_sig == 0:
            emphasis[col] = ""
        elif right_sig == n_steps:
            emphasis[col] = "bold"
        else:
            emphasis[col] = "underline"
    return emphasis


def compute_sig_markers(
    language: str, labels: set[str], judges: set[str], alpha: float = 0.05,
    leftovers: bool = False, correction: str = "holm",
) -> dict[str, str]:
    """For every displayed metric with non-zero expected sign, run the one-
    sided Wilcoxon over each adjacent (low->high readability) source step:

        '*' = significant in the expected direction
        '†' = significant in the opposite direction
        ''  = not significant OR no expected sign / missing data

    Significance is decided AFTER a multiple-comparison correction applied
    across the canonical per-language family (`family_metrics`): every
    (metric computed for this language, adjacent step) cell that has an
    expected sign and a non-missing p -- NOT just the metrics this
    particular table displays, so the main table, the leftovers table and
    the metrics grid all agree on every step. `correction`
    is one of 'none' | 'bonferroni' | 'holm' (default 'holm'); 'none'
    reproduces the raw per-test alpha=0.05 behaviour. The wrong-direction
    cells are corrected too, so a marginal wrong-direction flip that stops
    being significant under correction no longer hard-zeros its metric.

    Uses the canonical source chain from lib/stats_direction.py (which may differ
    from the local SOURCES_{DE,EN} display order). For the four-source
    canonical chain N=3.

    Returns `(markers, per_metric)`; `per_metric` is the raw
    `compute_sig_cells` output, handed back so callers (e.g. the label
    bold/underline emphasis) can derive from the same computation instead
    of re-running the Wilcoxon family.
    """
    sources = STATS_SOURCES_DE if language == "de" else STATS_SOURCES_EN
    per_metric = compute_sig_cells(
        language, labels, judges, alpha=alpha, leftovers=leftovers,
        correction=correction,
    )
    markers: dict[str] = {}
    for col, info in per_metric.items():
        if info.get("skip") or info.get("missing"):
            #markers[col] = ""
            #continue
            for source in sources[:-1]:
                markers[(col, source)] = ""
        if info.get("skip"):
            continue  # no expected sign -> no cells to iterate
        cells = info["cells"]
        #wrong_sig = any((not c["right"]) and c["sig"] for c in cells)
        #right_sig = sum(1 for c in cells if c["right"] and c["sig"])
        #markers[col] = "" if (wrong_sig or right_sig == 0) else "*" * right_sig
        for c in cells:
            step_a, step_b = c["step"]
            if (not c["right"]) and c["sig"]:
                markers[(col, step_a)] = "\\textdagger"
            elif c["right"] and c["sig"]:
                markers[(col, step_a)] = "*"
            else:
                markers[(col, step_a)] = ""
    return markers, per_metric


def build_table(
    language: str, markers: dict[str, str], emphasis: dict[str, str] | None = None,
    leftovers: bool = False, correction: str = "holm",
) -> str:
    sources = SOURCES_DE if language == "de" else SOURCES_EN
    cache = load_cache(language, sources)

    header = ["Metric"] # + [SOURCE_LABELS[s] for s in sources]
    for i, s in enumerate(sources):
        if i > 0:
            header.append("")
        header.append(SOURCE_LABELS[s])

    col_spec = "l" + "c" * (2*len(sources) - 1)

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for metric in _displayed_metrics(language, leftovers=leftovers):
        col = metric.key
        emph = (emphasis or {}).get(col, "")
        if emph == "bold":
            label = f"\\textbf{{{metric.title_en}}}"
        elif emph == "underline":
            label = f"\\underline{{{metric.title_en}}}"
        else:
            label = metric.title_en
        cells = [label]
        for i, s in enumerate(sources):
            if i > 0:
                cells.append(markers.get((col, s), ""))
            m, sd = cell_value(cache[metric.dir_key].get(s), col)
            cells.append(fmt(m * metric.scale, sd * metric.scale,
                             nd=metric.decimals))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    # superscripts = # of adjacent low->high steps that are right-signed
    # and significant (one-sided Wilcoxon, alpha=0.05): *** all 3, ** two,
    # * one, none = zero right-direction significant steps or a wrong-
    # direction flip
    lines.append("\\end{table*}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["de", "en"], default="de")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to write the .tex table. Defaults to "
             "latex_tables/metrics_table_<lang>.tex. Use --stdout to print.",
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="Print to stdout instead of writing to --output.",
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
    parser.add_argument(
        "--leftovers", action="store_true",
        help="DE only: emit the appendix table of Scholz metrics not "
             "retained in the space-constrained main-body table "
             "(DISPLAY_KEYS_DE_LEFTOVERS).",
    )
    parser.add_argument(
        "--correction", choices=["none", "bonferroni", "holm"], default="holm",
        help="Multiple-comparison correction applied across the whole "
             "per-table family of metric x step Wilcoxon tests before "
             "starring. 'none' = raw per-test alpha (old behaviour). "
             "Default 'holm'.",
    )
    args = parser.parse_args()

    if args.leftovers and args.language != "de":
        parser.error("--leftovers is only valid with --language de")

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

    markers, per_metric = compute_sig_markers(
        args.language, labels, judges, leftovers=args.leftovers,
        correction=args.correction,
    )
    emphasis = compute_label_emphasis(args.language, per_metric)
    table = build_table(args.language, markers, emphasis=emphasis,
                        leftovers=args.leftovers, correction=args.correction)
    if args.stdout:
        print(table)
    else:
        suffix = "_leftovers" if args.leftovers else ""
        output = args.output or (
            Path("latex_tables")
            / f"metrics_table_{args.language}{suffix}.tex"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table, encoding="utf-8")
        print(f"wrote {output}")
