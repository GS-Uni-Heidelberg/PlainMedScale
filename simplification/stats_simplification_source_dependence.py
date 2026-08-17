"""Within-language source-dependence test on SIMPLIFICATIONS.

Hypothesis: input complexity bleeds through the LLM simplifier. Simplifying
a harder source (e.g. MSD Prof.) leaves the output less readable than
simplifying an easier source (e.g. gesund.bund) of the same topic.

Mirror of `readability/lib/stats_direction.py`, but the metrics are computed on
the *simplified* texts rather than the originals. Same monotonicity chain,
same alignment-judgments-based article pairing.

Per language:
  DE: msd_expert -> msd_lay -> gesund -> apoum   (each is the EASIER side)
  EN: msd_expert -> msd_lay -> gesund -> nhs

For each adjacent (harder, easier) step and each metric, we form per-pair
deltas
    delta_i = sign * [ simp_harder(a_i) - simp_easier(b_i) ]
over aligned article pairs from `alignment/judgment/alignment_judgments.json`,
where (a_i, b_i) are the harder/easier sides of one cluster. `sign` is
`-expected` so positive delta means "harder-source simplification is harder
to read than easier-source simplification" — the H1 direction.

One-sided paired Wilcoxon, H1: median(delta) > 0. A non-leveling simplifier
will show monotonic positive shifts; a perfect leveler shows ns shifts
everywhere.

Pair sources per step (same as lib/stats_direction.py):
  msd_expert->msd_lay : site self-loops `msd:X <-> msd:X` with views
                        (<lang>_professional, <lang>_amateur).
  msd_lay->gesund     : cross-source `msd <-> gesund` qualifying pairs.
  gesund->apoum (DE)  : `gesund <-> apoum` qualifying pairs.
  gesund->nhs (EN)    : `gesund <-> nhs` qualifying pairs.

Usage (from repo root):
  PYTHONPATH=readability python simplification/stats_simplification_source_dependence.py \\
      --run-dir simplification/results/qwen3-30b --language de
  PYTHONPATH=readability python simplification/stats_simplification_source_dependence.py \\
      --run-dir simplification/results/qwen3-30b --language en --monotonic-only
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

# Reuse alignment-pair plumbing from the originals-side directionality test.
from lib.stats_direction import (
    SOURCES_DE,
    SOURCES_EN,
    SOURCE_LABELS,
    VALID_LABELS,
    VALID_JUDGES,
    ALIGNMENT_JSON,
    adjacent_pairs,
    load_pairs,
    pairs_for_step,
    sig_marker,
    wilcoxon_p,
)
from lib.metrics_config import load_metrics


# Per-language confirmed-monotonic sets — refreshed from
# make_metrics_table.compute_sig_markers() (the 3/3 right-signed-significant
# tier, alpha=0.05, labels={richtig}, judges={human,llm,site}). If
# metrics_config.toml signs change OR the human-side dataset changes,
# regenerate via:
#   PYTHONPATH=readability python3 -c "from reporting.make_metrics_table import \
#       compute_sig_markers; print({k for k,v in compute_sig_markers('de', \
#       {'richtig'}, {'human','llm','site'}).items() if v=='***'})"
MONOTONIC_DE = {
    "wstf",
    "adjective_ratio",
    "negations_ratio",
    "word_freq_filtered",
    "jargon_per_100",
}
MONOTONIC_EN = {
    "fkgl",
    "noun_ratio",
    "negations_ratio",
    "crossing_lexical_chains",
    "jargon_per_100",
}


# Subfolder per metric family inside the simplification run dir. Differs from
# the human-side layout: the simp pipeline only emits APPL perplexity (not
# the full BERT NSP/MLM family), so the LLM folder is `metrics_appl_strip`
# rather than `metrics_llm_strip`.
SIMP_FAMILY_FOR_DIRKEY = {
    "std":  "metrics_strip",
    "llm":  "metrics_appl_strip",
    "jarg": "jargon_density",
}


def _toml_metrics() -> list[tuple[str, str, int]]:
    """Build the (column, simp_family_dir, expected_sign) tuples from
    metrics_config.toml. Skips metrics with expected_sign == 0 (those have
    no committed direction, so a one-sided test is meaningless)."""
    out: list[tuple[str, str, int]] = []
    for m in load_metrics():
        if m.expected_sign == 0:
            continue
        family = SIMP_FAMILY_FOR_DIRKEY[m.dir_key]
        out.append((m.key, family, m.expected_sign))
    return out


METRICS = _toml_metrics()

# TSV-source key -> (simplification src, simplification subtree).
# Simplification TSV path = run_dir / <family> / <lang>_<src>_<subtree>.tsv
SIMP_FILE_FOR_SOURCE = {
    "msd_expert": ("msd",    "professional"),
    "msd_lay":    ("msd",    "amateur"),
    "gesund":     ("gesund", "amateur"),
    "apoum":      ("apoum",  "amateur"),
    "nhs":        ("nhs",    "amateur"),
}


def load_simplified_values(
    run_dir: Path, language: str, source: str, col: str, family: str,
) -> dict[str, float]:
    """Return {short_article_id -> metric value} for one source's
    simplification TSV. -1 sentinels and NaN rows are dropped. Empty dict
    if the file or column is missing.

    The simplification TSVs already use bare short_ids that match the
    alignment_judgments.json keys, so no _short_article_id stripping is
    needed here (unlike the originals side).
    """
    simp_src, simp_subtree = SIMP_FILE_FOR_SOURCE[source]
    fp = run_dir / family / f"{language}_{simp_src}_{simp_subtree}.tsv"
    if not fp.exists():
        return {}
    df = pd.read_csv(fp, sep="\t")
    if col not in df.columns or "article_id" not in df.columns:
        return {}
    df["_val"] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["_val"] != -1].dropna(subset=["_val"])
    if df.empty:
        return {}
    return dict(zip(df["article_id"].astype(str), df["_val"].astype(float)))


def rank_biserial_r(deltas: list[float]) -> float:
    """Paired-Wilcoxon rank-biserial r.

    r = (W+ - W-) / (W+ + W-), where W+ and W- are sums of absolute-value
    ranks for positive and negative deltas (zeros dropped, matching
    scipy's zero_method='wilcox'). r in [-1, +1]; conventionally
    |r| < 0.1 trivial, 0.1-0.3 small, 0.3-0.5 medium, >0.5 large.
    """
    arr = np.asarray(deltas, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = arr[arr != 0]
    if arr.size == 0:
        return float("nan")
    abs_ranks = rankdata(np.abs(arr))
    w_plus = float(abs_ranks[arr > 0].sum())
    w_minus = float(abs_ranks[arr < 0].sum())
    total = w_plus + w_minus
    if total == 0:
        return float("nan")
    return (w_plus - w_minus) / total


def pct_predicted_direction(deltas: list[float]) -> float:
    """Fraction of non-zero pairs going the predicted way (positive delta).

    Ignores zeros (consistent with Wilcoxon zero_method='wilcox').
    Returns nan if no non-zero pairs. Useful as a 'how often does it
    actually happen' companion to rank-biserial r.
    """
    arr = np.asarray(deltas, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = arr[arr != 0]
    if arr.size == 0:
        return float("nan")
    return float((arr > 0).sum()) / float(arr.size)


def run(
    run_dir: Path, language: str, alpha: float,
    alignment_labels: set[str], alignment_judges: set[str],
    metric_subset: list[tuple[str, str, int]],
) -> pd.DataFrame:
    sources = SOURCES_DE if language == "de" else SOURCES_EN
    steps = adjacent_pairs(sources)
    _, pairs = load_pairs(alignment_labels, alignment_judges)

    # The DE/EN source lists from lib/stats_direction.py are ordered low->high
    # readability (msd_expert easiest-input...apoum/nhs hardest-input is
    # backwards: msd_expert is HARDEST, apoum/nhs is EASIEST). For each
    # adjacent step (a, b), `a` is the harder source and `b` the easier
    # source — i.e. we expect simplified(a) to read HARDER than
    # simplified(b) under our hypothesis. Sign is built from `expected`
    # so positive delta = harder-than-easier in the metric's direction.
    step_pairs = {
        (a, b): pairs_for_step(pairs, language, a, b) for a, b in steps
    }

    rows: list[dict] = []
    for col, family, expected in metric_subset:
        sign = -expected  # +1 => higher is harder
        row: dict = {
            "metric": col,
            "expected": {-1: "down", +1: "up"}.get(expected, "?"),
        }
        any_missing = False
        for a, b in steps:
            label = f"{SOURCE_LABELS[a]}->{SOURCE_LABELS[b]}"
            a_vals = load_simplified_values(run_dir, language, a, col, family)
            b_vals = load_simplified_values(run_dir, language, b, col, family)
            n_pairs_available = len(step_pairs[(a, b)])
            deltas: list[float] = []
            for a_id, b_id in step_pairs[(a, b)]:
                va = a_vals.get(a_id)
                vb = b_vals.get(b_id)
                if va is None or vb is None:
                    continue
                deltas.append(sign * (va - vb))
            n = len(deltas)
            if n == 0:
                row[f"{label} delta"] = float("nan")
                row[f"{label} p"] = float("nan")
                row[f"{label} sig"] = ""
                row[f"{label} r"] = float("nan")
                row[f"{label} pct_pos"] = float("nan")
                row[f"{label} n"] = 0
                row[f"{label} n_pairs"] = n_pairs_available
                any_missing = True
                continue
            # H1: sign*(simp_a - simp_b) > 0 ⇔ harder simp is harder.
            delta_med, p = wilcoxon_p(deltas, alternative="greater")
            row[f"{label} delta"] = delta_med
            row[f"{label} p"] = p
            row[f"{label} sig"] = sig_marker(p)
            row[f"{label} r"] = rank_biserial_r(deltas)
            row[f"{label} pct_pos"] = pct_predicted_direction(deltas)
            row[f"{label} n"] = n
            row[f"{label} n_pairs"] = n_pairs_available
        row["any_missing"] = any_missing
        rows.append(row)
    return pd.DataFrame(rows)


def format_human(df: pd.DataFrame, language: str,
                 labels: set[str], judges: set[str]) -> str:
    sources = SOURCES_DE if language == "de" else SOURCES_EN
    steps = adjacent_pairs(sources)

    lines = [
        f"=== Source-dependence of simplifications ({language.upper()}) ===",
        "One-sided paired Wilcoxon on per-cluster deltas of SIMPLIFIED texts.",
        "H1: harder-source simplification is HARDER to read than easier-source"
        " simplification of the same topic (input complexity bleeds through).",
        f"Alignment labels = {sorted(labels)}; judges = {sorted(judges)}.",
        "delta = sign * (simp_harder - simp_easier); positive => bleed-through.",
        "",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['metric']:32s}  expected {r['expected']:>4s}"
        )
        for a, b in steps:
            label = f"{SOURCE_LABELS[a]}->{SOURCE_LABELS[b]}"
            d = r.get(f"{label} delta")
            p = r.get(f"{label} p")
            s = r.get(f"{label} sig", "")
            rb = r.get(f"{label} r")
            pp = r.get(f"{label} pct_pos")
            n = r.get(f"{label} n", 0)
            n_pairs = r.get(f"{label} n_pairs", 0)
            extra = f"n={int(n or 0)}/{int(n_pairs or 0)}"
            if pd.isna(d):
                lines.append(f"    {label:35s}  --  ({extra})")
            else:
                lines.append(
                    f"    {label:35s}  delta={d:+.4f}  p={p:.4g}  "
                    f"{s}  r={rb:+.3f}  pct_pos={pp:.2f}  ({extra})"
                )
        lines.append("")

    # Summary: count point-positive / significant across all (metric, step)
    delta_cols = [f"{SOURCE_LABELS[a]}->{SOURCE_LABELS[b]} delta"
                  for a, b in steps]
    p_cols     = [f"{SOURCE_LABELS[a]}->{SOURCE_LABELS[b]} p"
                  for a, b in steps]
    total = 0
    n_pos = 0
    n_sig = 0
    for _, r in df.iterrows():
        for dc, pc in zip(delta_cols, p_cols):
            d = r.get(dc)
            p = r.get(pc)
            if pd.isna(d):
                continue
            total += 1
            if d > 0:
                n_pos += 1
            if not pd.isna(p) and p < 0.05:
                n_sig += 1
    lines.append("--- Summary ---")
    lines.append(
        f"Steps with delta > 0 (bleed-through, point):  {n_pos}/{total}"
    )
    lines.append(
        f"Steps significant at p<0.05 (bleed-through):  {n_sig}/{total}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Run dir holding simplified metrics, e.g. "
             "simplification/results/qwen3-30b",
    )
    parser.add_argument("--language", choices=["de", "en"], default="de")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--monotonic-only", action="store_true",
        help="Restrict METRICS to those that pass the directionality test "
             "in the CHOSEN --language (see readability/lib/stats_direction.py "
             "--one-sided). DE keeps {wstf, word_freq_filtered, "
             "jargon_per_100}; EN keeps {avg_syllable_length, fkgl, wstf, "
             "noun_ratio, jargon_per_100}. This is the right filter for a "
             "within-language test: a metric only needs to track readability "
             "in the language whose simplifications we're comparing.",
    )
    parser.add_argument(
        "--alignment-labels", default="richtig",
        help="Comma-separated label set. Allowed: "
             f"{sorted(VALID_LABELS)}. Default: 'richtig'.",
    )
    parser.add_argument(
        "--include-spez-gen", action="store_true",
        help="Add 'spezialisierung' and 'generalisierung' to the label set.",
    )
    parser.add_argument(
        "--alignment-judges", default="human,llm,site",
        help="Comma-separated judges. Allowed: "
             f"{sorted(VALID_JUDGES)}. Default: 'human,llm,site'.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional TSV path for the full results.",
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

    if args.monotonic_only:
        keep_set = MONOTONIC_DE if args.language == "de" else MONOTONIC_EN
        metric_subset = [m for m in METRICS if m[0] in keep_set]
        if not metric_subset:
            parser.error(
                f"--monotonic-only kept 0 metrics for {args.language.upper()}; "
                f"keep_set={sorted(keep_set)} doesn't intersect METRICS."
            )
        dropped = [m[0] for m in METRICS if m[0] not in keep_set]
        print(
            f"--monotonic-only ({args.language.upper()}): "
            f"keeping {[m[0] for m in metric_subset]}; dropping {dropped}"
        )
        print()
    else:
        metric_subset = METRICS

    df = run(
        args.run_dir, args.language, args.alpha,
        alignment_labels=labels, alignment_judges=judges,
        metric_subset=metric_subset,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, sep="\t", index=False)
    print(format_human(df, args.language, labels=labels, judges=judges))


if __name__ == "__main__":
    main()
