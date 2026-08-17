"""Directionality test: do readability metrics shift in the expected
direction as source readability increases, and is the shift significant
between adjacent sources?

Source order is low -> high readability:
    DE: msd_expert -> msd_lay -> gesund -> apoum
    EN: msd_expert -> msd_lay -> gesund -> nhs

For each metric and each adjacent (A, B) step we form per-pair deltas
delta_i = value_B(article_i) - value_A(article_i_partner) over the
aligned article pairs from data/alignment/alignment_judgments.json, then
run a one-sample Wilcoxon signed-rank test against H0: median(delta) = 0
(one-sided when --one-sided is set, matching the metric's expected sign).

Pair sources per step:
  msd_expert->msd_lay : site self-loops `msd:X <-> msd:X` with views
                        (<lang>_professional, <lang>_amateur).
  msd_lay->gesund     : cross-source `msd <-> gesund` qualifying pairs.
  gesund->apoum (DE)  : `gesund <-> apoum` qualifying pairs.
  gesund->nhs (EN)    : `gesund <-> nhs` qualifying pairs.

A "qualifying" pair has label in --alignment-labels and judge in
--alignment-judges. Site self-loops have label=richtig, judge=site,
so they are always included when judge "site" is in the filter set.

Usage:
    PYTHONPATH=readability python readability/lib/stats_direction.py --language de
    PYTHONPATH=readability python readability/lib/stats_direction.py --language en --one-sided --output out.tsv
    PYTHONPATH=readability python readability/lib/stats_direction.py --language de --one-sided \\
        --alignment-labels richtig,spezialisierung --alignment-judges human,site
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lib.metrics_config import load_metrics


# low -> high readability
SOURCES_DE = ["msd_expert", "msd_lay", "gesund", "apoum"]
SOURCES_EN = ["msd_expert", "msd_lay", "gesund", "nhs"]

SOURCE_LABELS = {
    "apoum": "Apotheken Umschau",
    "nhs": "NHS",
    "gesund": "GesundBund",
    "msd_lay": "MSD Cons.",
    "msd_expert": "MSD Prof.",
}

# (column, source_dir_key, expected_sign) tuples derived from the canonical
# metric config at readability/metrics_config.toml. Edit signs there, not here.
#   +1 = should INCREASE as readability increases (e.g. word_freq)
#   -1 = should DECREASE as readability increases (e.g. jargon_per_100)
#    0 = no clear prior; reported but not graded
METRICS = [(m.key, m.dir_key, m.expected_sign) for m in load_metrics()]

DIR_KEYS = {
    "std":  Path("data/metrics_strip"),
    "llm":  Path("data/metrics_llm_strip"),
    "jarg": Path("data/jargon_density"),
}

ALIGNMENT_JSON = Path("data/alignment/alignment_judgments.json")
VALID_LABELS = {"richtig", "spezialisierung", "generalisierung"}
VALID_JUDGES = {"human", "llm", "site"}

# Map per-source-TSV name -> the source key used in alignment_judgments.json.
TSV_SOURCE_TO_NODE_SOURCE = {
    "apoum":      "apoum",
    "gesund":     "gesund",
    "nhs":        "nhs",
    "msd_expert": "msd",
    "msd_lay":    "msd",
}

# Per-TSV-source, the view tag we expect on a self-loop endpoint when
# that source represents that TSV. Used for tier disambiguation in the
# msd_expert <-> msd_lay step.
TSV_VIEW_TAG = {
    ("msd_expert", "de"): "de_professional",
    ("msd_expert", "en"): "en_professional",
    ("msd_lay",    "de"): "de_amateur",
    ("msd_lay",    "en"): "en_amateur",
}


def _short_article_id(source: str, tsv_id: str) -> str:
    """Reduce a TSV article_id to the bare short_id used after `<source>:`
    in alignment_judgments.json.

    - apoum / nhs: TSV id == bare id (numeric / hash).
    - gesund: TSV id is "<id>_<lang>", strip lang suffix.
    - msd_*: TSV id is "<short>_<lang>_<tier>_<lang>_<i>", short is the first
      "_"-split chunk (e.g. "1-105").
    """
    if source in ("apoum", "nhs"):
        return tsv_id
    return tsv_id.split("_", 1)[0]


def load_pairs(
    labels: set[str], judges: set[str], json_path: Path = ALIGNMENT_JSON,
) -> tuple[dict, list[dict]]:
    """Load alignment_judgments.json and return (articles, pairs).

    Pairs are filtered to those whose `label` is in `labels` and `judge`
    is in `judges`. Each returned pair dict carries normalized fields:
        a_source, a_id, a_view, b_source, b_id, b_view, label, judge.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"alignment file not found: {json_path}")
    with json_path.open() as f:
        d = json.load(f)
    articles = d["articles"]
    out: list[dict] = []
    for p in d["pairs"]:
        if p["label"] not in labels or p["judge"] not in judges:
            continue
        a_meta = articles[p["a"]]
        b_meta = articles[p["b"]]
        out.append({
            "a_source": a_meta["source"],
            "a_id":     p["a"].split(":", 1)[1],
            "a_view":   p.get("a_view"),
            "b_source": b_meta["source"],
            "b_id":     p["b"].split(":", 1)[1],
            "b_view":   p.get("b_view"),
            "label":    p["label"],
            "judge":    p["judge"],
        })
    return articles, out


def pairs_for_step(
    pairs: list[dict], language: str, a_src_tsv: str, b_src_tsv: str,
) -> list[tuple[str, str]]:
    """Return ordered (a_short_id, b_short_id) tuples for this adjacency
    step, where the first element is the article in `a_src_tsv` and the
    second is the article in `b_src_tsv`.

    For the msd_expert <-> msd_lay step we require:
        - both endpoints have source "msd"
        - same short_id on both endpoints (a self-loop), and
        - views match the (professional, amateur) tier pair in this language.

    For cross-source steps we require:
        - the two endpoint sources match {a_node, b_node}.

    Output is deduped (a label/judge may produce duplicate (a, b)).
    """
    a_node = TSV_SOURCE_TO_NODE_SOURCE[a_src_tsv]
    b_node = TSV_SOURCE_TO_NODE_SOURCE[b_src_tsv]
    intra = a_node == b_node  # only the msd_expert -> msd_lay step
    a_view_needed = TSV_VIEW_TAG.get((a_src_tsv, language))
    b_view_needed = TSV_VIEW_TAG.get((b_src_tsv, language))

    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for p in pairs:
        if intra:
            if p["a_source"] != a_node or p["b_source"] != b_node:
                continue
            if p["a_id"] != p["b_id"]:
                continue  # only self-loops indicate a within-article tier link
            # Endpoint order is arbitrary; match views to our (a, b) order
            if p["a_view"] == a_view_needed and p["b_view"] == b_view_needed:
                a_id, b_id = p["a_id"], p["b_id"]
            elif p["a_view"] == b_view_needed and p["b_view"] == a_view_needed:
                a_id, b_id = p["b_id"], p["a_id"]
            else:
                continue
        else:
            if p["a_source"] == a_node and p["b_source"] == b_node:
                a_id, b_id = p["a_id"], p["b_id"]
            elif p["a_source"] == b_node and p["b_source"] == a_node:
                a_id, b_id = p["b_id"], p["a_id"]
            else:
                continue
        key = (a_id, b_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def load_values_by_id(
    language: str, source: str, col: str, dir_key: str,
) -> dict[str, float]:
    """Return {short_id -> metric value} for one source/language TSV.
    Sentinel -1 and NaN rows are dropped. If the file or column is
    missing, returns an empty dict.
    """
    fp = DIR_KEYS[dir_key] / f"{language}_{source}.tsv"
    if not fp.exists():
        return {}
    df = pd.read_csv(fp, sep="\t")
    if col not in df.columns:
        return {}
    df["_short"] = df["article_id"].astype(str).map(
        lambda x: _short_article_id(source, x)
    )
    df["_val"] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["_val"] != -1)].dropna(subset=["_val"])
    return dict(zip(df["_short"], df["_val"].astype(float)))


def sig_marker(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def is_right_direction(delta: float, expected: int) -> bool:
    """Whether a per-step shift matches the metric's expected sign."""
    if delta == 0:
        return False
    return (delta > 0 and expected > 0) or (delta < 0 and expected < 0)


def wrong_direction_marker(delta: float, p: float, expected: int,
                            one_sided: bool) -> str:
    """Significance marker for a step's shift *in the wrong direction*.

    Returns "" for right-sign steps (nothing wrong to flag) or when the
    metric has no directional expectation. For wrong-sign steps, mirrors
    `grade`'s wrong_p convention: one-sided tests report p toward the
    expected direction, so the wrong-direction p-value is 1 - p; under
    two-sided we just use p.
    """
    if expected == 0 or pd.isna(delta) or pd.isna(p):
        return ""
    if is_right_direction(delta, expected):
        return ""
    wrong_p = (1.0 - p) if one_sided else p
    return sig_marker(wrong_p)


def grade(expected: int, shifts: list[float], pvals: list[float],
          alpha: float, one_sided: bool) -> str:
    """Four-level verdict over the per-step shifts.

    For each step we classify the shift as right- vs wrong-sign and as
    significant vs not. When the test is one-sided in the expected
    direction, the wrong-direction p-value is 1 - p; under two-sided we
    just use p.

    Levels:
      - PASS           : all right-sign and significant
      - monotonic, X/N : all right-sign, but only X<N significant
      - NEAR-MONO      : at least 1 wrong-sign flip, but no flip is
                         significant in the wrong direction
      - WRONG DIRECTION: at least 1 wrong-sign flip that IS significant
                         in the wrong direction
    """
    if expected == 0:
        return "(no expectation)"
    if not shifts:
        return "missing data"

    n = len(shifts)
    right_sig = right_nonsig = wrong_sig = wrong_nonsig = 0
    for s, p in zip(shifts, pvals):
        if is_right_direction(s, expected):
            if p < alpha:
                right_sig += 1
            else:
                right_nonsig += 1
        else:
            wrong_p = (1.0 - p) if one_sided else p
            if wrong_p < alpha:
                wrong_sig += 1
            else:
                wrong_nonsig += 1

    if wrong_sig:
        return f"WRONG DIRECTION ({wrong_sig}/{n} sig.)"
    if wrong_nonsig:
        return f"NEAR-MONO (flip ns; {right_sig}/{n} right-sig.)"
    if right_sig == n:
        return "PASS (monotonic, all sig.)"
    return f"monotonic, {right_sig}/{n} sig."


def adjacent_pairs(sources: list[str]) -> list[tuple[str, str]]:
    return list(zip(sources[:-1], sources[1:]))


def wilcoxon_p(deltas: list[float], alternative: str) -> tuple[float, float]:
    """Return (median_delta, p) for one-sample Wilcoxon signed-rank.

    Pairs with delta == 0 are dropped by `zero_method="wilcox"`. If after
    that fewer than 1 non-zero pairs remain we return (nan, nan).
    """
    arr = np.asarray(deltas, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(arr))
    try:
        _, p = stats.wilcoxon(arr, alternative=alternative, zero_method="wilcox")
        return med, float(p)
    except ValueError:
        return med, float("nan")


def run(language: str, alpha: float, one_sided: bool,
        alignment_labels: set[str], alignment_judges: set[str]) -> pd.DataFrame:
    sources = SOURCES_DE if language == "de" else SOURCES_EN
    steps = adjacent_pairs(sources)
    _, pairs = load_pairs(alignment_labels, alignment_judges)

    # Pre-compute per-step pair lists once (metric-independent)
    step_pairs = {
        (a, b): pairs_for_step(pairs, language, a, b) for a, b in steps
    }

    rows = []
    for col, dir_key, expected in METRICS:
        values = {
            s: load_values_by_id(language, s, col, dir_key) for s in sources
        }
        # Wilcoxon alternative: paired delta = value_B - value_A
        #   expected=+1 (should INCREASE A->B) => H1: median(delta) > 0
        #   expected=-1 (should DECREASE A->B) => H1: median(delta) < 0
        if one_sided and expected == +1:
            alternative = "greater"
        elif one_sided and expected == -1:
            alternative = "less"
        else:
            alternative = "two-sided"

        row: dict = {
            "metric": col,
            "expected": {-1: "down", +1: "up", 0: "?"}[expected],
            "alternative": alternative,
        }
        shifts: list[float] = []
        pvals: list[float] = []
        any_missing = False
        for a, b in steps:
            label = f"{SOURCE_LABELS[a]}->{SOURCE_LABELS[b]}"
            n_pairs_available = len(step_pairs[(a, b)])
            deltas: list[float] = []
            for a_id, b_id in step_pairs[(a, b)]:
                va = values[a].get(a_id)
                vb = values[b].get(b_id)
                if va is None or vb is None:
                    continue
                deltas.append(vb - va)
            n = len(deltas)
            if n == 0:
                row[f"{label} delta"] = float("nan")
                row[f"{label} p"] = float("nan")
                row[f"{label} sig"] = ""
                row[f"{label} wrong_sig"] = ""
                row[f"{label} n"] = 0
                row[f"{label} n_pairs"] = n_pairs_available
                any_missing = True
                continue
            delta_med, p = wilcoxon_p(deltas, alternative)
            shifts.append(delta_med)
            pvals.append(p)
            row[f"{label} delta"] = delta_med
            row[f"{label} p"] = p
            row[f"{label} sig"] = sig_marker(p)
            row[f"{label} wrong_sig"] = wrong_direction_marker(
                delta_med, p, expected, one_sided
            )
            row[f"{label} n"] = n
            row[f"{label} n_pairs"] = n_pairs_available
        row["verdict"] = "missing data" if any_missing else grade(
            expected, shifts, pvals, alpha, one_sided=one_sided
        )
        rows.append(row)

    return pd.DataFrame(rows)


def format_human(df: pd.DataFrame, language: str,
                 labels: set[str], judges: set[str]) -> str:
    sources = SOURCES_DE if language == "de" else SOURCES_EN
    pairs = adjacent_pairs(sources)

    lines = [
        f"=== Directionality of readability metrics ({language.upper()}) ===",
        f"Paired test (Wilcoxon signed-rank on per-pair deltas).",
        f"Alignment labels = {sorted(labels)}; judges = {sorted(judges)}.",
        "",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['metric']:32s}  expected {r['expected']:>4s}   -> {r['verdict']}"
        )
        for a, b in pairs:
            label = f"{SOURCE_LABELS[a]}->{SOURCE_LABELS[b]}"
            d = r.get(f"{label} delta")
            p = r.get(f"{label} p")
            s = r.get(f"{label} sig", "")
            ws = r.get(f"{label} wrong_sig", "")
            n = r.get(f"{label} n", 0)
            n_pairs = r.get(f"{label} n_pairs", 0)
            extra = f"n={int(n or 0)}/{int(n_pairs or 0)}"
            wrong_flag = f"  WRONG-DIR:{ws}" if ws not in ("", "ns") else ""
            if pd.isna(d):
                lines.append(f"    {label:35s}  --  ({extra})")
            else:
                lines.append(
                    f"    {label:35s}  delta={d:+.4f}  p={p:.4g}  "
                    f"{s}  ({extra}){wrong_flag}"
                )
        lines.append("")

    passing = df[df["verdict"].str.startswith("PASS", na=False)]["metric"].tolist()
    monotonic = df[df["verdict"].str.startswith("monotonic", na=False)]["metric"].tolist()
    near_mono = df[df["verdict"].str.startswith("NEAR-MONO", na=False)]["metric"].tolist()
    wrong = df[df["verdict"].str.startswith("WRONG DIRECTION", na=False)]["metric"].tolist()
    lines.append("--- Summary ---")
    lines.append(f"PASS  (monotonic & all significant) [{len(passing)}]: {', '.join(passing) or '-'}")
    lines.append(f"PARTIAL (monotonic, some sig.)      [{len(monotonic)}]: {', '.join(monotonic) or '-'}")
    lines.append(f"NEAR-MONO (flip ns)                 [{len(near_mono)}]: {', '.join(near_mono) or '-'}")
    lines.append(f"WRONG DIRECTION (flip sig)          [{len(wrong)}]: {', '.join(wrong) or '-'}")
    lines.append("")
    lines.append("Per-step n = pairs with both metric values present "
                 "(out of n_pairs aligned at that step).")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["de", "en"], default="de")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--one-sided", action="store_true",
        help="Use a directional Wilcoxon per metric (greater/less from "
             "the expected sign). Sign=0 metrics stay two-sided. Run "
             "this only AFTER expected signs are committed to avoid "
             "double-dipping.",
    )
    parser.add_argument(
        "--alignment-labels",
        default="richtig",
        help="Comma-separated label set used to define 'aligned'. "
             f"Allowed: {sorted(VALID_LABELS)}. Default: 'richtig'. "
             "Pass e.g. 'richtig,spezialisierung,generalisierung' to "
             "include all positive judgements.",
    )
    parser.add_argument(
        "--include-spez-gen", action="store_true",
        help="Convenience shortcut: add 'spezialisierung' and "
             "'generalisierung' to the active label set on top of "
             "whatever --alignment-labels selects.",
    )
    parser.add_argument(
        "--alignment-judges",
        default="human,llm,site",
        help="Comma-separated judges to trust. "
             f"Allowed: {sorted(VALID_JUDGES)}. Default: 'human,llm,site'. "
             "Note: site self-loops are required for the msd_expert<->"
             "msd_lay tier-transition step; excluding 'site' empties that "
             "step.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write the full results as TSV.",
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

    df = run(args.language, alpha=args.alpha, one_sided=args.one_sided,
             alignment_labels=labels, alignment_judges=judges)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, sep="\t", index=False)
    print(format_human(df, args.language, labels=labels, judges=judges))
