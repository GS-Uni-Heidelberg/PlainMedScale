"""
Emit a small LaTeX table of corpus composition: per-source/tier article
counts plus, for each row, how many articles are aligned with k=1, 2, or 3
other tiers in the canonical 4-tier chain (msd_short excluded).

The canonical chains match make_metrics_table.SOURCES_* and
lib.stats_direction.SOURCES_*:

    DE: apoum, gesund, msd_lay, msd_prof  (msd_short omitted)
    EN: nhs,   gesund, msd_lay, msd_prof  (msd_short omitted)

An MSD article is in the msd_lay (resp. msd_prof) row iff its crawled
JSON entry has the language's `amateur` (resp. `professional`) view.
By construction every MSD article has both views in both languages, so
the msd_prof <-> msd_lay site self-loop always counts as one tier partner
for any MSD-row article. Cross-language pairs are ignored: alignment to
gesund (EN) doesn't count for the DE row.

Usage:
    PYTHONPATH=readability python readability/reporting/make_corpus_table.py
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

from lib import corpus


RELEASE_DIR = Path("data/corpus")
ALIGNMENT_JSON = Path("data/alignment/alignment_judgments.json")

VALID_LABELS = {"richtig", "spezialisierung", "generalisierung"}
VALID_JUDGES = {"human", "llm", "site"}

# Display order per language (matches make_metrics_table.SOURCES_*,
# high -> low readability). msd_short intentionally excluded.
CHAIN_DE = ["apoum", "gesund", "msd_lay", "msd_prof"]
CHAIN_EN = ["nhs",   "gesund", "msd_lay", "msd_prof"]
CHAIN = {"de": CHAIN_DE, "en": CHAIN_EN}

# A tier is one (source, subtree) view of the corpus. msd/short has no tier.
TIER_TO_SOURCE_VIEW = {
    "apoum":    ("apoum",  "amateur"),
    "nhs":      ("nhs",    "amateur"),
    "gesund":   ("gesund", "amateur"),
    "msd_lay":  ("msd",    "amateur"),
    "msd_prof": ("msd",    "professional"),
}
TIER_FOR_VIEW = {view: tier for tier, view in TIER_TO_SOURCE_VIEW.items()}

ROW_LABELS = {
    "apoum":    "\\makecell[l]{Apotheken \\\\ Umschau}",
    "nhs":      "NHS",
    "gesund":   "GesundBund",
    "msd_lay":  "MSD Consumer",
    "msd_prof": "MSD Prof.",
}

LANG_LABEL = {"de": "German", "en": "English"}


def _short_id(source: str, article_id: str) -> str:
    """Reduce a release article_id to the canonical short id used after
    `<source>:` in alignment_judgments.json (mirrors
    lib.stats_direction._short_article_id)."""
    if source in ("apoum", "nhs"):
        return article_id
    return article_id.split("_", 1)[0]


def build_tier_membership(release_dir: Path = RELEASE_DIR
                          ) -> dict[tuple[str, str], set[str]]:
    """Returns {(lang, tier): set of canonical 'source:short_id' strings}.

    Membership is determined by which release files an article appears in —
    not by whether it appears in alignment_judgments. So this is the corpus N,
    not the aligned N.
    """
    members: dict[tuple[str, str], set[str]] = {}
    for split in corpus.SPLITS:
        tier = TIER_FOR_VIEW.get((split.source, split.subtree))
        if tier is None:  # msd/short is not part of the 4-tier chain
            continue
        for rec in corpus.load(release_dir, split):
            art = rec.get("article_id")
            if not art:
                continue
            node = f"{split.source}:{_short_id(split.source, art)}"
            members.setdefault((split.language, tier), set()).add(node)
    return members


def load_pairs(
    labels: set[str], judges: set[str], json_path: Path = ALIGNMENT_JSON,
) -> tuple[dict, list[dict]]:
    """Mirror of lib.stats_direction.load_pairs but inline here so the module
    is independent."""
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
            "a":        p["a"],
            "b":        p["b"],
            "a_source": a_meta["source"],
            "b_source": b_meta["source"],
            "a_view":   p.get("a_view"),
            "b_view":   p.get("b_view"),
        })
    return articles, out


def node_to_tiers(
    node: str, lang: str, members: dict[tuple[str, str], set[str]],
) -> set[str]:
    """Which tiers in `lang` does this canonical 'source:short_id'
    belong to? Cross-language nodes return an empty set."""
    return {tier for tier in CHAIN[lang]
            if node in members.get((lang, tier), set())}


def compute_partner_tier_counts(
    lang: str, members: dict[tuple[str, str], set[str]],
    pairs: list[dict],
) -> dict[str, Counter]:
    """For each tier in `lang`'s chain, return {k -> #articles in that
    tier aligned with exactly k OTHER tiers in the same chain}."""
    chain = CHAIN[lang]
    chain_set = set(chain)
    # partners[node] = set of tiers this node is aligned to in `lang`
    partners: dict[str, set[str]] = {}

    # Seed: every MSD article in this language gets an automatic
    # partner = the *other* MSD tier (site self-loop). True for all
    # 1594 MSD articles since every one has both amateur+professional
    # views in both languages.
    for tier in ("msd_lay", "msd_prof"):
        other = "msd_prof" if tier == "msd_lay" else "msd_lay"
        for node in members.get((lang, tier), set()):
            partners.setdefault(node, set()).add(other)

    # Cross-source pairs: contribute the partner's tier(s) in this lang
    # to each endpoint's partner set (only when both endpoints have a
    # tier in this language).
    for p in pairs:
        # Skip intra-MSD self-loops; we already handled them above.
        if p["a"] == p["b"]:
            continue
        a_tiers = node_to_tiers(p["a"], lang, members)
        b_tiers = node_to_tiers(p["b"], lang, members)
        if not a_tiers or not b_tiers:
            continue
        # `a`'s row gains b's tiers (minus a's own); symmetric for `b`.
        for at in a_tiers:
            partners.setdefault(p["a"], set()).update(
                t for t in b_tiers if t in chain_set and t != at
            )
        for bt in b_tiers:
            partners.setdefault(p["b"], set()).update(
                t for t in a_tiers if t in chain_set and t != bt
            )

    out: dict[str, Counter] = {}
    for tier in chain:
        c: Counter = Counter()
        for node in members.get((lang, tier), set()):
            # An MSD article sits in BOTH msd_lay and msd_prof tiers and
            # therefore accumulates its own row tier in `partners[node]`
            # via the symmetric seed/cross-source updates. Subtract the
            # row tier so k counts only the OTHER aligned tiers.
            k = len(partners.get(node, set()) - {tier})
            c[k] += 1
        out[tier] = c
    return out


def build_table(labels: set[str], judges: set[str]) -> str:
    """Emit a compact LaTeX table with the language as a rotated
    multirow on the left edge and sources sorted hardest-first within
    each language. Requires \\usepackage{multirow}, graphicx, makecell."""
    members = build_tier_membership()
    _, pairs = load_pairs(labels, judges)
    counts_per_lang = {
        lang: compute_partner_tier_counts(lang, members, pairs)
        for lang in ("de", "en")
    }

    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append("\\begin{tabular}{cl rrrr}")
    lines.append("\\toprule")
    lines.append(" &  &  & \\multicolumn{3}{c}"
                 "{Aligned w/ $k$ tiers} \\\\")
    lines.append("\\cmidrule(lr){4-6}")
    lines.append(" & \\makecell[l]{Source\\\\\\footnotesize"
                 "(prof.~$\\to$ consumer)} & $N$ & 1 & 2 & 3 \\\\")
    lines.append("\\midrule")

    for i, lang in enumerate(("de", "en")):
        # CHAIN_* is consumer→professional (high→low readability). Reverse
        # it so the table reads hardest→easiest within each language.
        tiers = list(reversed(CHAIN[lang]))
        if i > 0:
            lines.append("\\midrule")
        n_rows = len(tiers)
        for j, tier in enumerate(tiers):
            n = len(members.get((lang, tier), set()))
            c = counts_per_lang[lang][tier]
            lang_cell = (
                f"\\multirow{{{n_rows}}}{{*}}"
                f"{{\\rotatebox[origin=c]{{90}}{{{LANG_LABEL[lang]}}}}}"
                if j == 0 else ""
            )
            cells = [
                lang_cell,
                ROW_LABELS[tier],
                str(n),
                str(c.get(1, 0)),
                str(c.get(2, 0)),
                str(c.get(3, 0)),
            ]
            lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    # cols 1/2/3 = articles aligned with that many of the other 3 tiers
    lines.append("\\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to write the .tex table. Defaults to "
             "latex_tables/corpus_table.tex. Use --stdout to print.",
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="Print to stdout instead of writing to --output.",
    )
    parser.add_argument(
        "--alignment-labels", default="richtig",
        help=f"Comma-separated label set. Allowed: {sorted(VALID_LABELS)}. "
             "Default: 'richtig'.",
    )
    parser.add_argument(
        "--include-spez-gen", action="store_true",
        help="Convenience: add 'spezialisierung' and 'generalisierung' "
             "to --alignment-labels.",
    )
    parser.add_argument(
        "--alignment-judges", default="human,llm,site",
        help=f"Comma-separated judges. Allowed: {sorted(VALID_JUDGES)}. "
             "Default: 'human,llm,site'.",
    )
    args = parser.parse_args()

    labels = {s.strip() for s in args.alignment_labels.split(",") if s.strip()}
    bad = labels - VALID_LABELS
    if bad:
        parser.error(f"unknown --alignment-labels values: {sorted(bad)}")
    if args.include_spez_gen:
        labels |= {"spezialisierung", "generalisierung"}
    judges = {s.strip() for s in args.alignment_judges.split(",") if s.strip()}
    bad = judges - VALID_JUDGES
    if bad:
        parser.error(f"unknown --alignment-judges values: {sorted(bad)}")

    table = build_table(labels, judges)
    if args.stdout:
        print(table)
    else:
        output = args.output or (Path("latex_tables") / "corpus_table.tex")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table, encoding="utf-8")
        print(f"wrote {output}")
