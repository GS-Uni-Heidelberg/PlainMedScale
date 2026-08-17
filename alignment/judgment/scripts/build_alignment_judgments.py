"""Build a unified alignment-judgments JSON combining human and LLM labels.

Policy: **title-identity expansion**. A judgment is fundamentally a claim
about a (title_a, title_b) pair — the *condition* on each side, not the
specific (source, article_id) tuple it was recorded against. So both
sides are expanded source-and-language-blind across every article that
carries an exact-match title string. Cross-source title collisions
(e.g. "Listeriosis" → MSD + gesund + NHS) become a feature: a single
annotation produces one merged pair per cross-product cell.

This also retroactively re-applies LLM verdicts to pairs that
`verify_full_pairs.py` had dropped via title-pair dedup before judging:
if the LLM judged "Listeriosis × Listeriose" once on `msd × apoum`, that
verdict now flows to `gesund × apoum` and `nhs × apoum` automatically.

Inputs:
  - data/alignment/instance_alignments_labeled.xlsx, sheet 'apoum'
      Human cluster rows. MSD ids in the xlsx are SYSTEMATICALLY UNRELIABLE
      (positional-pairing bug), so we ignore them; resolution is purely
      title-based. apoum_id is also ignored — apoum_title is sufficient
      to resolve via the title index.
  - alignment/judgment/results/full_align_judge_mini/predictions.tsv
      LLM judgments. We use `title_a` / `title_b` from the row (not the
      article_ids) so the same expansion logic applies.

Output JSON (normalized format):
  {
    "articles": {"{source}:{article_id}": {"source", "titles": {"de"?, "en"?}}, ...},
    "pairs":    [{"a", "b", "label", "judge", "model", "a_view"?, "b_view"?}, ...]
  }

`titles` is a dict keyed by language: monolingual sources (apoum/NHS) have a
single key; bilingual sources (gesund.bund, MSD) have both. Title-identity
expansion at resolve time is language-blind, so a pair never needs to record
which lang the judge read — provenance lives only in `judge` / `model`.

Pairs with label `falsch` are dropped (positive evidence only).
After expansion, output pairs are deduped on (a, b, label, judge, model).

Usage (from repo root):
    python alignment/judgment/scripts/build_alignment_judgments.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RELEASE_DIR = DATA_DIR / "corpus"

sys.path.insert(0, str(REPO_ROOT / "readability"))
from lib import corpus  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


# Manual overrides for title strings in Leo's xlsx that don't match any
# current JSON title (typos / drift / removed articles). Key is the
# post-strip Leo title scoped by source column (`(source, title)`). Value:
#   - str: single article_id_short
#   - list[str]: multiple ids (annotation applies to each)
#   - None: skip this title entirely (article gone from current snapshot)
HARDCODED_OVERRIDES: dict[tuple[str, str], "str | list[str] | None"] = {
    # Two MSD titles concatenated with `,` instead of `;` (row 105)
    ("msd", "Overview of Salmonella Infections, Nontyphoidal Salmonella Infections"):
        ["16-26", "16-35"],
    # Typo "Septic" should be "Sepsis;" (row 169) → Neonatal Sepsis
    ("msd", "Septic  Neonatal Sepsis"): "21-75",
    # Trailing comma (row 176)
    ("gesund", "COVID-19,"): "25",
    # Concatenated EN gesund title (row 178) → gesund 968 is the combined
    # COVID/cold/flu overview article
    ("gesund", "Flu (influenza),  cold, flu – symptoms at a glance"): "968",
    # Apoum articles no longer in the 20251103 snapshot — skip
    ("apoum", "Krätze"): None,                    # row 145
    ("apoum", "Speiseröhrenentzündung"): None,    # rows 52-54 cluster
}

LABEL_NORM = {
    "richtig": "richtig",
    "richtig_1": "richtig",
    "richtig_2": "richtig",
    "Spezialisierung": "spezialisierung",
    "spezialisierung": "spezialisierung",
    "Generalisierung": "generalisierung",
    "generalisierung": "generalisierung",
    "Falsch": "falsch",
    "falsch": "falsch",
    "False": "falsch",
}
VALID_LABELS = {"richtig", "spezialisierung", "generalisierung", "falsch"}

# Map Leo's xlsx column suffix -> internal source key.
SOURCE_COL_MAP = {"MSD": "msd", "gesund": "gesund", "NHS": "nhs"}

# Primary subtree per source for the article meta's canonical title.
# Title-identity expansion at *resolve* time uses every available title across
# subtrees & langs; PRIMARY_SUBTREE only controls which subtree's title we
# store per language in `articles[...]["titles"]`.
PRIMARY_SUBTREE = {
    "apoum":  "amateur",
    "gesund": "amateur",
    "nhs":    "amateur",
    "msd":    "professional",
}


def build_article_meta() -> dict[tuple[str, str], dict]:
    """{(source, article_id_short): {source, titles: {lang: title, ...}}}.

    Loops over both languages per source; records the title from each lang
    that carries non-empty content for the source's primary subtree. Falls
    back to other subtrees within the same lang if the primary is missing.
    Monolingual sources yield a one-key titles map; bilingual sources
    (gesund.bund, MSD) yield a two-key map.
    """
    subtree_order = ("amateur", "professional", "short")
    meta: dict[tuple[str, str], dict] = {}

    for source in ("msd", "gesund", "apoum", "nhs"):
        primary_sub = PRIMARY_SUBTREE[source]
        sub_search = (primary_sub, *(s for s in subtree_order if s != primary_sub))
        # {(instance_id, lang): {subtree: record}}
        by_article: dict[str, dict[str, dict[str, dict]]] = {}
        for split, rec in corpus.iter_records(RELEASE_DIR, source=source):
            if not corpus.has_text(rec):
                continue
            iid = str(rec["instance_id"])
            by_article.setdefault(iid, {}).setdefault(split.language, {})[split.subtree] = rec

        for iid, per_lang in by_article.items():
            titles: dict[str, str] = {}
            for lang in ("de", "en"):
                for sub_name in sub_search:
                    rec = (per_lang.get(lang) or {}).get(sub_name)
                    if not rec:
                        continue
                    title = (rec.get("title") or "").strip()
                    if title:
                        titles[lang] = title
                        break
            if titles:
                meta[(source, iid)] = {"source": source, "titles": titles}
    return meta


def build_views_index() -> dict[tuple[str, str], list[tuple[str, str]]]:
    """{(source, article_id_short): [(lang, subtree)]}.

    Every (lang, subtree) combination that has non-empty plain_text for an
    article. Used to emit implicit "site" alignment pairs: prof/amateur/short
    of the same MSD article, DE/EN of the same bilingual article, etc. —
    these are the alignments the source itself guarantees (same article in
    different complexity levels or languages).
    """
    out: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for split, rec in corpus.iter_records(RELEASE_DIR):
        if not corpus.has_text(rec):
            continue
        out[(split.source, str(rec["instance_id"]))].append(
            (split.language, split.subtree))
    return out


def generate_site_pairs(
    views_index: dict[tuple[str, str], list[tuple[str, str]]],
) -> list[tuple[str, str, str, str, str, str, str]]:
    """Yield (source, aid, source, aid, "richtig", a_view, b_view) for each
    pair of distinct views within the same article. a_view < b_view lexically.
    """
    out: list[tuple[str, str, str, str, str, str, str]] = []
    for (source, aid), views in views_index.items():
        if len(views) < 2:
            continue
        view_tags = sorted({f"{lang}_{sub}" for (lang, sub) in views})
        for i in range(len(view_tags)):
            for j in range(i + 1, len(view_tags)):
                out.append((source, aid, source, aid, "richtig",
                            view_tags[i], view_tags[j]))
    return out


def build_title_to_articles() -> dict[str, set[tuple[str, str]]]:
    """{title: {(source, article_id_short)}}.

    A title is mapped to every article that carries it as the title of any
    subtree (amateur/professional/short) in any language (de/en). This is the
    *source-and-language-blind* index that drives title-identity expansion.
    """
    out: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for split, rec in corpus.iter_records(RELEASE_DIR):
        title = (rec.get("title") or "").strip()
        if not title:
            continue
        out[title].add((split.source, str(rec["instance_id"])))
    return out


def parse_title_list(v) -> list[str]:
    if pd.isna(v):
        return []
    return [x.strip() for x in str(v).split(";") if x.strip()]


def expand_title(
    title: str,
    title_to_articles: dict[str, set[tuple[str, str]]],
    override_key: tuple[str, str] | None = None,
) -> set[tuple[str, str]]:
    """Expand a title string to the set of (source, article_id_short) tuples
    that carry it. Honors HARDCODED_OVERRIDES when `override_key` is provided.
    """
    if override_key is not None and override_key in HARDCODED_OVERRIDES:
        ov = HARDCODED_OVERRIDES[override_key]
        if ov is None:
            return set()
        ids = ov if isinstance(ov, list) else [ov]
        override_source = override_key[0]
        return {(override_source, aid) for aid in ids}
    return set(title_to_articles.get(title, set()))


def load_human_pairs(
    xlsx_path: Path,
    title_to_articles: dict[str, set[tuple[str, str]]],
) -> tuple[list[tuple[str, str, str, str, str]], list[dict]]:
    """Return (pairs, unresolved_records).

    pairs: list of (a_source, a_aid, b_source, b_aid, label), unordered.
    unresolved_records: rows/sides we couldn't resolve.

    For each Leo annotation row:
      1. Normalize the label.
      2. Expand every apoum_title via title_to_articles (source-blind).
      3. For each source column (MSD / gesund / NHS), expand each non-empty
         source title source-blind (the column is only used as the
         HARDCODED_OVERRIDES key — the expansion itself ignores source).
      4. Emit the cross-product (src_target × apoum_target), skipping
         self-pairs. apoum_id from the xlsx is NOT used.
    """
    df = pd.read_excel(xlsx_path, sheet_name="apoum")

    pairs: list[tuple[str, str, str, str, str]] = []
    unresolved: list[dict] = []

    for row_idx, row in df.iterrows():
        raw_label = row.get("comments")
        label_key = raw_label if isinstance(raw_label, (str, bool)) else None
        label = LABEL_NORM.get(label_key)
        if label not in VALID_LABELS:
            continue

        apoum_titles = parse_title_list(row.get("apoum_title"))
        if not apoum_titles:
            unresolved.append({"row": row_idx, "reason": "no apoum_title",
                               "label": label})
            continue

        apoum_targets: set[tuple[str, str]] = set()
        for ap_t in apoum_titles:
            targets = expand_title(ap_t, title_to_articles, override_key=("apoum", ap_t))
            if not targets and ("apoum", ap_t) not in HARDCODED_OVERRIDES:
                unresolved.append({"row": row_idx, "side": "apoum",
                                   "title": ap_t, "label": label,
                                   "reason": "title not in JSON title index"})
            apoum_targets |= targets

        if not apoum_targets:
            continue

        for col_prefix, src_key in SOURCE_COL_MAP.items():
            src_titles = parse_title_list(row.get(f"{col_prefix}_title"))
            if not src_titles:
                continue
            for src_t in src_titles:
                src_targets = expand_title(src_t, title_to_articles,
                                           override_key=(src_key, src_t))
                if not src_targets and (src_key, src_t) not in HARDCODED_OVERRIDES:
                    unresolved.append({"row": row_idx, "side": src_key,
                                       "title": src_t, "label": label,
                                       "reason": "title not in JSON title index"})
                    continue
                for (sa_src, sa_aid) in src_targets:
                    for (ta_src, ta_aid) in apoum_targets:
                        if (sa_src, sa_aid) == (ta_src, ta_aid):
                            continue  # self-pair
                        pairs.append((sa_src, sa_aid, ta_src, ta_aid, label))

    return pairs, unresolved


def load_llm_pairs(
    predictions_path: Path,
    title_to_articles: dict[str, set[tuple[str, str]]],
) -> tuple[list[tuple[str, str, str, str, str]], int]:
    """Return (pairs, n_skipped_unknown_title).

    For each LLM prediction row, expand title_a and title_b source-blind and
    emit the cross-product. Self-pairs skipped.
    """
    pairs: list[tuple[str, str, str, str, str]] = []
    n_skipped = 0
    with predictions_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            title_a = r["title_a"].strip()
            title_b = r["title_b"].strip()
            pred = r["pred"]
            a_targets = title_to_articles.get(title_a, set())
            b_targets = title_to_articles.get(title_b, set())
            if not a_targets or not b_targets:
                n_skipped += 1
                continue
            for (sa, aa) in a_targets:
                for (sb, ab) in b_targets:
                    if (sa, aa) == (sb, ab):
                        continue
                    pairs.append((sa, aa, sb, ab, pred))
    return pairs, n_skipped


def key(source: str, aid: str) -> str:
    return f"{source}:{aid}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--leo-xlsx", type=Path,
                   default=REPO_ROOT / "data" / "alignment" / "instance_alignments_labeled.xlsx")
    p.add_argument("--llm-predictions", type=Path,
                   default=REPO_ROOT / "alignment" / "judgment" / "results"
                           / "full_align_judge_mini" / "predictions.tsv")
    p.add_argument("--llm-model", default="gpt-5.4-mini-2026-03-17",
                   help="Stored as 'model' on each LLM pair.")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "data" / "alignment" / "alignment_judgments.json")
    p.add_argument("--keep-falsch", action="store_true",
                   help="Include falsch pairs (default: drop, positive evidence only).")
    p.add_argument("--unresolved-report", type=Path,
                   default=REPO_ROOT / "data" / "alignment" / "alignment_unresolved.tsv",
                   help="Where to write the list of Leo annotations that didn't resolve.")
    args = p.parse_args()

    log.info("Building article metadata index from data/*.json")
    article_meta = build_article_meta()
    log.info(f"  {len(article_meta)} (source, article_id) entries")

    title_to_articles = build_title_to_articles()
    log.info(f"  built source-blind title→articles index "
             f"({len(title_to_articles)} distinct titles)")

    log.info(f"Loading human pairs from {args.leo_xlsx}")
    human_pairs, unresolved = load_human_pairs(args.leo_xlsx, title_to_articles)
    log.info(f"  expanded to {len(human_pairs)} (a, b, label) human-pair occurrences "
             f"({len({(a,b,c,d,l) for a,b,c,d,l in human_pairs})} unique), "
             f"{len(unresolved)} unresolved annotations")

    log.info(f"Loading LLM pairs from {args.llm_predictions}")
    llm_pairs, n_skipped_llm = load_llm_pairs(args.llm_predictions, title_to_articles)
    log.info(f"  expanded to {len(llm_pairs)} LLM-pair occurrences "
             f"({len({(a,b,c,d,l) for a,b,c,d,l in llm_pairs})} unique); "
             f"skipped {n_skipped_llm} predictions with title not in JSON")

    log.info("Building views index for implicit site alignments")
    views_index = build_views_index()
    site_pairs = generate_site_pairs(views_index)
    log.info(f"  {len(site_pairs)} site-alignment pairs across "
             f"{sum(1 for v in views_index.values() if len(v) >= 2)} articles with ≥2 views")

    # --- Build articles dict + pairs list ---
    articles: dict[str, dict] = {}
    pairs_out: list[dict] = []
    skipped_unknown_article = 0
    seen: set[tuple] = set()

    def ensure_article(source: str, aid: str) -> str | None:
        k = key(source, aid)
        if k in articles:
            return k
        m = article_meta.get((source, aid))
        if m is None:
            return None
        articles[k] = {"source": source, "titles": dict(m["titles"])}
        return k

    def emit(a_src, a_aid, b_src, b_aid, label, judge, model,
             a_view=None, b_view=None):
        nonlocal skipped_unknown_article
        if not args.keep_falsch and label == "falsch":
            return
        ka = ensure_article(a_src, a_aid)
        kb = ensure_article(b_src, b_aid)
        if ka is None or kb is None:
            skipped_unknown_article += 1
            return
        # For site pairs ka==kb; preserve the given view order. Otherwise
        # canonicalize on article key (lexicographic). Directional labels
        # (spezialisierung / generalisierung) are defined as "a is subtype
        # of b" / "b is subtype of a", so swapping a<->b must flip them.
        if ka == kb:
            a, b = ka, kb
            va, vb = a_view, b_view
        elif ka <= kb:
            a, b = ka, kb
            va, vb = a_view, b_view
        else:
            a, b = kb, ka
            va, vb = b_view, a_view
            if label == "spezialisierung":
                label = "generalisierung"
            elif label == "generalisierung":
                label = "spezialisierung"
        k_ = (a, b, label, judge, model, va, vb)
        if k_ in seen:
            return
        seen.add(k_)
        pair = {"a": a, "b": b, "label": label, "judge": judge, "model": model}
        if va is not None:
            pair["a_view"] = va
        if vb is not None:
            pair["b_view"] = vb
        pairs_out.append(pair)

    for (a_src, a_aid, b_src, b_aid, label) in human_pairs:
        emit(a_src, a_aid, b_src, b_aid, label, "human", "leo")

    for (a_src, a_aid, b_src, b_aid, label) in llm_pairs:
        emit(a_src, a_aid, b_src, b_aid, label, "llm", args.llm_model)

    for (a_src, a_aid, b_src, b_aid, label, va, vb) in site_pairs:
        emit(a_src, a_aid, b_src, b_aid, label, "site", "implicit",
             a_view=va, b_view=vb)

    if skipped_unknown_article:
        log.warning(f"Skipped {skipped_unknown_article} pair occurrences whose "
                    f"article_ids are not in data/*.json")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"articles": articles, "pairs": pairs_out}, f,
                  ensure_ascii=False, indent=2)
    log.info(f"Wrote {len(articles)} articles, {len(pairs_out)} pairs to {args.output}")

    by_judge = Counter(p["judge"] for p in pairs_out)
    by_label = Counter((p["judge"], p["label"]) for p in pairs_out)
    log.info(f"By judge: {dict(by_judge)}")
    log.info("By (judge, label):")
    for k_, v in sorted(by_label.items()):
        log.info(f"  {k_}: {v}")

    if unresolved:
        with args.unresolved_report.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["row", "side", "title", "label",
                                              "candidates", "reason"],
                               delimiter="\t")
            w.writeheader()
            for u in unresolved:
                w.writerow({
                    "row": u.get("row"),
                    "side": u.get("side", ""),
                    "title": u.get("title", ""),
                    "label": u.get("label", ""),
                    "candidates": ";".join(u.get("candidates", []) or []),
                    "reason": u.get("reason", ""),
                })
        log.info(f"Unresolved annotations -> {args.unresolved_report}")


if __name__ == "__main__":
    main()
