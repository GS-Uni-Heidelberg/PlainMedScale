"""Generate all alignment pairs from data/alignment_index.json that still
need LLM evaluation.

Atom = (source, article_id). MSD article_ids are short form "13-53"; their
prof/amateur/short subtree variants are aligned by the MSD website itself
(positional via msd_urls.tsv) and so naturally collapse to a single atom.
Cross-article-id MSD pairs (e.g. 13-53 ↔ 1-112 inside the same cluster) do
come from our DOID-based grouping and ARE included.

For each cluster (per language, all tiers), emit every unordered pair of
distinct atoms. Pairs are deduped across clusters via canonical (a,b)
ordering. Titles are looked up in the source JSON dumps.

`--msd-title` selects which MSD subtree supplies the title shown to the judge.
This changes which alignment question the judge answers, so the two views are
kept as separate outputs rather than merged; see docs/msd_subtree_findings.md.
Non-MSD sources always use the amateur title.

Output TSV columns:
    language, source_a, article_id_a, title_a,
              source_b, article_id_b, title_b,
              cluster_ids, max_tier

Usage (from repo root):
    python alignment/judgment/scripts/generate_full_pairs.py
    python alignment/judgment/scripts/generate_full_pairs.py --msd-title amateur
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RELEASE_DIR = DATA_DIR / "corpus"

sys.path.insert(0, str(REPO_ROOT / "readability"))
from lib import corpus  # noqa: E402

TIER_RANK = {"4_tier": 4, "3_tier": 3, "2_tier": 2}


def build_title_index(msd_title: str) -> dict[tuple[str, str, str], str]:
    """Return {(source, lang, article_id_short): title} where article_id_short
    is the bare form used in alignment_index.json (e.g. "13-53" for MSD,
    "611" for gesund, "749541" for apoum, hash for nhs).

    MSD professional titles preserve article identity; amateur titles collapse
    to chapter-level names (all 6 chapter-9 sub-articles share "Gastritis") but
    describe article scope better for lay-titled umbrella articles elsewhere.
    Other sources only have an amateur subtree.
    """
    index: dict[tuple[str, str, str], str] = {}

    for split, rec in corpus.iter_records(RELEASE_DIR, subtree="amateur"):
        if split.source == "msd":
            continue
        title = (rec.get("title") or "").strip()
        if title:
            index[(split.source, split.language, str(rec["instance_id"]))] = title

    msd_titles: dict[tuple[str, str], dict[str, str]] = {}
    for split, rec in corpus.iter_records(RELEASE_DIR, source="msd"):
        if split.subtree not in ("amateur", "professional"):
            continue
        title = (rec.get("title") or "").strip()
        if title:
            key = (split.language, str(rec["instance_id"]))
            msd_titles.setdefault(key, {})[split.subtree] = title

    fallbacks = 0
    other = "professional" if msd_title == "amateur" else "amateur"
    for (lang, iid), by_subtree in msd_titles.items():
        if msd_title in by_subtree:
            index[("msd", lang, iid)] = by_subtree[msd_title]
        elif other in by_subtree:
            index[("msd", lang, iid)] = by_subtree[other]
            fallbacks += 1

    print(f"MSD titles: {msd_title}, {fallbacks} fell back to {other}")
    return index


def iter_clusters(alignment_index: dict):
    """Yield (language, tier_name, cluster) for every cluster in the index."""
    for lang in ("de", "en"):
        for tier_name, clusters in alignment_index[lang].items():
            for cluster in clusters:
                yield lang, tier_name, cluster


def cluster_atoms(cluster: dict, lang: str, titles: dict) -> list[tuple[str, str, str]]:
    """Return list of (source, article_id, title) atoms in this cluster, in a
    deterministic order. Atoms with missing titles are dropped.
    """
    sources = ("msd", "gesund", "apoum") if lang == "de" else ("msd", "nhs")
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        for aid in cluster.get(source, []):
            key = (source, aid)
            if key in seen:
                continue
            seen.add(key)
            title = titles.get((source, lang, aid))
            if title is None:
                continue
            out.append((source, aid, title))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alignment-index", type=Path,
                   default=DATA_DIR / "alignment_index.json")
    p.add_argument("--msd-title", choices=("professional", "amateur"),
                   default="professional")
    p.add_argument("--output", type=Path, default=None,
                   help="Defaults to full_pairs.tsv, or "
                        "full_pairs_msdamateur.tsv when --msd-title amateur.")
    args = p.parse_args()

    if args.output is None:
        stem = "full_pairs" if args.msd_title == "professional" else "full_pairs_msdamateur"
        args.output = REPO_ROOT / "data" / "alignment" / f"{stem}.tsv"

    titles = build_title_index(args.msd_title)
    print(f"Loaded {len(titles)} (source, lang, article_id) → title entries")

    with args.alignment_index.open() as f:
        index = json.load(f)

    # canonical key -> dict with all metadata; cluster_ids accumulate as list
    pairs: dict[tuple[str, str, str, str, str], dict] = {}
    skipped_missing_title = 0

    for lang, tier_name, cluster in iter_clusters(index):
        atoms = cluster_atoms(cluster, lang, titles)
        # Count atoms we would have had if titles were complete (for reporting)
        nominal = sum(len(cluster.get(s, [])) for s in
                      (("msd","gesund","apoum") if lang=="de" else ("msd","nhs")))
        skipped_missing_title += nominal - len(atoms)

        cid = cluster["cluster_id"]
        tier_n = TIER_RANK[tier_name]
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                a, b = atoms[i], atoms[j]
                # canonical order: (source, article_id) ascending
                if (a[0], a[1]) > (b[0], b[1]):
                    a, b = b, a
                key = (lang, a[0], a[1], b[0], b[1])
                entry = pairs.get(key)
                if entry is None:
                    pairs[key] = {
                        "language": lang,
                        "source_a": a[0],
                        "article_id_a": a[1],
                        "title_a": a[2],
                        "source_b": b[0],
                        "article_id_b": b[1],
                        "title_b": b[2],
                        "cluster_ids": [cid],
                        "max_tier": tier_n,
                    }
                else:
                    entry["cluster_ids"].append(cid)
                    if tier_n > entry["max_tier"]:
                        entry["max_tier"] = tier_n

    # write TSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cols = ["language", "source_a", "article_id_a", "title_a",
            "source_b", "article_id_b", "title_b",
            "cluster_ids", "max_tier"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for entry in pairs.values():
            entry = dict(entry)
            entry["cluster_ids"] = ",".join(str(c) for c in sorted(set(entry["cluster_ids"])))
            w.writerow(entry)

    # report
    n_total = len(pairs)
    by_lang: dict[str, int] = {}
    by_combo: dict[tuple[str, str, str], int] = {}
    by_tier: dict[int, int] = {}
    for k, e in pairs.items():
        by_lang[e["language"]] = by_lang.get(e["language"], 0) + 1
        combo = (e["language"], e["source_a"], e["source_b"])
        by_combo[combo] = by_combo.get(combo, 0) + 1
        by_tier[e["max_tier"]] = by_tier.get(e["max_tier"], 0) + 1

    print(f"\nWrote {n_total} pairs to {args.output}")
    print(f"  by language: {by_lang}")
    print(f"  by (lang, source_a, source_b):")
    for combo, n in sorted(by_combo.items()):
        print(f"    {combo}: {n}")
    print(f"  by max_tier: {dict(sorted(by_tier.items(), reverse=True))}")
    if skipped_missing_title:
        print(f"  dropped {skipped_missing_title} atom occurrences with no title in source JSON")


if __name__ == "__main__":
    main()
