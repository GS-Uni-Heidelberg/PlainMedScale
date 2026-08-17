"""Sanity-check generate_full_pairs.py output.

Verifications:
  1a. Canonical-key duplicates  (lang, src_a, id_a, src_b, id_b)  -- should be 0.
  1b. Title-pair duplicates (unordered title pair appearing in >1 row).
  1c. Pairs already present in instance_alignments_labeled.xlsx ("apoum" sheet)
      matched by unordered title pair -- those have gold labels already.
  2.  Multilingual variants -- same (src_a, id_a, src_b, id_b) appearing in
      both DE and EN clusters (the SAME alignment question asked twice with
      DE titles and again with EN titles).

Optionally writes a filtered TSV with the duplicates / overlaps stripped
out (`--write-filtered <path>`).

Usage (from repo root):
    python alignment/judgment/scripts/verify_full_pairs.py \\
        --pairs data/alignment/full_pairs.tsv \\
        --leo-xlsx data/alignment/instance_alignments_labeled.xlsx \\
        --raw-xlsx data/alignment/instance_alignments_raw.xlsx
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "alignment" / "judgment" / "scripts"))
from eval_alignment import load_eval_pairs  # noqa: E402


def load_pairs(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def leo_unordered_titlepairs(leo_xlsx: Path, raw_xlsx: Path, sheet: str) -> set[tuple[str, str]]:
    eval_pairs = load_eval_pairs(leo_xlsx, sheet, raw_file=raw_xlsx)
    return {tuple(sorted([s.strip(), t.strip()])) for (s, t) in eval_pairs.keys()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path,
                   default=REPO_ROOT / "data" / "alignment" / "full_pairs.tsv")
    p.add_argument("--leo-xlsx", type=Path,
                   default=REPO_ROOT / "data" / "alignment" / "instance_alignments_labeled.xlsx")
    p.add_argument("--raw-xlsx", type=Path,
                   default=REPO_ROOT / "data" / "alignment" / "instance_alignments_raw.xlsx")
    p.add_argument("--leo-sheet", default="apoum")
    p.add_argument("--write-filtered", type=Path, default=None,
                   help="If set, write a TSV that drops rows whose unordered "
                        "title-pair already appears in Leo's eval set; also "
                        "collapses DE/EN duplicates of the same article-id pair "
                        "into a single row (keeps DE).")
    args = p.parse_args()

    rows = load_pairs(args.pairs)
    print(f"Loaded {len(rows)} generated pairs from {args.pairs}")

    # --- (1a) canonical-key duplicates ---
    canon_keys = Counter(
        (r["language"], r["source_a"], r["article_id_a"], r["source_b"], r["article_id_b"])
        for r in rows
    )
    dup_canon = {k: c for k, c in canon_keys.items() if c > 1}
    print(f"\n(1a) Canonical-key duplicates: {len(dup_canon)}  (expected 0)")
    for k, c in list(dup_canon.items())[:5]:
        print(f"     x{c}: {k}")

    # --- (1b) title-pair duplicates ---
    titlepairs: Counter = Counter()
    for r in rows:
        titlepairs[tuple(sorted([r["title_a"], r["title_b"]]))] += 1
    dup_titles = Counter({p: c for p, c in titlepairs.items() if c > 1})
    print(f"\n(1b) Unordered title-pairs occurring >1 time: {len(dup_titles)}")
    for p, c in dup_titles.most_common(5):
        print(f"     x{c}: {p}")

    # --- (1c) overlap with Leo's eval xlsx ---
    leo = leo_unordered_titlepairs(args.leo_xlsx, args.raw_xlsx, args.leo_sheet)
    print(f"\nLeo eval set ({args.leo_sheet}): {len(leo)} unordered title-pairs")
    overlap_idx = []
    overlap_lang = Counter()
    for i, r in enumerate(rows):
        if tuple(sorted([r["title_a"], r["title_b"]])) in leo:
            overlap_idx.append(i)
            overlap_lang[r["language"]] += 1
    print(f"(1c) Generated pairs already in Leo's eval set: {len(overlap_idx)}")
    print(f"     by language: {dict(overlap_lang)}")

    # --- (2) multilingual variants of the same article-id pair ---
    by_idpair: dict[tuple, set[str]] = defaultdict(set)
    for r in rows:
        key = (r["source_a"], r["article_id_a"], r["source_b"], r["article_id_b"])
        by_idpair[key].add(r["language"])
    both = [k for k, langs in by_idpair.items() if len(langs) > 1]
    combo_counts = Counter((k[0], k[2]) for k in both)
    print(f"\n(2) Article-id-pairs appearing in BOTH DE and EN: {len(both)}")
    print(f"     by (source_a, source_b): {dict(combo_counts)}")
    de_rows = {(r["source_a"], r["article_id_a"], r["source_b"], r["article_id_b"]): r
               for r in rows if r["language"] == "de"}
    en_rows = {(r["source_a"], r["article_id_a"], r["source_b"], r["article_id_b"]): r
               for r in rows if r["language"] == "en"}
    print("     sample:")
    for k in both[:3]:
        de = de_rows[k]; en = en_rows[k]
        print(f"      [{k[0]} {k[1]} × {k[2]} {k[3]}]")
        print(f"        DE: {de['title_a']!r}  ×  {de['title_b']!r}")
        print(f"        EN: {en['title_a']!r}  ×  {en['title_b']!r}")

    # --- optional filtered TSV ---
    if args.write_filtered is not None:
        leo_idx = set(overlap_idx)
        # 1. Drop Leo-overlap rows
        # 2. Drop EN rows for MSD-MSD article-id-pairs that also appear in DE
        # 3. Collapse remaining rows by (language, unordered title-pair):
        #    keep the first occurrence (canonical-key sorted). Identical
        #    title-pairs in the same language yield identical LLM judgments.
        en_dropped = 0
        stage_kept: list[dict] = []
        for i, r in enumerate(rows):
            if i in leo_idx:
                continue
            key = (r["source_a"], r["article_id_a"], r["source_b"], r["article_id_b"])
            if r["language"] == "en" and "de" in by_idpair[key] and "en" in by_idpair[key]:
                en_dropped += 1
                continue
            stage_kept.append(r)

        # Dedup by unordered title-pair (language-agnostic): identical
        # title strings produce identical LLM judgments regardless of which
        # language cluster they came from. Sort puts DE first so DE wins
        # ties; within a language the lex-smallest canonical-key wins.
        stage_kept.sort(
            key=lambda r: (r["language"], r["source_a"], r["article_id_a"],
                           r["source_b"], r["article_id_b"])
        )
        seen_titlepairs: set[tuple[str, str]] = set()
        keep: list[dict] = []
        title_dropped = 0
        for r in stage_kept:
            tp = tuple(sorted([r["title_a"], r["title_b"]]))
            if tp in seen_titlepairs:
                title_dropped += 1
                continue
            seen_titlepairs.add(tp)
            keep.append(r)

        cols = list(rows[0].keys())
        args.write_filtered.parent.mkdir(parents=True, exist_ok=True)
        with args.write_filtered.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            w.writerows(keep)
        print(f"\nFiltered TSV: {args.write_filtered}")
        print(f"  dropped {len(leo_idx)} Leo-overlap, "
              f"{en_dropped} EN multilingual-twin, "
              f"{title_dropped} same-title-pair duplicate rows")
        print(f"  remaining: {len(keep)}")


if __name__ == "__main__":
    main()
