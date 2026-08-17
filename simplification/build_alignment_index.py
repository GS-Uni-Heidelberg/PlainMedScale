"""
Build an alignment index from instance_alignments_raw.xlsx.

Parses the MSD-anchored sheet (one row = one topic cluster), splits the
semicolon-separated ID cells into lists, and groups topics by tier coverage
per language. Writes a JSON overview.

Tier definitions:
  DE tier 1 = msd.de.professional   (always present for MSD anchors)
  DE tier 2 = msd.de.amateur        (always present for MSD anchors)
  DE tier 3 = gesund_bund.de.amateur
  DE tier 4 = apoum.de.amateur

  EN tier 1 = msd.en.professional   (always present)
  EN tier 2 = msd.en.amateur        (always present)
  EN tier 3 = nhs.en.amateur

A topic is "k-tier" if it has non-empty IDs for exactly k of the possible
tiers (MSD prof + MSD lay always count, cross-source tiers count only if
the ID list is non-empty).

Usage (from repo root):
    python simplification/build_alignment_index.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd


def split_ids(cell) -> list[str]:
    if not isinstance(cell, str):
        return []
    return [e.strip() for e in cell.split(";") if e.strip()]


def build_topic(row_idx: int, row: pd.Series) -> dict:
    return {
        "cluster_id": row_idx,
        "msd":    split_ids(row.get("MSD_id")),
        "gesund": split_ids(row.get("gesund_id")),
        "nhs":    split_ids(row.get("NHS_id")),
        "apoum":  split_ids(row.get("apoum_id")),
        "titles": {
            "msd":    split_ids(row.get("MSD_title")),
            "gesund": split_ids(row.get("gesund_title")),
            "nhs":    split_ids(row.get("NHS_title")),
            "apoum":  split_ids(row.get("apoum_title")),
        },
    }


def classify_de(t: dict) -> str | None:
    has_msd    = bool(t["msd"])       # tiers 1+2
    has_gesund = bool(t["gesund"])    # tier 3
    has_apoum  = bool(t["apoum"])     # tier 4
    if not has_msd:
        return None
    n = 2 + int(has_gesund) + int(has_apoum)
    return {4: "4_tier", 3: "3_tier", 2: "2_tier"}[n]


def classify_en(t: dict) -> str | None:
    has_msd = bool(t["msd"])
    has_nhs = bool(t["nhs"])
    if not has_msd:
        return None
    n = 2 + int(has_nhs)
    return {3: "3_tier", 2: "2_tier"}[n]


def de_topic(t: dict) -> dict:
    return {
        "cluster_id": t["cluster_id"],
        "msd":    t["msd"],
        "gesund": t["gesund"],
        "apoum":  t["apoum"],
        "titles": {k: t["titles"][k] for k in ("msd", "gesund", "apoum")},
    }


def en_topic(t: dict) -> dict:
    return {
        "cluster_id": t["cluster_id"],
        "msd": t["msd"],
        "nhs": t["nhs"],
        "titles": {k: t["titles"][k] for k in ("msd", "nhs")},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--xlsx", type=Path,
        default=Path("data/alignment/instance_alignments_raw.xlsx"),
    )
    p.add_argument("--sheet", default="MSD")
    p.add_argument("--output", type=Path, default=Path("data/alignment_index.json"))
    args = p.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name=args.sheet)

    index: dict = {
        "source": str(args.xlsx),
        "sheet": args.sheet,
        "n_rows": len(df),
        "de": {"4_tier": [], "3_tier": [], "2_tier": []},
        "en": {"3_tier": [], "2_tier": []},
    }

    for i, row in df.iterrows():
        t = build_topic(i, row)

        de_bucket = classify_de(t)
        if de_bucket is not None:
            index["de"][de_bucket].append(de_topic(t))

        en_bucket = classify_en(t)
        if en_bucket is not None:
            index["en"][en_bucket].append(en_topic(t))

    index["summary"] = {
        "de": {k: len(v) for k, v in index["de"].items()},
        "en": {k: len(v) for k, v in index["en"].items()},
        "bilingual_de4_en3": sum(
            1 for t_de in index["de"]["4_tier"]
            if any(t_en["cluster_id"] == t_de["cluster_id"]
                   for t_en in index["en"]["3_tier"])
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output}")
    print(json.dumps(index["summary"], indent=2))


if __name__ == "__main__":
    main()
