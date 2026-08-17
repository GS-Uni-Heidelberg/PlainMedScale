"""
Pipeline that counts jargon words per 100 words using the SimpleJargonDetector.

Uses SoMaJo tokenization with default preprocessing, OR > 20 threshold,
and ref_freq_threshold <= 0.001 to identify jargon terms.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# jargon detector lives at readability/jargon/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "jargon" / "detection"))
from simple_jargon_detector import SimpleJargonDetector

from lib import corpus
from lib.run_metadata import write_info

KEYNESS_DIR = Path(__file__).resolve().parent.parent / "jargon" / "keys" / "somajo"

COMPARISONS = {
    "de": "drks-v-deleipzig",
    "en": "pubmed-v-enleipzig",
}

OR_THRESHOLD = 20.0
REF_FREQ_THRESHOLD = 0.001

FIELDNAMES = ["article_id", "language", "total_tokens", "jargon_count", "jargon_per_100"]


def jargon_for_data(
    items,
    total: int | None = None,
    language="de",
    output_tsv_path: str | Path | None = None,
    overwrite: bool = False,
):
    results = []

    detector = SimpleJargonDetector(
        keyness_dir=KEYNESS_DIR,
        comparison=COMPARISONS[language],
        preprocessing="default",
    )

    writer = None
    tsv_file = None
    processed_ids = set()

    if output_tsv_path is not None:
        output_tsv_path = Path(output_tsv_path)
        output_tsv_path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite and output_tsv_path.exists():
            output_tsv_path.unlink()
            print(f"Overwriting {output_tsv_path} (removed existing file).")

        file_exists = output_tsv_path.exists() and output_tsv_path.stat().st_size > 0

        if file_exists:
            try:
                existing = pd.read_csv(output_tsv_path, sep="\t", usecols=["article_id"], dtype=str)
                processed_ids = set(existing["article_id"].astype(str))
                print(f"Found {len(processed_ids)} already processed entries.")
            except Exception as e:
                print(f"Warning: could not read existing TSV ({e}); continuing without skip list.")

        tsv_file = open(output_tsv_path, "a", encoding="utf-8", newline="")
        writer = csv.DictWriter(tsv_file, fieldnames=FIELDNAMES, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writeheader()

    try:
        for text, id_ in tqdm(items, total=total):
            id_str = str(id_)
            if id_str in processed_ids:
                continue

            tokens = detector.tokenize(text)
            total_tokens = len(tokens)

            jargon_terms_set = set(detector.get_jargon_terms(
                text,
                threshold=OR_THRESHOLD,
                metric="or",
                ref_freq_threshold=REF_FREQ_THRESHOLD,
            ))

            jargon_count = sum(1 for t in tokens if t in jargon_terms_set)
            jargon_per_100 = (jargon_count / total_tokens * 100) if total_tokens > 0 else -1

            result = {
                "article_id": id_str,
                "language": language,
                "total_tokens": total_tokens,
                "jargon_count": jargon_count,
                "jargon_per_100": round(jargon_per_100, 4),
            }
            results.append(result)

            if writer is not None:
                writer.writerow(result)
                tsv_file.flush()
    finally:
        if tsv_file is not None:
            tsv_file.close()

    return results


def run(input_dir: Path, goal_dir: Path, overwrite: bool = False) -> None:
    goal_dir.mkdir(parents=True, exist_ok=True)

    write_info(
        goal_dir,
        pipeline="readability/pipelines/pipeline_jargon.py",
        inputs=[split.path(input_dir) for split in corpus.SPLITS],
        extra={
            "overwrite": overwrite,
            "keyness_dir": str(KEYNESS_DIR),
            "comparisons": COMPARISONS,
            "or_threshold": OR_THRESHOLD,
            "ref_freq_threshold": REF_FREQ_THRESHOLD,
        },
    )

    # Grouped by language: the detector holds a ~750 MB keyness dict, so
    # process every split of one language before switching.
    for language in ("de", "en"):
        for split in corpus.splits(language=language):
            print(f"--- {split.stem}")
            records = corpus.load(input_dir, split)
            jargon_for_data(corpus.iter_texts(records), total=len(records),
                            language=language,
                            output_tsv_path=goal_dir / f"{split.stem}.tsv",
                            overwrite=overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute jargon density (jargon words per 100 tokens).")
    parser.add_argument("--input-dir", type=Path, default=Path("data/corpus"),
                        help="Directory holding the Zenodo release JSONs.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/jargon_density"))
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing output TSVs and recompute from scratch.")
    args = parser.parse_args()

    run(args.input_dir, args.output_dir, overwrite=args.overwrite)
