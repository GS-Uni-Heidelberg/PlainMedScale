"""
Apply readability/jargon/LLM metrics to simplified texts produced by
run_simplify.py.

Reads:
  simplification/results/<run>/simplifications_<lang>.tsv

Writes (per language, per source, per subtree — mirrors the layout of
`data/metrics_strip/`, `data/jargon_density/`, `data/metrics_llm_strip/`):
  simplification/results/<run>/metrics_strip/<lang>_<source>_<subtree>.tsv
  simplification/results/<run>/jargon_density/<lang>_<source>_<subtree>.tsv
  simplification/results/<run>/metrics_llm_strip/<lang>_<source>_<subtree>.tsv   (only with --metric llm)

`article_id` in the output TSVs is the `instance_id` from the simplification
TSV — join against `data/metrics_*/<lang>_<equivalent>.tsv` by that id.

Usage (from repo root):
  python simplification/pipeline_metrics.py \\
      --run-dir simplification/results/qwen-30b-de \\
      --metrics spacy,jargon

  python simplification/pipeline_metrics.py \\
      --run-dir simplification/results/qwen3-30b \\
      --metrics llm --table-mode strip          # needs a GPU

  # Laptop-friendly subset: only sentence-level GPT-2 perplexity (4 fields),
  # writes to metrics_appl_strip/ alongside the full metrics_llm_strip/.
  python simplification/pipeline_metrics.py \\
      --run-dir simplification/results/qwen3-30b \\
      --metrics llm --table-mode strip --appl-only
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

# Make `from lib...` resolve like the readability pipelines do.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "readability"))

from pipeline_spacy import metrics_for_data as spacy_metrics_for_data  # noqa: E402
from pipeline_jargon import (  # noqa: E402
    COMPARISONS as JARGON_COMPARISONS,
    FIELDNAMES as JARGON_FIELDNAMES,
    KEYNESS_DIR as JARGON_KEYNESS_DIR,
    OR_THRESHOLD as JARGON_OR_THRESHOLD,
    REF_FREQ_THRESHOLD as JARGON_REF_FREQ_THRESHOLD,
)

sys.path.insert(0, str(REPO_ROOT / "readability" / "jargon" / "detection"))
from simple_jargon_detector import SimpleJargonDetector  # noqa: E402

from lib.run_metadata import write_info  # noqa: E402


# --- simplified-text source loader ------------------------------------------

def load_simplifications(run_dir: Path) -> pd.DataFrame:
    """Concatenate all simplifications_*.tsv under run_dir into one frame."""
    tsvs = sorted(run_dir.glob("simplifications_*.tsv"))
    if not tsvs:
        raise SystemExit(f"No simplifications_*.tsv found under {run_dir}")
    frames = []
    for p in tsvs:
        df = pd.read_csv(p, sep="\t", dtype=str, quoting=csv.QUOTE_MINIMAL)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Drop rows where the model produced empty text (e.g. truncated <think>).
    df = df[df["simplified_text"].fillna("").str.strip().ne("")].reset_index(drop=True)
    print(f"Loaded {len(df)} non-empty simplification rows from {len(tsvs)} TSVs")
    return df


def iter_rows(rows):
    """Yield (text, instance_id) — the item shape the metric pipelines take."""
    for row in rows:
        yield row["simplified_text"], row["instance_id"]


def groups(df: pd.DataFrame):
    """Yield (language, source, subtree, [rows]) for each distinct triple."""
    for (lang, source, subtree), grp in df.groupby(["language", "source", "subtree"]):
        rows = grp.to_dict(orient="records")
        yield lang, source, subtree, rows


# --- runners ----------------------------------------------------------------

def run_spacy(df: pd.DataFrame, out_dir: Path, table_mode: str, overwrite: bool):
    for lang, source, subtree, rows in groups(df):
        out = out_dir / f"{lang}_{source}_{subtree}.tsv"
        print(f"[spacy] {lang}/{source}/{subtree}  n={len(rows)}  -> {out}")
        spacy_metrics_for_data(
            iter_rows(rows), total=len(rows), language=lang, table_mode=table_mode,
            output_tsv_path=out, overwrite=overwrite,
        )


def run_jargon(df: pd.DataFrame, out_dir: Path, overwrite: bool):
    """Like pipeline_jargon.jargon_for_data but the 749 MB keyness dict is
    loaded once per language and reused across all (source, subtree) groups."""
    from tqdm import tqdm

    detectors: dict[str, SimpleJargonDetector] = {}

    def get_detector(lang: str) -> SimpleJargonDetector:
        if lang not in detectors:
            print(f"[jargon] loading detector for {lang} "
                  f"(this reads a multi-GB keyness TSV; can take ~1-2 min)...")
            detectors[lang] = SimpleJargonDetector(
                keyness_dir=JARGON_KEYNESS_DIR,
                comparison=JARGON_COMPARISONS[lang],
                preprocessing="default",
            )
        return detectors[lang]

    for lang, source, subtree, rows in groups(df):
        out = out_dir / f"{lang}_{source}_{subtree}.tsv"
        print(f"[jargon] {lang}/{source}/{subtree}  n={len(rows)}  -> {out}")

        out.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and out.exists():
            out.unlink()
            print(f"  overwriting {out}")

        processed_ids: set[str] = set()
        if out.exists() and out.stat().st_size > 0:
            try:
                existing = pd.read_csv(out, sep="\t", usecols=["article_id"], dtype=str)
                processed_ids = set(existing["article_id"].astype(str))
                print(f"  resume: {len(processed_ids)} already done")
            except Exception as e:
                print(f"  could not read existing TSV ({e}); continuing without skip list")

        detector = get_detector(lang)
        write_header = not (out.exists() and out.stat().st_size > 0)
        with open(out, "a", encoding="utf-8", newline="") as tsv_file:
            writer = csv.DictWriter(tsv_file, fieldnames=JARGON_FIELDNAMES,
                                    delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            if write_header:
                writer.writeheader()
            for text, id_ in tqdm(iter_rows(rows), total=len(rows)):
                id_str = str(id_)
                if id_str in processed_ids:
                    continue
                tokens = detector.tokenize(text)
                total_tokens = len(tokens)
                jargon_terms_set = set(detector.get_jargon_terms(
                    text,
                    threshold=JARGON_OR_THRESHOLD,
                    metric="or",
                    ref_freq_threshold=JARGON_REF_FREQ_THRESHOLD,
                ))
                jargon_count = sum(1 for t in tokens if t in jargon_terms_set)
                jargon_per_100 = (jargon_count / total_tokens * 100) if total_tokens > 0 else -1
                writer.writerow({
                    "article_id":     id_str,
                    "language":       lang,
                    "total_tokens":   total_tokens,
                    "jargon_count":   jargon_count,
                    "jargon_per_100": round(jargon_per_100, 4),
                })
                tsv_file.flush()


def run_llm(df: pd.DataFrame, out_dir: Path, table_mode: str, overwrite: bool,
            appl_only: bool = False):
    # Imported lazily — loading torch/transformers is slow and not needed for
    # the common spacy+jargon invocation.
    from pipeline_llms import metrics_for_data as llm_metrics_for_data, get_device
    device = get_device()
    print(f"[llm] using device: {device}  appl_only={appl_only}")
    for lang, source, subtree, rows in groups(df):
        out = out_dir / f"{lang}_{source}_{subtree}.tsv"
        print(f"[llm] {lang}/{source}/{subtree}  n={len(rows)}  -> {out}")
        llm_metrics_for_data(
            iter_rows(rows), total=len(rows), language=lang, table_mode=table_mode,
            device=device, output_tsv_path=out, overwrite=overwrite,
            appl_only=appl_only,
        )


# --- CLI --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply readability metrics to LLM-simplified texts.",
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Directory containing simplifications_<lang>.tsv "
             "(e.g. simplification/results/qwen-30b-de).",
    )
    parser.add_argument(
        "--metrics", default="spacy,jargon",
        help="Comma-separated subset of {spacy,jargon,llm}. "
             "LLM needs a GPU; excluded from default.",
    )
    parser.add_argument(
        "--table-mode", choices=["strip", "edit"], default="strip",
        help="Markdown table handling (same semantics as readability pipelines).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--appl-only", action="store_true",
        help="LLM subset: only sentence-level GPT-2 perplexity "
             "(skip ANSP, LMFM, appl_fulltext). Writes a reduced schema "
             "to metrics_appl_<table_mode>/ — separate dir so it doesn't "
             "collide with the full LLM-metrics output.",
    )
    args = parser.parse_args()

    wanted = {m.strip() for m in args.metrics.split(",") if m.strip()}
    unknown = wanted - {"spacy", "jargon", "llm"}
    if unknown:
        parser.error(f"unknown metric(s): {unknown}")

    df = load_simplifications(args.run_dir)
    simp_tsvs = sorted(args.run_dir.glob("simplifications_*.tsv"))

    def _stamp(out_dir: Path, family: str, extra: dict | None = None) -> None:
        e = {"run_dir": str(args.run_dir), "n_rows": len(df),
             "overwrite": args.overwrite}
        if extra:
            e.update(extra)
        write_info(out_dir, pipeline=f"simplification/pipeline_metrics.py ({family})",
                   inputs=simp_tsvs, extra=e)

    if "spacy" in wanted:
        out = args.run_dir / f"metrics_{args.table_mode}"
        _stamp(out, "spacy", {"table_mode": args.table_mode})
        run_spacy(df, out, args.table_mode, args.overwrite)
    if "jargon" in wanted:
        out = args.run_dir / "jargon_density"
        _stamp(out, "jargon")
        run_jargon(df, out, args.overwrite)
    if "llm" in wanted:
        subdir = f"metrics_appl_{args.table_mode}" if args.appl_only else f"metrics_llm_{args.table_mode}"
        out = args.run_dir / subdir
        _stamp(out, "llm", {"table_mode": args.table_mode, "appl_only": args.appl_only})
        run_llm(df, out, args.table_mode, args.overwrite, appl_only=args.appl_only)


if __name__ == "__main__":
    main()
