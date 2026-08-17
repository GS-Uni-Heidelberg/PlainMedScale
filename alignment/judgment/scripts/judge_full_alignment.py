"""Run an OpenAI judge over the full deduplicated alignment-pair set
(`full_pairs_filtered.tsv` from `generate_full_pairs.py` + `verify_full_pairs.py`).

Unlike eval_alignment_openai.py, there are no gold labels here — we only
emit predictions to be used downstream to filter / cluster the alignment
set. Reuses the SYSTEM_PROMPT and `classify_pair` from the OpenAI
evaluator. Resume is keyed on
(language, source_a, article_id_a, source_b, article_id_b).

Usage:
    OPENAI_API_KEY=... python alignment/judgment/scripts/judge_full_alignment.py \\
        --pairs data/alignment/full_pairs_filtered.tsv \\
        --model gpt-5.4-mini-2026-03-05 \\
        --output-dir alignment/judgment/results/full_align_judge_mini
"""
from __future__ import annotations

import csv
import logging
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import typer
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from eval_alignment_openai import UsageTracker, classify_pair

IN_COLS = ["language", "source_a", "article_id_a", "title_a",
           "source_b", "article_id_b", "title_b",
           "cluster_ids", "max_tier"]
PRED_COLS = IN_COLS + ["pred"]
PRED_FULL_COLS = PRED_COLS + ["raw_response"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_pairs(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_completed_keys(output_dir: Path) -> set[tuple]:
    pred_path = output_dir / "predictions.tsv"
    if not pred_path.exists():
        return set()
    done = set()
    with pred_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            done.add((row["language"], row["source_a"], row["article_id_a"],
                      row["source_b"], row["article_id_b"]))
    return done


app = typer.Typer(add_completion=False)


@app.command()
def main(
    pairs: Path = typer.Option(
        Path("data/alignment/full_pairs_filtered.tsv"),
        help="Deduplicated pair TSV from generate_full_pairs + verify_full_pairs --write-filtered.",
    ),
    model: str = typer.Option(
        "gpt-5.4-mini-2026-03-17", help="OpenAI model ID."
    ),
    reasoning_effort: str = typer.Option(
        "none",
        help="'none' disables reasoning; 'low'/'medium'/'high'/'xhigh' enable progressively more.",
    ),
    max_completion_tokens: int = typer.Option(
        512,
        help="Per-call cap. With reasoning=none, ~64 covers a single label easily.",
    ),
    output_dir: Path = typer.Option(
        Path("alignment/judgment/results/full_align_judge_mini"),
        help="Directory for predictions.tsv and usage.txt.",
    ),
    concurrency: int = typer.Option(
        1, help="Concurrent API calls. Keep at 1 to maximize OpenAI prompt-cache hits."
    ),
    limit: int = typer.Option(0, help="Cap number of pairs to classify (0 = all)."),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_pairs(pairs)
    log.info(f"Loaded {len(all_rows)} pairs from {pairs}")
    if limit:
        all_rows = all_rows[:limit]
        log.info(f"Limiting to {limit} pairs")

    done_keys = load_completed_keys(output_dir)
    if done_keys:
        log.info(f"Resuming: {len(done_keys)} pairs already judged, "
                 f"{len(all_rows) - len(done_keys)} remaining")

    pred_path = output_dir / "predictions.tsv"
    pred_full_path = output_dir / "predictions_full.tsv"
    pred_file = pred_path.open("a", newline="", encoding="utf-8")
    pred_full_file = pred_full_path.open("a", newline="", encoding="utf-8")
    pred_writer = csv.DictWriter(pred_file, fieldnames=PRED_COLS, delimiter="\t")
    pred_full_writer = csv.DictWriter(pred_full_file, fieldnames=PRED_FULL_COLS, delimiter="\t")
    if pred_file.tell() == 0:
        pred_writer.writeheader()
        pred_file.flush()
    if pred_full_file.tell() == 0:
        pred_full_writer.writeheader()
        pred_full_file.flush()

    load_dotenv()
    api_key = os.environ.get("API_KEY_OPENAI") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("No API key found (API_KEY_OPENAI in .env or OPENAI_API_KEY in env)")
        sys.exit(1)
    client = OpenAI(api_key=api_key)
    log.info(
        f"Using OpenAI model '{model}' reasoning={reasoning_effort or 'none'} "
        f"concurrency={concurrency}"
    )

    write_lock = Lock()
    usage = UsageTracker()

    remaining = [r for r in all_rows
                 if (r["language"], r["source_a"], r["article_id_a"],
                     r["source_b"], r["article_id_b"]) not in done_keys]
    log.info(f"Submitting {len(remaining)} pairs")

    def sigterm_handler(signum, frame):
        log.warning(f"Caught signal {signum} — closing files")
        pred_file.close()
        pred_full_file.close()
        (output_dir / "usage.txt").write_text(usage.report() + "\n", encoding="utf-8")
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    def _process(row):
        pred, raw_response = classify_pair(
            client, model, row["title_a"], row["title_b"], max_completion_tokens,
            reasoning_effort=reasoning_effort, usage_tracker=usage,
        )
        return row, pred, raw_response

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(_process, r) for r in remaining]
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Pairs")
            for fut in pbar:
                row, pred, raw = fut.result()
                out = {k: row[k] for k in IN_COLS}
                out["pred"] = pred
                with write_lock:
                    pred_writer.writerow(out)
                    pred_file.flush()
                    pred_full_writer.writerow({**out, "raw_response": raw})
                    pred_full_file.flush()
                pbar.set_postfix_str(f"cache {usage.cache_hit_rate():.0%}")
    finally:
        pred_file.close()
        pred_full_file.close()

    log.info(f"Usage: {usage.report()}")
    (output_dir / "usage.txt").write_text(usage.report() + "\n", encoding="utf-8")


if __name__ == "__main__":
    app()
