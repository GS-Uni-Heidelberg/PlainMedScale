"""OpenAI-based alignment evaluator.

Mirrors eval_alignment.py but calls a hosted OpenAI chat-completions model
instead of a local GPU model. Same predictions.csv schema, same resume
support, same metrics report. One pair per API call (no batching) for
clean apples-to-apples comparison across models.

Usage:
    OPENAI_API_KEY=... python alignment/judgment/scripts/eval_alignment_openai.py \\
        --eval-file data/alignment/instance_alignments_labeled.xlsx \\
        --raw-file data/alignment/instance_alignments_raw.xlsx \\
        --model gpt-5.4-mini-2026-03-17 \\
        --output-dir alignment/judgment/results/gpt-5.4-mini
"""

import csv
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Literal

import typer
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from eval_alignment import (
    PRED_FIELDS,
    PRED_FULL_FIELDS,
    finalize_results,
    load_completed_pairs,
    load_eval_pairs,
)


class AlignmentLabel(BaseModel):
    """Strict OpenAI JSON-schema response: single enum label."""
    label: Literal["richtig", "spezialisierung", "generalisierung", "falsch"]


class UsageTracker:
    """Thread-safe accumulator for OpenAI usage stats — verifies prompt caching."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.calls = 0
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0

    def add(self, usage) -> None:
        if usage is None:
            return
        with self._lock:
            self.calls += 1
            self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            ptd = getattr(usage, "prompt_tokens_details", None)
            if ptd is not None:
                self.cached_tokens += getattr(ptd, "cached_tokens", 0) or 0
            ctd = getattr(usage, "completion_tokens_details", None)
            if ctd is not None:
                self.reasoning_tokens += getattr(ctd, "reasoning_tokens", 0) or 0

    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.input_tokens if self.input_tokens else 0.0

    def report(self) -> str:
        return (
            f"calls={self.calls}  "
            f"input={self.input_tokens}  cached={self.cached_tokens} "
            f"({self.cache_hit_rate():.1%})  "
            f"output={self.output_tokens}  reasoning={self.reasoning_tokens}"
        )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a medical terminology alignment evaluator. You will be given two medical article titles (Source and Target) and must classify how they align.

The titles may be in different languages (German vs English) or use different registers (technical vs lay name). Cross-language and lay/technical equivalents that refer to the same condition are "richtig", not "falsch".

CATEGORIES
==========
- richtig: The two titles refer to the SAME clinical condition. Different language or different terminology is fine — what matters is that a clinician would treat them as the same diagnosis.
- spezialisierung: The Source is a strict SUBTYPE of the Target. Source narrower, Target umbrella.
- generalisierung: The Target is a strict SUBTYPE of the Source. Mirror of spezialisierung.
- falsch: anything else.

TRANSLATION / LAY-NAME EQUIVALENCE IS "RICHTIG"
===============================================
Do NOT label cross-language pairs "falsch" just because the words look different. Recognise translations and lay/technical equivalents as the same condition:
  "Otitis media" vs "Mittelohrentzündung"   => richtig
  "Gout" vs "Gicht"                          => richtig
  "Hives" vs "Nesselsucht"                   => richtig
  "Frostbite" vs "Erfrierung"                => richtig

PARENTHETICAL GLOSSES DO NOT MAKE A TITLE MORE SPECIFIC
=======================================================
"X (Y)" where X and Y are synonyms is the same condition stated twice, not a subtype.
  "Kidney stones (nephrolithiasis)" vs "Nierensteine" => richtig

BE STRICT — THESE ARE "FALSCH", NOT "RICHTIG"
=============================================
- Two related but distinct conditions under a common umbrella, or sister diseases of the same family
  ("Glaucoma" vs "Makuladegeneration" — both eye diseases, but distinct).
- A symptom vs the disease that causes it, or a disease vs one of its complications
  ("Nausea" vs "Gastroenteritis").
- A specific drug, drug class, or vaccine vs the disease it targets — these articles are about the drug/vaccine itself, not the disease
  ("HPV Vaccine" vs "Gebärmutterhalskrebs").
  Note the contrast: a "Treatment of X" or "Management of X" article is ABOUT X (different angle on the same condition) and should be richtig — see below.

SPEZIALISIERUNG / GENERALISIERUNG ONLY FOR CLEAR TAXONOMIC SUBTYPES
====================================================================
Use only when one is unambiguously a sub-type of the other.
  "Plantar fasciitis" vs "Fasciitis"          => spezialisierung
  "Headache" vs "Cluster-Kopfschmerz"         => generalisierung

When in doubt between a hierarchy label and "falsch", prefer "falsch".

"OVERVIEW OF X" AND "TREATMENT OF X" ARTICLES
=============================================
These are alternative entries on the same disease, not different topics:
- "Overview of X" vs the article on X itself => richtig (the Overview IS the canonical entry on X).
- "Overview of X" vs an article on a strict subtype Y of X => generalisierung (X is broader than Y).
- "Overview of X" vs an article on something unrelated => falsch.
- "Treatment of X" / "Management of X" / "Diagnosis of X" vs the article on X => richtig
  (different facet of the same disease topic, not a different topic).
Examples:
  "Overview of Asthma" vs "Asthma bronchiale"        => richtig
  "Overview of Hepatitis" vs "Hepatitis A"           => generalisierung (Hepatitis A = subtype)
  "Management of Asthma" vs "Asthma bronchiale"      => richtig

OUTPUT
======
You will receive one pair in the user message with "Source:" and "Target:" lines. Respond with a JSON object {"label": "<one of richtig, spezialisierung, generalisierung, falsch>"}. No other fields, no explanation."""


def classify_pair(
    client: OpenAI,
    model: str,
    source: str,
    target: str,
    max_completion_tokens: int,
    reasoning_effort: str,
    usage_tracker: UsageTracker | None = None,
    max_retries: int = 4,
) -> tuple[str, str]:
    user_msg = f"Source: {source}\nTarget: {target}"
    last_err: Exception | None = None
    extra_body: dict = {"max_completion_tokens": max_completion_tokens}
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort
    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format=AlignmentLabel,
                extra_body=extra_body,
            )
            if usage_tracker is not None:
                usage_tracker.add(getattr(response, "usage", None))
            msg = response.choices[0].message
            if getattr(msg, "refusal", None):
                raise RuntimeError(f"Model refused: {msg.refusal}")
            if msg.parsed is None:
                raise RuntimeError(f"No parsed output; raw content={msg.content!r}")
            return msg.parsed.label, msg.content or ""
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            log.warning(f"OpenAI call failed (attempt {attempt + 1}/{max_retries}): {e}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"OpenAI call failed after {max_retries} retries") from last_err


app = typer.Typer(add_completion=False)


@app.command()
def main(
    eval_file: Path = typer.Option(
        Path("data/alignment/instance_alignments_labeled.xlsx"),
        help="Excel file with human-evaluated pairs.",
    ),
    raw_file: Path = typer.Option(
        Path("data/alignment/instance_alignments_raw.xlsx"),
        help="Excel file with all raw (automatic) pairs. "
        "Pairs not evaluated by annotator are added as 'falsch'.",
    ),
    sheet: str = typer.Option("apoum", help="Sheet name to evaluate."),
    model: str = typer.Option("gpt-5.4-2026-03-05", help="OpenAI model ID."),
    reasoning_effort: str = typer.Option(
        "none",
        help="Reasoning effort for gpt-5 family: 'none' = no thinking; "
        "'low' / 'medium' / 'high' / 'xhigh' enable progressively more reasoning.",
    ),
    max_completion_tokens: int = typer.Option(
        512,
        help="Per-call cap. With reasoning_effort=none, ~64 covers a single label easily; "
        "bump for thinking modes.",
    ),
    output_dir: Path = typer.Option(
        Path("alignment/judgment/results/gpt-5.4"),
        help="Directory for predictions.csv and eval_report.txt.",
    ),
    concurrency: int = typer.Option(
        1, help="Concurrent API calls. Increase for faster wall-time, watch rate limits."
    ),
    limit: int = typer.Option(0, help="Cap number of pairs to classify (0 = all)."),
    hard_set_csv: Path = typer.Option(
        None,
        help="If set, load pairs from a CSV with columns source,apoum,gold (extra cols ignored) "
        "instead of the xlsx — used for re-running on a curated subset.",
    ),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if hard_set_csv:
        log.info(f"Loading eval pairs from hard-set CSV '{hard_set_csv}'")
        eval_pairs = {}
        with open(hard_set_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                eval_pairs[(row["source"], row["apoum"])] = row["gold"]
    else:
        log.info(f"Loading eval pairs from '{eval_file}', sheet='{sheet}'")
        eval_pairs = load_eval_pairs(eval_file, sheet, raw_file=raw_file)
    log.info(f"Loaded {len(eval_pairs)} labeled pairs")
    if limit:
        eval_pairs = dict(list(eval_pairs.items())[:limit])
        log.info(f"Limiting to {limit} pairs")

    done_pairs = load_completed_pairs(output_dir)
    if done_pairs:
        log.info(
            f"Resuming: {len(done_pairs)} pairs already classified, "
            f"{len(eval_pairs) - len(done_pairs)} remaining"
        )

    results: list[dict] = []
    pred_path = output_dir / "predictions.csv"
    if pred_path.exists():
        with open(pred_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({k: row[k] for k in PRED_FIELDS})

    load_dotenv()
    api_key = os.environ.get("API_KEY_OPENAI") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("No API key found (API_KEY_OPENAI in .env or OPENAI_API_KEY in env)")
        sys.exit(1)
    client = OpenAI(api_key=api_key)
    log.info(f"Using OpenAI model '{model}' with concurrency={concurrency}")

    pred_file = open(output_dir / "predictions.csv", "a", newline="", encoding="utf-8")
    pred_full_file = open(output_dir / "predictions_full.csv", "a", newline="", encoding="utf-8")
    pred_writer = csv.DictWriter(pred_file, fieldnames=PRED_FIELDS)
    pred_full_writer = csv.DictWriter(pred_full_file, fieldnames=PRED_FULL_FIELDS)
    if pred_file.tell() == 0:
        pred_writer.writeheader()
        pred_file.flush()
    if pred_full_file.tell() == 0:
        pred_full_writer.writeheader()
        pred_full_file.flush()

    write_lock = Lock()

    def sigterm_handler(signum, frame):
        log.warning(f"Caught signal {signum} — writing metrics from {len(results)} results")
        finalize_results(results, output_dir)
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    remaining = [(k, v) for k, v in eval_pairs.items() if k not in done_pairs]
    log.info(f"Submitting {len(remaining)} pairs")
    usage = UsageTracker()

    def _process(item):
        (source, apoum), gold = item
        pred, raw_response = classify_pair(
            client, model, source, apoum, max_completion_tokens,
            reasoning_effort=reasoning_effort, usage_tracker=usage,
        )
        return source, apoum, gold, pred, raw_response

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(_process, item) for item in remaining]
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Pairs")
            for fut in pbar:
                source, apoum, gold, pred, raw_response = fut.result()
                row = {"source": source, "apoum": apoum, "gold": gold, "pred": pred}
                with write_lock:
                    results.append(row)
                    pred_writer.writerow(row)
                    pred_file.flush()
                    pred_full_writer.writerow({**row, "reasoning": raw_response})
                    pred_full_file.flush()
                pbar.set_postfix_str(f"cache {usage.cache_hit_rate():.0%}")
    finally:
        pred_file.close()
        pred_full_file.close()

    log.info(f"Usage: {usage.report()}")
    (output_dir / "usage.txt").write_text(usage.report() + "\n", encoding="utf-8")

    finalize_results(results, output_dir)


if __name__ == "__main__":
    app()
