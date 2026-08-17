"""Automatic alignment evaluator for HPC clusters.

Classifies human-evaluated source/apoum title pairs using a local LLM
and reports accuracy against gold labels.

Features:
- GPU preflight check: fails fast if layers offloaded to CPU
- Incremental CSV writes: results saved after each classification
- Resume support: skips already-classified pairs on restart
- SIGTERM handler: writes metrics report on SLURM preemption

Usage:
    python eval_alignment.py [OPTIONS]
    python eval_alignment.py --help
"""

import csv
import logging
import re
import signal
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import typer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical terminology alignment evaluator. You will be given two medical titles: Source and Target. Classify their alignment into exactly one category:

- richtig: The titles refer to the same condition/disease.
- falsch: The titles refer to unrelated conditions.
- spezialisierung: The source title is MORE SPECIFIC than the target title (e.g. source is a sub-type).
- generalisierung: The source title is MORE GENERAL than the target title (e.g. target is a sub-type).

The titles may have different languages (German vs English) and may use different terminology (e.g. "Diabetes" vs "Zuckerkrankheit") but should still be classified as "richtig" if they refer to the same condition under the same circumstances.

Examples:
- "Grauer Star" (source) vs "Cataract" (target) => richtig
- "Diabetes mellitus Typ 2" (source) vs "Diabetes mellitus Typ 1" (target) => falsch
- "Diabetes mellitus Typ 1" (source) vs "Diabetes mellitus" (target) => spezialisierung
- "Cancer" (source) vs "Cancer in children" (target) => generalisierung

Respond with ONLY one word: richtig, falsch, spezialisierung, or generalisierung."""

VALID_LABELS = {"richtig", "falsch", "spezialisierung", "generalisierung"}

PRED_FIELDS = ["source", "apoum", "gold", "pred"]
PRED_FULL_FIELDS = ["source", "apoum", "gold", "pred", "reasoning"]


def gpu_preflight(model: AutoModelForCausalLM) -> None:
    """Check that the model is fully on GPU. Exit if layers spilled to CPU."""
    if not torch.cuda.is_available():
        log.warning("No CUDA device available — running on CPU (will be slow)")
        return

    # Claim a small amount of GPU memory to prevent other processes from taking it
    _pin = torch.zeros(1, device="cuda")  # noqa: F841

    # Log GPU memory
    free, total = torch.cuda.mem_get_info()
    log.info(f"GPU memory: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

    # Check device map for CPU-offloaded layers
    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        log.info("No device map found (single-device model)")
        return

    cpu_layers = [name for name, dev in device_map.items() if str(dev) == "cpu"]
    if cpu_layers:
        log.error(
            f"Model has {len(cpu_layers)} layers on CPU — will run ~35x slower. "
            f"First 5: {cpu_layers[:5]}"
        )
        log.error("Aborting. Request a node with more GPU memory.")
        sys.exit(1)

    devices = set(str(v) for v in device_map.values())
    log.info(f"Model device map OK — all layers on: {devices}")


def _extract_title_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    """Extract all (source, apoum) title pairs from a DataFrame with title columns."""
    source_cols = ["MSD_title", "gesund_title", "NHS_title"]
    pairs: set[tuple[str, str]] = set()
    for _, row in df.iterrows():
        apoum = row["apoum_title"]
        if not isinstance(apoum, list):
            continue
        for apoum_val in apoum:
            for col in source_cols:
                if not isinstance(row[col], list):
                    continue
                for title in row[col]:
                    source, target = title.strip(), apoum_val.strip()
                    if source and target:
                        pairs.add((source, target))
    return pairs


def _read_title_df(filepath: Path, sheet: str) -> pd.DataFrame:
    """Read an Excel sheet and split semicolon-delimited title columns into lists."""
    df = pd.read_excel(filepath, sheet_name=sheet)
    title_cols = [col for col in df.columns if "title" in col.lower()]
    comment_cols = [col for col in df.columns if "comment" in col.lower()]
    df = df[title_cols + comment_cols]
    df[title_cols] = df[title_cols].map(
        lambda x: [e.strip() for e in x.split(";")] if isinstance(x, str) else x
    )
    return df


def load_eval_pairs(
    eval_file: Path, sheet: str, raw_file: Path | None = None,
) -> dict[tuple[str, str], str]:
    df = _read_title_df(eval_file, sheet)
    df["comments"] = df["comments"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else x
    )
    df["comments"] = df["comments"].replace(
        {"False": "falsch", "richtig_1": "richtig", False: "falsch", "richtig_2": "richtig"}
    )

    source_cols = ["MSD_title", "gesund_title", "NHS_title"]
    pairs: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        apoum = row["apoum_title"]
        comment = row["comments"]
        if not isinstance(apoum, list) or not isinstance(comment, str):
            continue
        for apoum_val in apoum:
            for col in source_cols:
                if not isinstance(row[col], list):
                    continue
                for title in row[col]:
                    source, target = title.strip(), apoum_val.strip()
                    if source and target:
                        pairs[(source, target)] = comment

    # Add unevaluated pairs from the raw file as "falsch"
    if raw_file is not None:
        raw_df = _read_title_df(raw_file, sheet)
        raw_pairs = _extract_title_pairs(raw_df)
        unevaluated = raw_pairs - pairs.keys()
        log.info(f"Adding {len(unevaluated)} unevaluated pairs as 'falsch'")
        for pair in unevaluated:
            pairs[pair] = "falsch"

    return pairs


def load_completed_pairs(output_dir: Path) -> set[tuple[str, str]]:
    """Load already-classified pairs from predictions.csv for resume support."""
    pred_path = output_dir / "predictions.csv"
    if not pred_path.exists():
        return set()

    done = set()
    with open(pred_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["source"], row["apoum"]))
    return done


def classify_pair(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    source: str,
    apoum: str,
    max_new_tokens: int,
    enable_thinking: bool = False,
) -> tuple[str, str]:
    """Classify a pair and return (parsed_label, raw_response)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Source title: {source}\nTarget title: {apoum}"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    if enable_thinking:
        # Qwen3 thinking mode requires sampling with these parameters
        generate_kwargs.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    else:
        generate_kwargs.update(do_sample=False)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # Strip thinking block emitted by reasoning models before parsing
    cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip().lower()
    for label in VALID_LABELS:
        if label in cleaned:
            return label, raw_response
    return cleaned, raw_response  # return raw output if no valid label matched


def compute_metrics(results: list[dict]) -> dict:
    per_class: dict[str, dict] = {}
    for r in results:
        gold = r["gold"]
        if gold not in per_class:
            per_class[gold] = {"correct": 0, "total": 0, "preds": Counter()}
        per_class[gold]["total"] += 1
        per_class[gold]["preds"][r["pred"]] += 1
        if r["gold"] == r["pred"]:
            per_class[gold]["correct"] += 1

    total = len(results)
    correct = sum(r["gold"] == r["pred"] for r in results)
    return {"overall": {"correct": correct, "total": total}, "per_class": per_class}


def write_report(metrics: dict, output_dir: Path) -> None:
    ov = metrics["overall"]
    lines = [
        "=== Alignment Evaluation Report ===\n\n",
        f"Overall accuracy: {ov['correct']}/{ov['total']} = {ov['correct'] / ov['total']:.2%}\n",
        "\nPer-class breakdown:\n",
    ]
    for label, stats in sorted(metrics["per_class"].items()):
        acc = stats["correct"] / stats["total"]
        preds_str = ", ".join(f"{k}: {v}" for k, v in sorted(stats["preds"].items()))
        lines.append(
            f"  {label:>16}: {stats['correct']:>3}/{stats['total']:<3} = {acc:.2%}"
            f"  |  preds: {preds_str}\n"
        )

    report = "".join(lines)
    log.info("\n" + report)

    path = output_dir / "eval_report.txt"
    path.write_text(report, encoding="utf-8")
    log.info(f"Report saved to {path}")


def finalize_results(results: list[dict], output_dir: Path) -> None:
    """Compute and write metrics from whatever results we have."""
    if not results:
        log.warning("No results to compute metrics from")
        return
    metrics = compute_metrics(results)
    write_report(metrics, output_dir)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    eval_file: Path = typer.Option(
        Path("../data/instance_alignments_labeled.xlsx"),
        help="Excel file with human-evaluated pairs.",
    ),
    raw_file: Path = typer.Option(
        Path("../data/instance_alignments_raw.xlsx"),
        help="Excel file with all raw (automatic) pairs. "
        "Pairs not evaluated by annotator are added as 'falsch'.",
    ),
    sheet: str = typer.Option("apoum", help="Sheet name to evaluate."),
    model_name: str = typer.Option(
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="HuggingFace model ID.",
    ),
    max_new_tokens: int = typer.Option(
        2048, help="Max new tokens per call (covers thinking + answer)."
    ),
    output_dir: Path = typer.Option(
        Path("../results"), help="Directory for predictions.csv and eval_report.txt."
    ),
    thinking: bool = typer.Option(False, help="Enable Qwen3 thinking mode for better reasoning."),
    limit: int = typer.Option(0, help="Cap number of pairs to classify (0 = all)."),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading eval pairs from '{eval_file}', sheet='{sheet}'")
    eval_pairs = load_eval_pairs(eval_file, sheet, raw_file=raw_file)
    log.info(f"Loaded {len(eval_pairs)} labeled pairs")
    if limit:
        eval_pairs = dict(list(eval_pairs.items())[:limit])
        log.info(f"Limiting to {limit} pairs")

    # --- Resume support: load already-classified pairs ---
    done_pairs = load_completed_pairs(output_dir)
    if done_pairs:
        log.info(f"Resuming: {len(done_pairs)} pairs already classified, "
                 f"{len(eval_pairs) - len(done_pairs)} remaining")

    # Reload existing results for metrics computation at the end
    results: list[dict] = []
    pred_path = output_dir / "predictions.csv"
    if pred_path.exists():
        with open(pred_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({k: row[k] for k in PRED_FIELDS})

    log.info(f"Loading model '{model_name}'")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # --- GPU preflight check ---
    gpu_preflight(model)

    # --- Register SIGTERM handler for SLURM preemption ---
    def sigterm_handler(signum, frame):
        log.warning(f"Caught signal {signum} — writing metrics from {len(results)} results")
        finalize_results(results, output_dir)
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    # --- Incremental classification with streaming CSV writes ---
    pred_file = open(output_dir / "predictions.csv", "a", newline="", encoding="utf-8")
    pred_full_file = open(output_dir / "predictions_full.csv", "a", newline="", encoding="utf-8")

    pred_writer = csv.DictWriter(pred_file, fieldnames=PRED_FIELDS)
    pred_full_writer = csv.DictWriter(pred_full_file, fieldnames=PRED_FULL_FIELDS)

    # Write headers only if files are empty (fresh run)
    if pred_file.tell() == 0:
        pred_writer.writeheader()
        pred_file.flush()
    if pred_full_file.tell() == 0:
        pred_full_writer.writeheader()
        pred_full_file.flush()

    try:
        remaining = {k: v for k, v in eval_pairs.items() if k not in done_pairs}
        for (source, apoum), gold in tqdm(remaining.items(), desc="Classifying"):
            pred, raw_response = classify_pair(
                model, tokenizer, source, apoum, max_new_tokens, enable_thinking=thinking,
            )
            row = {"source": source, "apoum": apoum, "gold": gold, "pred": pred}
            results.append(row)

            pred_writer.writerow(row)
            pred_file.flush()

            pred_full_writer.writerow({**row, "reasoning": raw_response})
            pred_full_file.flush()
    finally:
        pred_file.close()
        pred_full_file.close()

    finalize_results(results, output_dir)


if __name__ == "__main__":
    app()
