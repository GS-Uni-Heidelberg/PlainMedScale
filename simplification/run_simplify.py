"""
Generate simplifications for the paper experiment — HPC runner.

Matches the pattern in alignment/judgment/eval_alignment.py: direct transformers,
GPU preflight, streaming TSV writes, SIGTERM handler, resume support, typer CLI.

Reads:
  data/alignment_index.json        (from build_alignment_index.py)
  data/corpus/*.json               (the Zenodo corpus release)

Writes (per-language, resumable):
  <output-dir>/simplifications_<language>.tsv

Usage:
    python simplification/run_simplify.py --help

    # Local smoke test on a tiny model
    python simplification/run_simplify.py \\
        --language de --tiers 4_tier --limit 3 \\
        --model-name Qwen/Qwen3-0.6B \\
        --output-dir simplification/results/smoke

    # HPC full run (see run_simplify.sh)
    sbatch simplification/run_simplify.sh
"""
from __future__ import annotations
import csv
import json
import logging
import random
import re
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import typer
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "readability"))
from lib import corpus  # noqa: E402
from lib.run_metadata import write_info  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

RELEASE_DIR = DATA_DIR / "corpus"

# (tier_label, source_key, subtree_key, alignment_cluster_key)
DE_TIERS = [
    ("t1_msd_prof", "msd",    "professional", "msd"),
    ("t2_msd_lay",  "msd",    "amateur",      "msd"),
    ("t3_gesund",   "gesund", "amateur",      "gesund"),
    ("t4_apoum",    "apoum",  "amateur",      "apoum"),
]
EN_TIERS = [
    ("t1_msd_prof", "msd",    "professional", "msd"),
    ("t2_msd_lay",  "msd",    "amateur",      "msd"),
    ("t3_nhs",      "nhs",    "amateur",      "nhs"),
    # gesund.bund EN articles aren't in alignment_index['en'] (the index was
    # built only against MSD/NHS on the EN side). cluster_key=None means
    # "iterate the source file directly, unaligned".
    ("t3_gesund",   "gesund", "amateur",      None),
]

UNALIGNED_BUCKET = "unaligned"

ROW_FIELDS = [
    "language", "source", "subtree", "instance_id",
    "source_word_count", "generation_time_s",
    "simplified_text", "raw_response",
]


# ─── GPU preflight (lifted from eval_alignment.py) ──────────────────────────

def gpu_preflight(model: AutoModelForCausalLM) -> None:
    if not torch.cuda.is_available():
        log.warning("No CUDA device available — running on CPU (will be very slow)")
        return

    _pin = torch.zeros(1, device="cuda")  # noqa: F841

    free, total = torch.cuda.mem_get_info()
    log.info(f"GPU memory: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

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


# ─── Data loading ────────────────────────────────────────────────────────────

@dataclass
class Article:
    text: str
    word_count: int


def index_view(release_dir: Path, source: str, subtree: str,
               language: str) -> dict[str, dict]:
    """{instance_id: release record} for one (source, subtree, language) view."""
    try:
        split = corpus.split_for(source, subtree, language)
    except KeyError:
        return {}
    if not split.path(release_dir).exists():
        return {}
    return {str(rec["instance_id"]): rec
            for rec in corpus.load(release_dir, split)}


def get_article(record: dict) -> Article | None:
    # Feed the model the Markdown form (headings, lists, emphasis) rather than
    # the flattened `plain_text`, so it sees the same structure it is asked to
    # emit. Both views carry the same content in the same order; only inline
    # markers and the H1 title differ. Word count is taken from `plain_text`,
    # which is the prose-only measure.
    text = corpus.to_markdown(record).strip()
    plain = (record.get("plain_text") or "").strip()
    if not text:
        text = plain
    if not text:
        return None
    return Article(text=text, word_count=len(plain.split()))


def build_instances(
    index_path: Path,
    language: str,
    tier_buckets: list[str],
    max_source_words: int | None,
) -> list[dict]:
    """One instance per unique (source, subtree, instance_id) pair.

    If an article appears in multiple clusters/buckets, we keep the
    highest-priority bucket (earliest in `tier_buckets`) for stratified
    ordering. Output rows do not carry cluster/tier metadata — rejoin with
    `alignment_index.json` downstream if you need it.
    """
    index = json.loads(index_path.read_text())
    tiers = DE_TIERS if language == "de" else EN_TIERS
    # One index per (source, subtree) view used by this language's tiers.
    views = {(src, sub): index_view(RELEASE_DIR, src, sub, language)
             for _label, src, sub, _cluster in tiers}

    best_bucket: dict[tuple[str, str, str], str] = {}
    priority = {b: i for i, b in enumerate(tier_buckets)}
    for bucket in tier_buckets:
        for cluster in index[language].get(bucket, []):
            for _tier_label, src_key, sub_key, cluster_key in tiers:
                if cluster_key is None:
                    continue
                if not views.get((src_key, sub_key)):
                    continue
                for instance_id in cluster.get(cluster_key, []):
                    key = (src_key, sub_key, instance_id)
                    cur = best_bucket.get(key)
                    if cur is None or priority[bucket] < priority[cur]:
                        best_bucket[key] = bucket

    # Unaligned tiers (cluster_key=None): walk the source file directly. Used
    # for sources that exist in this language but weren't included in the
    # alignment index (e.g. EN gesund.bund).
    for _tier_label, src_key, sub_key, cluster_key in tiers:
        if cluster_key is not None:
            continue
        for instance_id in views.get((src_key, sub_key), {}):
            key = (src_key, sub_key, instance_id)
            best_bucket.setdefault(key, UNALIGNED_BUCKET)

    instances: list[dict] = []
    for (src_key, sub_key, instance_id), bucket in best_bucket.items():
        record = views.get((src_key, sub_key), {}).get(instance_id)
        if record is None:
            continue
        art = get_article(record)
        if art is None:
            continue
        if max_source_words and art.word_count > max_source_words:
            continue
        instances.append({
            "text":        art.text,
            "bucket":      bucket,
            "language":    language,
            "source":      src_key,
            "subtree":     sub_key,
            "instance_id": instance_id,
            "source_word_count": art.word_count,
        })
    return instances


def load_prompts(path: Path, language: str) -> tuple[str, str]:
    with path.open() as f:
        prompts = yaml.safe_load(f)
    key = f"simplify_{language}"
    if key not in prompts:
        raise KeyError(f"Prompt '{key}' not found in {path}")
    return prompts[key]["sysprompt"].strip(), prompts[key]["userprompt"]


# ─── Resume / output ─────────────────────────────────────────────────────────

def instance_key(inst: dict) -> tuple[str, str, str, str]:
    return (inst["language"], inst["source"], inst["subtree"], inst["instance_id"])


def load_completed(output_dir: Path, language: str) -> set[tuple[str, str, str, str]]:
    tsv_path = output_dir / f"simplifications_{language}.tsv"
    if not tsv_path.exists():
        return set()
    done: set[tuple[str, str, str, str]] = set()
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Tolerate legacy rows that lack `subtree` (pre-dedup output).
            subtree = row.get("subtree") or ""
            done.add((row["language"], row["source"], subtree, row["instance_id"]))
    return done


# ─── Generation ──────────────────────────────────────────────────────────────

def _render_prompt(tokenizer, sysprompt: str, userprompt_template: str, text: str) -> str:
    messages = [
        {"role": "system", "content": sysprompt},
        {"role": "user",   "content": userprompt_template.replace("{TEXT}", text)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def _clean_raw(raw: str) -> str:
    # Strip thinking blocks. If a <think>…</think> block is present, keep
    # everything after the last </think>. If <think> is open but never closed
    # (truncated thinking), the cleaned result is empty — that is the honest answer.
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[1].strip()
    if "<think>" in raw:
        return ""
    return raw.strip()


def simplify_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sysprompt: str,
    userprompt_template: str,
    texts: list[str],
    max_new_tokens: int,
) -> list[tuple[str, str]]:
    """Run a batch of simplifications. Returns [(cleaned, raw), ...]."""
    prompts = [_render_prompt(tokenizer, sysprompt, userprompt_template, t) for t in texts]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # left padding so generated tokens align at the end for all sequences
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    finally:
        tokenizer.padding_side = prev_side

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs.input_ids.shape[1]
    results: list[tuple[str, str]] = []
    for row in output_ids:
        new_tokens = row[input_len:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append((_clean_raw(raw), raw))
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

app = typer.Typer(add_completion=False)


@app.command()
def main(
    index: Path = typer.Option(
        DATA_DIR / "alignment_index.json",
        help="Alignment index JSON from build_alignment_index.py",
    ),
    language: str = typer.Option(..., help="de or en"),
    tiers: list[str] = typer.Option(
        ["4_tier", "3_tier"],
        help="Alignment buckets to include (4_tier, 3_tier, 2_tier)",
    ),
    prompts_path: Path = typer.Option(
        Path(__file__).resolve().parent / "prompts.yaml",
        "--prompts",
    ),
    model_name: str = typer.Option(
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="HuggingFace model ID.",
    ),
    max_new_tokens: int = typer.Option(2048, help="Max new tokens per generation."),
    max_source_words: int = typer.Option(
        4000, help="Skip source articles longer than this (token-budget guard)."
    ),
    limit: int = typer.Option(0, help="Cap number of instances (0 = no cap)."),
    batch_size: int = typer.Option(4, help="Batch size for generation."),
    seed: int = typer.Option(42, help="RNG seed for stratified shuffle order."),
    bucket_priority: list[str] = typer.Option(
        ["4_tier", "3_tier", "2_tier"],
        help="Process buckets in this order — earliest first. On timeout, completed work is concentrated in earlier buckets.",
    ),
    output_dir: Path = typer.Option(
        Path("simplification/results/qwen3-30b"),
        help="Directory for simplifications_<language>.tsv.",
    ),
    dry_run: bool = typer.Option(False, help="Print summary and first instance, no model load."),
) -> None:
    if language == "en":
        tiers = [t for t in tiers if t != "4_tier"]

    output_dir.mkdir(parents=True, exist_ok=True)

    write_info(
        output_dir,
        pipeline="simplification/run_simplify.py",
        inputs=[index, prompts_path,
                *(sp.path(RELEASE_DIR) for sp in corpus.SPLITS
                  if sp.path(RELEASE_DIR).exists())],
        extra={
            "language": language,
            "tiers": tiers,
            "bucket_priority": bucket_priority,
            "model_name": model_name,
            "decoding": "greedy (do_sample=False)",
            "max_new_tokens": max_new_tokens,
            "max_source_words": max_source_words,
            "batch_size": batch_size,
            "seed": seed,
            "limit": limit,
            "dry_run": dry_run,
        },
    )

    log.info(f"Loading prompts from '{prompts_path}'")
    sysprompt, userprompt_template = load_prompts(prompts_path, language)

    log.info(f"Building instances: language={language}, tiers={tiers}")
    instances = build_instances(index, language, tiers, max_source_words=max_source_words)
    from collections import Counter
    counts = Counter((i["bucket"], i["source"], i["subtree"]) for i in instances)
    log.info(f"Built {len(instances)} unique instances (deduped on source+subtree+instance_id):")
    for (bucket, source, subtree), n in sorted(counts.items()):
        log.info(f"  {bucket:8s} {source:8s} {subtree:14s}  n={n}")

    if limit:
        instances = instances[:limit]
        log.info(f"Limiting to {limit}")

    if dry_run:
        sample = {k: v for k, v in instances[0].items() if k != "text"}
        sample["text_preview"] = instances[0]["text"][:200] + "..."
        log.info("Dry run — first instance:\n" + json.dumps(sample, indent=2, ensure_ascii=False))
        return

    # --- Resume ---
    done = load_completed(output_dir, language)
    if done:
        log.info(f"Resuming: {len(done)} instances already done, "
                 f"{len(instances) - sum(1 for i in instances if instance_key(i) in done)} remaining")

    # --- Model ---
    log.info(f"Loading model '{model_name}'")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    gpu_preflight(model)

    # --- Open output file (append mode) ---
    tsv_path = output_dir / f"simplifications_{language}.tsv"
    tsv_file = tsv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(tsv_file, fieldnames=ROW_FIELDS, delimiter="\t")
    if tsv_file.tell() == 0:
        writer.writeheader(); tsv_file.flush()

    # --- SIGTERM handler for SLURM preemption ---
    results_count = [0]

    def sigterm_handler(signum, frame):
        log.warning(f"Caught signal {signum} — flushing and exiting "
                    f"({results_count[0]} new results this run)")
        tsv_file.flush()
        tsv_file.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    # --- Run ---
    remaining = [i for i in instances if instance_key(i) not in done]
    # Stratified order: process buckets in `bucket_priority` order, and within
    # each bucket round-robin across tier labels (each group shuffled with
    # `seed`). If the job times out, completed rows are concentrated in
    # earlier buckets and balanced across tiers within each bucket — so even
    # a partial run yields a usable stratified sample.
    rng = random.Random(seed)

    def stratified_order(items: list[dict]) -> list[dict]:
        by_bucket: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(lambda: defaultdict(list))
        for it in items:
            by_bucket[it["bucket"]][(it["source"], it["subtree"])].append(it)
        ordered: list[dict] = []
        buckets = list(bucket_priority) + [b for b in by_bucket if b not in bucket_priority]
        for bucket in buckets:
            tier_groups = by_bucket.get(bucket)
            if not tier_groups:
                continue
            for group in tier_groups.values():
                rng.shuffle(group)
            queues = [list(reversed(g)) for g in tier_groups.values()]  # pop from end
            while any(queues):
                for q in queues:
                    if q:
                        ordered.append(q.pop())
        return ordered

    remaining = stratified_order(remaining)
    try:
        pbar = tqdm(total=len(remaining), desc="Simplifying")
        for start in range(0, len(remaining), batch_size):
            batch = remaining[start:start + batch_size]
            t0 = time.time()
            outs = simplify_batch(
                model, tokenizer, sysprompt, userprompt_template,
                [inst["text"] for inst in batch], max_new_tokens,
            )
            dt = time.time() - t0
            per_item_dt = dt / len(batch)
            for inst, (cleaned, raw) in zip(batch, outs):
                row = {
                    "language":          inst["language"],
                    "source":            inst["source"],
                    "subtree":           inst["subtree"],
                    "instance_id":       inst["instance_id"],
                    "source_word_count": inst["source_word_count"],
                    "generation_time_s": f"{per_item_dt:.2f}",
                    "simplified_text":   cleaned,
                    "raw_response":      raw,
                }
                writer.writerow(row)
                results_count[0] += 1
            tsv_file.flush()
            pbar.update(len(batch))
        pbar.close()
    finally:
        tsv_file.close()

    log.info(f"Done. {results_count[0]} new rows written to {tsv_path}")


if __name__ == "__main__":
    app()
