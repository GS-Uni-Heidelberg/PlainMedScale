from __future__ import annotations
import argparse
import csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    BertForNextSentencePrediction,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

from lib import corpus
from lib import modelling
from lib import spacy_nlp
from lib import markdown_cleaner
from lib.run_metadata import write_info


FIELDNAMES = [
    "article_id",
    "language",
    "ansp",
    "appl_filtered_mean",
    "appl_filtered_median",
    "appl_full_mean",
    "appl_full_median",
    "appl_fulltext",
    "lmfm",
]

# Reduced schema used when `appl_only=True` — sentence-level GPT-2 perplexity
# without BERT models. Lets a laptop (MPS or CPU) compute the perplexity
# subset in minutes instead of waiting for the cluster.
APPL_ONLY_FIELDNAMES = [
    "article_id",
    "language",
    "appl_filtered_mean",
    "appl_filtered_median",
    "appl_full_mean",
    "appl_full_median",
]


def preprocess(text: str, table_mode: str, markdown_input: bool = True) -> str:
    # `plain_text` from the crawlers is already markdown-stripped (`*` markers
    # gone, paragraphs/bullets joined with single `\n`). Re-running it through
    # markdown-it would treat single newlines as soft breaks and merge bullet
    # items back into one paragraph — collapsing every list into a run-on
    # "sentence" and inflating avg_sentence_length. Skip it for plain_text.
    if markdown_input:
        if table_mode == "strip":
            text = markdown_cleaner.markdown_to_text(text)
        elif table_mode == "edit":
            text = spacy_nlp.edit_md_tables(text)
    return spacy_nlp.collapse_newlines(text)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(language: str, device: torch.device, appl_only: bool = False):
    if language == "de":
        bert_name = "bert-base-german-cased"
        gpt2_name = "dbmdz/german-gpt2"
    elif language == "en":
        bert_name = "bert-base-uncased"
        gpt2_name = "gpt2"
    else:
        raise ValueError(f"Unsupported language: {language}")

    appl_tok = AutoTokenizer.from_pretrained(gpt2_name)
    appl_model = AutoModelForCausalLM.from_pretrained(gpt2_name).to(device)
    appl_model.eval()

    if appl_only:
        return None, None, appl_tok, appl_model, None, None

    ansp_tok = AutoTokenizer.from_pretrained(bert_name)
    ansp_model = BertForNextSentencePrediction.from_pretrained(bert_name).to(device)
    ansp_model.eval()

    lmfm_tok = AutoTokenizer.from_pretrained(bert_name)
    lmfm_model = AutoModelForMaskedLM.from_pretrained(bert_name).to(device)
    lmfm_model.eval()

    return ansp_tok, ansp_model, appl_tok, appl_model, lmfm_tok, lmfm_model


def metrics_for_data(
    items,
    total: int | None = None,
    language: str = "de",
    table_mode: str = "strip",
    output_tsv_path: str | Path | None = None,
    device: torch.device | None = None,
    overwrite: bool = False,
    markdown_input: bool = True,
    appl_only: bool = False,
):
    """
    Compute LLM-based readability metrics for each (text, article_id) in `items`.
    Appends TSV rows incrementally if output_tsv_path is given.
    Skips already-processed article_ids automatically, unless overwrite=True.

    `appl_only=True` runs only sentence-level GPT-2 perplexity (the four
    appl_filtered/full × mean/median fields) under a reduced schema —
    useful for laptops without a CUDA device.
    """
    if device is None:
        device = get_device()

    match language:
        case "de":
            nlp = spacy_nlp.ger_nlp
        case "en":
            nlp = spacy_nlp.eng_nlp
        case _:
            raise ValueError(f"Unsupported language: {language}")

    fieldnames = APPL_ONLY_FIELDNAMES if appl_only else FIELDNAMES
    ansp_tok, ansp_model, appl_tok, appl_model, lmfm_tok, lmfm_model = load_models(
        language, device, appl_only=appl_only,
    )

    results = []
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
        writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writeheader()

    try:
        for text, id_ in tqdm(items, total=total):
            id_str = str(id_)
            if id_str in processed_ids:
                continue

            text = preprocess(text, table_mode, markdown_input=markdown_input)
            doc = nlp(text)

            ppls_mean, ppls_median, ppls_full_mean, ppls_full_median = modelling.appl_experiments(
                doc, appl_tok, appl_model, device
            )

            if appl_only:
                result = {
                    "article_id": id_str,
                    "language": language,
                    "appl_filtered_mean": ppls_mean,
                    "appl_filtered_median": ppls_median,
                    "appl_full_mean": ppls_full_mean,
                    "appl_full_median": ppls_full_median,
                }
            else:
                result = {
                    "article_id": id_str,
                    "language": language,
                    "ansp": modelling.ansp(doc, ansp_tok, ansp_model, device),
                    "appl_filtered_mean": ppls_mean,
                    "appl_filtered_median": ppls_median,
                    "appl_full_mean": ppls_full_mean,
                    "appl_full_median": ppls_full_median,
                    "appl_fulltext": modelling.appl_fulltext(doc, appl_tok, appl_model, device),
                    "lmfm": modelling.lmfm(doc, lmfm_tok, lmfm_model, device),
                }

            results.append(result)

            if writer is not None:
                writer.writerow(result)
                tsv_file.flush()

    finally:
        if tsv_file is not None:
            tsv_file.close()

    return results


def run(input_dir: Path, goal_dir: Path, table_mode: str, device: torch.device, overwrite: bool = False) -> None:
    goal_dir.mkdir(parents=True, exist_ok=True)

    write_info(
        goal_dir,
        pipeline="readability/pipelines/pipeline_llms.py",
        inputs=[split.path(input_dir) for split in corpus.SPLITS],
        extra={
            "table_mode": table_mode,
            "overwrite": overwrite,
            "device": str(device),
            "de_models": "bert-base-german-cased, dbmdz/german-gpt2",
            "en_models": "bert-base-uncased, gpt2",
            "metrics": ["ansp", "appl_filtered_mean", "appl_filtered_median",
                        "appl_full_mean", "appl_full_median", "appl_fulltext", "lmfm"],
        },
    )

    for split in corpus.SPLITS:
        print(f"--- {split.stem}")
        records = corpus.load(input_dir, split)
        metrics_for_data(corpus.iter_texts(records), total=len(records),
                         language=split.language, table_mode=table_mode,
                         device=device, output_tsv_path=goal_dir / f"{split.stem}.tsv",
                         overwrite=overwrite, markdown_input=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute LLM-based readability metrics (ANSP, APPL, LMFM).")
    parser.add_argument(
        "--table-mode", choices=["strip", "edit"], required=True,
        help="strip: full markdown removal; edit: convert table pipes to newlines",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/corpus"),
                        help="Directory holding the Zenodo release JSONs.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: data/metrics_llm_{table_mode}")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing output TSVs and recompute from scratch.")
    args = parser.parse_args()

    output_dir = args.output_dir or Path(f"data/metrics_llm_{args.table_mode}")
    device = get_device()
    print(f"Using device: {device}")

    run(args.input_dir, output_dir, args.table_mode, device, overwrite=args.overwrite)
