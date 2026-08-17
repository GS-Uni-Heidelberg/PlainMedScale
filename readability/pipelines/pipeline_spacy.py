from __future__ import annotations
import argparse
import csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lib import corpus
from lib import metrics
from lib import spacy_nlp
from lib import markdown_cleaner
from lib.run_metadata import write_info


# Each entry: (column_name, metric_fn)
# or:         (column_name, metric_fn, {"kwarg": "variable_name"})
# where variable_name refers to a local in the compute loop (currently just "language").
METRICS = [
    ("avg_sentence_length",          metrics.avg_sentence_length),
    ("avg_syllable_length",          metrics.avg_syllable_length),
    ("fkgl",                         metrics.fkgl),
    ("wstf",                         metrics.wstf),
    ("noun_ratio",                   metrics.noun_ratio),
    ("adjective_ratio",              metrics.adjective_ratio),
    ("function_word_ratio",          metrics.function_word_ratio, {"lang": "language"}),
    ("numbers_ratio",                metrics.numbers_ratio),
    ("negations_ratio",              metrics.negations_ratio,    {"lang": "language"}),
    ("grammar_frequency",            metrics.grammar_frequency),
    ("adjacent_sent_edit_distance",  metrics.adjacent_sent_edit_distance),
    ("dep_tree_depth",               metrics.dep_tree_depth),
    ("noun_phrase_complexity",       metrics.noun_phrase_complexity),
    ("word_freq",                    metrics.word_freq,          {"lang": "language"}),
    ("word_freq_filtered",           metrics.word_freq_filtered, {"lang": "language"}),
    ("lexical_chain_lens",           metrics.lexical_chain_lens),
    ("crossing_lexical_chains",      metrics.crossing_lexical_chains),
    ("bpe_sentence_length",          metrics.bpe_sentence_length),
    ("avg_bpe_length",               metrics.bpe_token_length),
    ("bpe_char_ratio",               metrics.bpe_char_ratio),
    ("bpe_char_ratio_lex",           metrics.bpe_char_ratio_lex),
    ("type_token_ratio",             metrics.type_token_ratio),
    ("mattr",                        metrics.mattr),
    ("heaps_law_beta",               metrics.heaps_law_beta),
    ("heaps_law_beta_lex",           metrics.heaps_law_beta_lex),
    ("verb_ratio",                   metrics.verb_ratio),
    ("herdans_c",                    metrics.herdans_c),
    ("text_length",                  metrics.text_length),
    ("text_length_lex",              metrics.text_length_lex),
    ("text_length_sentences",        metrics.text_length_sentences),
]

FIELDNAMES = ["article_id", "language"] + [name for name, *_ in METRICS]


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


def metrics_for_data(
    items,
    total: int | None = None,
    language="de",
    table_mode="strip",
    output_tsv_path: str | Path | None = None,
    overwrite: bool = False,
    markdown_input: bool = True,
):
    """
    Compute spacy readability metrics for each (text, article_id) in `items`.
    Appends TSV rows incrementally if output_tsv_path is given.
    Skips already-processed article_ids automatically, unless overwrite=True.
    """
    results = []

    match language:
        case "de":
            nlp = spacy_nlp.ger_nlp
        case "en":
            nlp = spacy_nlp.eng_nlp
        case _:
            raise ValueError(f"Unsupported language: {language}")

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

            text = preprocess(text, table_mode, markdown_input=markdown_input)
            doc = nlp(text)

            result = {"article_id": id_str, "language": language}
            ctx = {"language": language}
            for entry in METRICS:
                name, fn = entry[0], entry[1]
                kwargs = {k: ctx[v] for k, v in entry[2].items()} if len(entry) > 2 else {}
                result[name] = fn(doc, **kwargs)

            results.append(result)

            if writer is not None:
                writer.writerow(result)
                tsv_file.flush()

    finally:
        if tsv_file is not None:
            tsv_file.close()

    return results


def run(input_dir: Path, goal_dir: Path, table_mode: str, overwrite: bool = False) -> None:
    goal_dir.mkdir(parents=True, exist_ok=True)

    write_info(
        goal_dir,
        pipeline="readability/pipelines/pipeline_spacy.py",
        inputs=[split.path(input_dir) for split in corpus.SPLITS],
        extra={"table_mode": table_mode, "overwrite": overwrite,
               "metrics": [name for name, *_ in METRICS]},
    )

    for split in corpus.SPLITS:
        print(f"--- {split.stem}")
        records = corpus.load(input_dir, split)
        metrics_for_data(corpus.iter_texts(records), total=len(records),
                         language=split.language, table_mode=table_mode,
                         output_tsv_path=goal_dir / f"{split.stem}.tsv",
                         overwrite=overwrite, markdown_input=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute spacy readability metrics.")
    parser.add_argument(
        "--table-mode", choices=["strip", "edit"], required=True,
        help="strip: full markdown removal; edit: convert table pipes to newlines",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/corpus"),
                        help="Directory holding the Zenodo release JSONs.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: data/metrics_{table_mode}")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing output TSVs and recompute from scratch.")
    args = parser.parse_args()

    output_dir = args.output_dir or Path(f"data/metrics_{args.table_mode}")
    run(args.input_dir, output_dir, args.table_mode, overwrite=args.overwrite)
