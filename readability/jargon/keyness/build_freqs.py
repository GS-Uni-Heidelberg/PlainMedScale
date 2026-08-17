"""Build SoMaJo token-frequency JSONs for the keyness comparisons.

Tokenizes a corpus with SoMaJo (parallelized) and writes a
``{token: absolute_count}`` JSON — the input format expected by
``jargon.keyness.compare_freqs``. One loader per corpus format used in
the paper:

    drks          per-trial ``*.json`` files; uses the ``expert_abstract``
                  and ``h2`` fields (German Clinical Trials Register)
    pubmed-jsonl  one JSON record per line; uses the ``abstract`` field
    leipzig       Leipzig Corpora Collection ``*-sentences.txt`` files
                  (tab-separated, sentence in column 2)

Usage:
    python -m jargon.keyness.build_freqs --format drks \
        --input <drks_dir> --language de_CMC --output drks_somajo_freqs.json
    python -m jargon.keyness.build_freqs --format pubmed-jsonl \
        --input abstracts.jsonl --language en_PTB --output pubmed_somajo_freqs.json
    python -m jargon.keyness.build_freqs --format leipzig \
        --input <leipzig_deu_dir> --language de_CMC --output leipzig_de_somajo_freqs.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path

from somajo import SoMaJo
from tqdm import tqdm


def load_drks(path: Path) -> list[str]:
    docs = []
    for file in sorted(path.glob("*.json")):
        with file.open(encoding="utf-8") as f:
            data = json.load(f)
        docs.append(data.get("expert_abstract", ""))
        docs.append(data.get("h2", ""))
    return [d for d in docs if d.strip()]


def load_pubmed_jsonl(path: Path) -> list[str]:
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["abstract"])
    return [d for d in docs if d.strip()]


def load_leipzig(path: Path) -> list[str]:
    docs = []
    for file in sorted(path.rglob("*sentences.txt")):
        with file.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > 1:
                    docs.append(parts[1])
    return [d for d in docs if d.strip()]


LOADERS = {
    "drks": load_drks,
    "pubmed-jsonl": load_pubmed_jsonl,
    "leipzig": load_leipzig,
}


def _tokenize_chunk(args: tuple[list[str], str]) -> tuple[Counter, int]:
    documents, language = args
    tokenizer = SoMaJo(language, split_camel_case=False, split_sentences=False)
    counter: Counter = Counter()
    for doc in documents:
        for sentence in tokenizer.tokenize_text([doc]):
            counter.update(token.text for token in sentence)
    return counter, len(documents)


def tokenize_and_count(documents: list[str], language: str,
                       chunk_size: int = 500) -> dict[str, int]:
    """Count SoMaJo token frequencies over ``documents`` in parallel."""
    n_workers = max(1, mp.cpu_count() - 4)
    chunks = [documents[i:i + chunk_size]
              for i in range(0, len(documents), chunk_size)]
    total: Counter = Counter()
    with mp.Pool(processes=n_workers) as pool:
        with tqdm(total=len(documents), desc="Tokenizing", unit="doc") as pbar:
            for counter, n_docs in pool.imap(
                    _tokenize_chunk, ((c, language) for c in chunks)):
                total.update(counter)
                pbar.update(n_docs)
    return dict(total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", required=True, choices=sorted(LOADERS))
    parser.add_argument("--input", type=Path, required=True,
                        help="Corpus directory (drks, leipzig) or file "
                             "(pubmed-jsonl)")
    parser.add_argument("--language", required=True,
                        choices=["de_CMC", "en_PTB"])
    parser.add_argument("--output", type=Path, required=True,
                        help="Output frequency JSON")
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()

    documents = LOADERS[args.format](args.input)
    print(f"{len(documents)} documents")
    freqs = tokenize_and_count(documents, args.language, args.chunk_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(freqs, f, ensure_ascii=False)
    print(f"{len(freqs)} unique tokens, {sum(freqs.values())} total")


if __name__ == "__main__":
    main()
