# Keyness TSVs

`readability/pipelines/pipeline_jargon.py` classifies a token as jargon from its keyness
in medical research texts against a general-language reference corpus
(odds ratio > 20 and reference relative frequency ≤ 0.001; see paper
Appendix A). It expects, relative to the repo root:

```
readability/jargon/keys/somajo/default/drks-v-deleipzig.tsv    # German
readability/jargon/keys/somajo/default/pubmed-v-enleipzig.tsv  # English
```

These files are several hundred MB and are not tracked in git. To rebuild:

1. Obtain the corpora. The exact sources used for the shipped keyness runs:
   - **DRKS** (DE medical) — 17,405 trial records crawled from the German
     Clinical Trials Register (https://www.drks.de) in 2024; per record we
     use the expert abstract and the trial title.
   - **PubMed** (EN medical) — abstracts extracted from the first 113 files
     (`pubmed25n0001` – `pubmed25n0113`, ~20 GB XML) of the official PubMed
     2025 annual baseline (https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/),
     yielding ~1.99 M abstracts.
   - **Leipzig Corpora Collection** (general-language reference) — the 1M-
     sentence *news* corpora from
     https://wortschatz.uni-leipzig.de/en/download, all years concatenated:
     `deu_news_{1995..2024}_1M` (German, 30 corpora) and
     `eng_news_{2005..2010,2013..2020,2023,2024}_1M` (English, 16 corpora).
2. Tokenize each corpus with SoMaJo and write one JSON frequency dictionary
   per corpus (`{token: absolute_count, ...}`) using
   `readability/jargon/keyness/build_freqs.py`:

```bash
python -m jargon.keyness.build_freqs --format drks \
    --input <drks_dir> --language de_CMC --output drks_somajo_freqs.json
python -m jargon.keyness.build_freqs --format pubmed-jsonl \
    --input <pubmed_abstracts.jsonl> --language en_PTB --output pubmed_somajo_freqs.json
python -m jargon.keyness.build_freqs --format leipzig \
    --input <leipzig_deu_dir> --language de_CMC --output leipzig_de_somajo_freqs.json
python -m jargon.keyness.build_freqs --format leipzig \
    --input <leipzig_eng_dir> --language en_PTB --output leipzig_en_somajo_freqs.json
```

   Expected input layouts: DRKS as a directory of per-trial `*.json` files
   (fields `expert_abstract`, `h2`); PubMed as one JSONL file with an
   `abstract` field per record; Leipzig as the unpacked download directories
   (the `*-sentences.txt` files are found recursively).

3. Compute the keyness tables:

```bash
python -m jargon.keyness.compare_freqs \
    --study drks_somajo_freqs.json --reference leipzig_de_somajo_freqs.json \
    --output readability/jargon/keys/somajo/default/drks-v-deleipzig.tsv --strip

python -m jargon.keyness.compare_freqs \
    --study pubmed_somajo_freqs.json --reference leipzig_en_somajo_freqs.json \
    --output readability/jargon/keys/somajo/default/pubmed-v-enleipzig.tsv --strip
```

The two reference corpora differ in size and register, and SoMaJo keeps German
nominal compounds as single tokens — within-language comparisons of the
resulting jargon densities are meaningful, cross-language differences are not
(see the paper's Appendix A).
