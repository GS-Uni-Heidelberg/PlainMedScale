# `readability/`

```
pipelines/          text -> per-article metric TSVs
reporting/          TSVs -> tables and figures
lib/                corpus loaders, metrics, config
jargon/             jargon keyness + detection
tests/
metrics_config.toml metric keys, labels, direction of "simpler"
```

Entry points import the shared library as `from lib import ...`, so run them
from the repo root with `PYTHONPATH=readability`.

## `pipelines/`

| Script | Metrics |
|---|---|
| `pipeline_spacy.py` | Classical readability: length, lexical, syntactic |
| `pipeline_llms.py` | ANSP (BERT NSP), APPL (GPT-2 perplexity), LMFM (BERT masked LM) — GPU |
| `pipeline_jargon.py` | Jargon density, words per 100 |

`preflight_check.py` validates inputs before a long run; `run_spacy.sh` and
`run_llms.sh` are SLURM wrappers.

All three take `--output-dir` and `--overwrite`, and **resume by default** by
skipping `article_id`s already in the output TSV. After a metric change, pass
`--overwrite` — otherwise wider rows get appended under the old header.

`pipeline_spacy.py` and `pipeline_llms.py` also take `--table-mode
{strip|edit}` (strip = remove Markdown tables, edit = pipes to newlines).
`pipeline_jargon.py` does not; it tokenizes with SoMaJo.

New data sources go in `lib/corpus.py`.

## `reporting/`

| Script | Output |
|---|---|
| `make_metrics_table.py` | Metric tables, mean ± std |
| `make_corpus_table.py` | Corpus composition — the only script here that reads `data/corpus/` |
| `plot_metrics_grid.py` | Metric grid figures |
| `plot_diag_aligned_boxplot.py` | Aligned-pair diagnostic boxplots |

Metric selection, labels and direction come from `metrics_config.toml`.
`REPRODUCING.md` maps each paper table/figure to its command.

## `jargon/`

`keyness/` computes log-likelihood and odds ratio against reference corpora;
`keys/` holds the resulting TSVs, split by tokenizer, and is too large for git
(see `jargon/keys/README.md`); `detection/` classifies tokens using them
(OR > 20, `ref_freq_threshold <= 0.001`).

**`jargon_per_100` is not comparable across languages.** DE uses
`drks-v-deleipzig`, EN uses `pubmed-v-enleipzig`; the reference corpora differ
in size and register, and SoMaJo keeps German compounds as one token, changing
the denominator. Within-language comparisons are fine.
