# Reproducing the paper's tables and figures

Every script-generated table/figure in the paper, and the command that
regenerates it. Run from the repo root; outputs go to `latex_tables/` and
`figures/` (created on demand, gitignored). Unless noted otherwise, commands
read the shipped metric TSVs in `data/` — no GPU or corpus download needed.

Commands that need the shared library: prefix with `PYTHONPATH=readability`.

## Tables

| Paper location | Command | Output |
|---|---|---|
| Table 2 (corpus composition) | `python readability/reporting/make_corpus_table.py` (needs the corpus release in `data/corpus/`, see README "Data") | `latex_tables/corpus_table.tex` |
| Table 4 (DE readability metrics) | `python readability/reporting/make_metrics_table.py` | `latex_tables/metrics_table_de.tex` |
| Table 5 (DE metrics not in Table 4) | `python readability/reporting/make_metrics_table.py --leftovers` | `latex_tables/metrics_table_de_leftovers.tex` |
| Table 6 (EN readability metrics) | `python readability/reporting/make_metrics_table.py --language en` | `latex_tables/metrics_table_en.tex` |

Table 3 (LLM-judge accuracy on the gold set) was compiled manually from the
evaluation reports of the judge-model comparison; the judging code is in
`alignment/judgment/scripts/`, but the per-model evaluation runs are not part of
this package.

## Figures

| Paper location | Command | Output |
|---|---|---|
| Figure 1 (term frequency / noun ratio) | `python readability/reporting/plot_metrics_grid.py --metrics word_freq_filtered noun_ratio` | `figures/metrics_grid_de_word_freq_filtered_noun_ratio.pdf` |
| Figure 2 (bleed-through panels, DE) | `python simplification/plot_bleed_through.py --run-dir simplification/results/qwen3-30b` | `figures/bt_panels_de.pdf` |
| Figure 3 (all DE metrics grid) | `python readability/reporting/plot_metrics_grid.py` | `figures/metrics_grid_de.pdf` |
| Figure 4 (all EN metrics grid) | `python readability/reporting/plot_metrics_grid.py --language en` | `figures/metrics_grid_en.pdf` |

## Additional generators (not in the final paper)

| Command | Output |
|---|---|
| `python simplification/make_simplification_table.py --run-dir simplification/results/qwen3-30b [--appendix]` | Bleed-through significance table |
| `python simplification/make_bleed_through_worked_example.py --run-dir simplification/results/qwen3-30b` | Single-article bleed-through walkthrough |
| `python readability/reporting/plot_diag_aligned_boxplot.py` | Aligned-pair diagnostic boxplots |
