# Simplification experiment

LLM-simplifies each tier of the complexity ladder; downstream scoring compares
output readability vs. source readability across tiers. See
`docs/simplification_experiment_plan.md` for the research plan.

## Server setup (one-time per deploy)

1. Sync the crawled corpora into `<repo>/data/` (gitignored; not in-repo):
   ```
   data/corpus/*.json   (the Zenodo corpus release)
   ```
2. Ensure `data/alignment/instance_alignments_raw.xlsx` is present (same file
   the eval_alignment jobs use).
3. Build the alignment index (fast, runs locally on CPU):
   ```
   python simplification/build_alignment_index.py
   ```
   Writes `data/alignment_index.json`.

## Run

```
sbatch simplification/run_simplify.sh                        # DE, all tiers
LANGUAGE=en sbatch simplification/run_simplify.sh            # EN, all tiers
TIERS="4_tier 3_tier" sbatch simplification/run_simplify.sh  # DE, drop 2-tier
```

Output: `simplification/results/qwen3-30b/simplifications_<lang>.tsv`.
Shared OUTDIR across DE + EN (resume keys include language). Resumable —
re-submitting the same job picks up where it left off.

## Input format

`run_simplify.py::get_article` feeds the model the **reconstructed Markdown**
(`# {topic.name}` + `\n\n`.join(`paragraphs[*].text`)), NOT the flat
`plain_text`. This preserves headings, bullets, and emphasis so the model
can mirror structure in its output. Parity with `plain_text` was verified
on all 4 sources × both languages (9,752 articles, 96.7% whitespace-normalized
match; residual diffs are inline-marker artifacts — same words, same order).
Falls back to `plain_text` if `paragraphs` is empty.

## Rerunning after prompt / input changes

Default-resume keys on `(language, source, subtree, instance_id)`. If you
change prompts or the input format and point to the same OUTDIR, **all
instances will be skipped** and the change has no effect. Either:

- Rename the old results dir: `mv simplification/results/qwen3-30b simplification/results/qwen3-30b-promptv1`
- Or override OUTDIR: `OUTDIR=simplification/results/qwen3-30b-mdinput LANGUAGE=de sbatch simplification/run_simplify.sh`

There is no `--overwrite` flag — rename or new OUTDIR are the only options.

## Smoke test (local, CPU or tiny GPU)

```
python simplification/run_simplify.py \
    --language de --tiers 4_tier --limit 3 \
    --model-name Qwen/Qwen3-0.6B \
    --max-new-tokens 512 \
    --output-dir simplification/results/smoke
```
