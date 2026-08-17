# `data/`

Derived numbers only — metric values, alignment ids, titles and labels. **No
article text.** The corpus is released separately on Zenodo,
**doi:10.5281/zenodo.21728290** (subject to the access terms stated there);
download it into `data/corpus/`, which is gitignored and ships empty.

```
data/
    corpus/                 NOT in git — download from Zenodo
    alignment/              alignment metadata
    metrics_strip/          classical readability metrics, per article
    metrics_llm_strip/      LLM-based metrics, per article
    jargon_density/         jargon density, per article
    alignment_index.json    deduplicated aligned-instance index
```

Every table and figure regenerates from what is shipped here — see
`REPRODUCING.md`. The corpus download is needed only to recompute the metric
TSVs, to re-run the simplification experiment, or for Table 2, which counts
articles.

## Metric TSVs

One TSV per (language, source, subtree), one row per article, keyed on the
release's own `article_id` so they join directly against the corpus files.

Only the `strip` table mode ships; no shipped table or figure reads the `edit`
variant. The fulltext perplexity column is named `apll_full_median` (sic) —
kept for compatibility.

## `data/alignment/`

Evidence behind the alignment-quality numbers (§3.2).

| File | Role |
|---|---|
| `full_pairs*.tsv` | Candidate pairs from the lexical/embedding search |
| `full_pairs*_filtered.tsv` | After dedup and sanity checks — what the judge consumes |
| `instance_alignments_raw.xlsx` | Raw cross-source proposals; co-occurrence in a row is the only signal |
| `instance_alignments_labeled.xlsx` | Human annotation of those proposals |
| `alignment_judgments.json` | Merged human + LLM verdicts, tagged with `method` and `label` |
| `alignment_judgments_msdamateur.json` | Same, for the MSD amateur-title view |

The `_msdamateur` files are a second pass over the same clusters using MSD's
lay titles instead of professional ones; the paper's numbers use the
professional view.

These sit under `data/` rather than beside the alignment code because
`readability/` and `simplification/` read them too.
