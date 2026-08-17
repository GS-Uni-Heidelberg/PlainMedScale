# Simple Medical Jargon Detector Usage Guide

A straightforward tool for detecting medical jargon in German text using direct SomaJo token lookup.

## Overview

The simple jargon detector works by:
1. Tokenizing text with SomaJo (German tokenizer)
2. Looking up keyness scores (odds ratios) directly for each SomaJo token
3. Classifying tokens as jargon based on threshold

**Key differences from the tiktoken-based detector:**
- ✓ Faster processing (no tiktoken encoding step)
- ✓ Simpler implementation (direct token lookup)
- ✓ Easier interpretation (no subword aggregation)
- ✗ No subword analysis (doesn't break down compound words)
- ✗ No complexity factor adjustment

## Files

- `simple_jargon_detector.py`: Main SimpleJargonDetector class
- `example_simple_detection.py`: Example usage with real medical texts

## Quick Start

```python
from pathlib import Path
from simple_jargon_detector import SimpleJargonDetector

# Initialize detector
detector = SimpleJargonDetector(
    keyness_dir=Path("../keys/somajo"),
    comparison="drks-v-deleipzig",  # Medical vs everyday German
    preprocessing="default"
)

# Analyze a single word
result = detector.identify_jargon("Prostatakarzinom", metric="or")
print(f"Word: {result['word']}")
print(f"OR Score: {result['score']:.2f}")
print(f"In dictionary: {result['in_dict']}")

# Analyze full text
text = "Die Therapie umfasst Chemotherapie und Strahlentherapie."
jargon_terms = detector.get_jargon_terms(text, threshold=20, metric="or")
print(f"Jargon terms: {jargon_terms}")
```

## Configuration Options

### Keyness Comparison

- `drks-v-deleipzig`: Medical studies (DRKS) vs everyday German (Leipzig corpus)
- `drks-v-dewiki`: Medical studies vs German Wikipedia
- `pubmed-v-enleipzig`: Medical literature (PubMed) vs everyday English
- `pubmed-v-enwiki`: Medical literature vs English Wikipedia

### Preprocessing

- `default`: No preprocessing (case-sensitive)
- `lower`: Lowercase tokens before lookup

### Metrics

- `or` (Odds Ratio): How much more frequent a token is in medical vs everyday text
  - Recommended for most use cases
  - Values > 1: More common in medical texts
  - Values < 1: More common in everyday texts

- `ll` (Log-Likelihood): Statistical significance of frequency difference
- `pd` (Percent Difference): Percentage difference in frequencies
- `bf` (Bayes Factor): Bayesian measure of evidence

## Recommended Thresholds

Based on analysis of medical texts:

- **OR > 5**: Moderately technical terms
- **OR > 10**: Likely medical terminology
- **OR > 20**: Strong jargon candidates (recommended starting point)
- **OR > 50**: Very domain-specific jargon
- **OR > 100**: Highly specialized terminology

## Example Output

```
Word: Prostatakarzinom
OR Score: 1582.82
Found in dictionary: Yes

Interpretation: This word appears 1582.82x more frequently in medical texts
than in everyday German.
```

## Running the Example

```bash
cd jargon
source ../env/bin/activate
python example_simple_detection.py
```

This will:
1. Load sample medical texts from `final_data/input/gesund_markdown.json`
2. Analyze them for jargon
3. Show threshold exploration
4. Compare specific medical terms
5. Display score distributions
6. Compare preprocessing modes

## Key Methods

### `identify_jargon(word, metric="or")`

Analyze a single word:

```python
result = detector.identify_jargon("Koloskopie", metric="or")
# Returns: {
#     'word': 'Koloskopie',
#     'score': 371.12,
#     'metric': 'or',
#     'in_dict': True
# }
```

### `analyze_text(text, metric="or", threshold=None)`

Analyze entire text:

```python
results = detector.analyze_text(text, metric="or", threshold=20)
# Returns list of dicts with:
# - word: token text
# - score: keyness score
# - metric: metric used
# - in_dict: whether found in keyness dictionary
# - is_jargon: True/False (if threshold provided)
# - freq_in_text: frequency count in input text
```

### `get_jargon_terms(text, threshold, metric="or")`

Get just the jargon words:

```python
jargon = detector.get_jargon_terms(text, threshold=20, metric="or")
# Returns: ['Prostatakarzinom', 'Therapie', 'Diagnostik', ...]
```

### `compare_words(words, metric="or")`

Compare multiple words:

```python
words = ["Patient", "Prostatakarzinom", "Arzt"]
df = detector.compare_words(words, metric="or")
print(df)
#                   word     score  metric  in_dict
# 1  Prostatakarzinom  1582.82      or     True
# 0           Patient    10.00      or     True
# 2              Arzt     0.51      or     True
```

## Tips

1. **Start with OR metric and threshold 10-20** - most interpretable and practical
2. **Use `default` preprocessing for case-sensitive matching** - preserves original capitalization
3. **Use `lower` preprocessing to ignore case** - helpful for noisy text
4. **Check `in_dict` field** - words not in dictionary get neutral scores (1.0 for OR)
5. **Consider `freq_in_text`** - helps identify frequently occurring jargon
6. **Review results manually** - use this as a tool to assist human judgment

## When to Use Simple vs Tiktoken-based Detector

### Use Simple Detector when:
- You want fast, straightforward detection
- You're working with whole words (not compounds)
- You want direct, easy-to-interpret results
- Processing speed is important
- You don't need subword analysis

### Use Tiktoken Detector when:
- You need sophisticated compound word analysis
- You want complexity factor adjustments
- You're analyzing technical terms with subword patterns
- You want to distinguish common vs specialized based on tokenization

## Comparison Example

For the word "Prostatakarzinom":

**Simple Detector:**
- Direct lookup: OR = 1582.82
- Simple interpretation: appears 1582x more in medical texts

**Tiktoken Detector:**
- Tokenizes into: ['Pro', 'st', 'ata', 'kar', 'zin', 'om']
- Individual scores: [8.12, 1.96, 2.34, 19.39, 57.06, 19.46]
- Aggregated (max): 57.06
- With complexity factor (log): 120.83

The simple detector gives a higher, more direct score. The tiktoken detector
provides more nuanced analysis but requires more computation.

## Technical Details

### How It Works

1. **Tokenization**: SomaJo splits text into tokens using German-specific rules
2. **Preprocessing**: Optional lowercasing applied before lookup
3. **Lookup**: Direct dictionary lookup using token as key
4. **Scoring**: Returns pre-computed keyness score from TSV file
5. **Default**: Words not in dictionary get neutral score (1.0 for OR, 0.0 for others)

### Keyness File Format

TSV files in `keys/somajo/{preprocessing}/{comparison}.tsv`:

```
token	study_freq	ref_freq	ll	or	pd	bf
Prostatakarzinom	1234	2	5678.9	1582.82	158182.0	5000.0
```

Where:
- `study_freq`: Count in medical corpus
- `ref_freq`: Count in reference corpus
- `ll`: Log-likelihood score
- `or`: Odds ratio
- `pd`: Percent difference
- `bf`: Bayes factor

### Why SomaJo?

SomaJo is a state-of-the-art German tokenizer that:
- Handles German-specific tokenization rules
- Provides accurate sentence and word boundaries
- Works well with medical texts
- Respects compound words as single tokens
