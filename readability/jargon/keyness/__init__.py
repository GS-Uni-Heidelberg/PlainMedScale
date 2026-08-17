"""
Word frequency comparison and jargon detection tools.

This module provides tools for comparing word frequencies between corpora
and identifying domain-specific vocabulary (jargon) using statistical keyness metrics.

Main Components
---------------
- compare_freqs: CLI script to generate comparison statistics from two JSON frequency files
- merge_comparisons: CLI script to merge multiple comparison TSV files
- keyness: Statistical metrics (log-likelihood, odds ratio, percent difference, Bayes factor)
- contingency: Contingency table utilities for frequency comparison
- io_utils: File I/O and naming convention utilities

Usage
-----
Generate comparison table:
    python -m jargon.keyness.compare_freqs \\
        --study pubmed_freqs.json \\
        --reference leipzig_freqs.json \\
        --output comparison.tsv

With transformations (lowercase, strip whitespace):
    python -m jargon.keyness.compare_freqs \\
        --study pubmed_freqs.json \\
        --reference leipzig_freqs.json \\
        --output comparison.tsv \\
        --lowercase --strip

Merge comparison tables:
    python -m jargon.keyness.merge_comparisons \\
        --input comp1.tsv comp2.tsv comp3.tsv \\
        --output merged.tsv

Programmatic usage:
    from jargon.keyness import log_likelihood_rayson, odds_ratio
    from jargon.keyness.contingency import load_frequency_json, build_contingency_table

    study_freqs = load_frequency_json("study.json")
    ref_freqs = load_frequency_json("reference.json")
    # ... build contingency table and compute metrics

"""

# Export keyness metrics for programmatic use
from .keyness import (
    log_likelihood_rayson,
    log_likelihood_scipy,
    odds_ratio,
    percent_difference,
    bayes_factor
)

__version__ = '1.0.0'

__all__ = [
    'log_likelihood_rayson',
    'log_likelihood_scipy',
    'odds_ratio',
    'percent_difference',
    'bayes_factor',
]
