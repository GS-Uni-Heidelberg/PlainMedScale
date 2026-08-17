"""
Generate word frequency comparison statistics from two corpora.

This script compares word frequencies between a study corpus and a reference corpus,
computing statistical keyness metrics to identify distinctive vocabulary.

Usage:
    python -m jargon.keyness.compare_freqs -s STUDY -r REF -o OUTPUT

Example:
    python -m jargon.keyness.compare_freqs \\
        --study pubmed_freqs.json \\
        --reference leipzig_freqs.json \\
        --output pubmed_vs_leipzig.tsv \\
        --smoothing 0.1
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

from .contingency import (
    load_frequency_json,
    compute_corpus_stats,
    build_contingency_table,
    compute_unk_contingency,
    transform_frequencies
)
from .io_utils import write_comparison_tsv
from .keyness import (
    log_likelihood_rayson,
    odds_ratio,
    percent_difference,
    bayes_factor
)


AVAILABLE_METRICS = {
    'll': ('log_likelihood_rayson', log_likelihood_rayson),
    'or': ('odds_ratio', odds_ratio),
    'pd': ('percent_difference', percent_difference),
    'bf': ('bayes_factor', bayes_factor)
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate word frequency comparison statistics from two corpora.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input JSON Format:
    Supported formats:
    - {"word_freqs_lemma": {token: freq, ...}}
    - {"word_freqs_wordform": {token: freq, ...}}
    - {token: freq, ...}

Output TSV Columns:
    - token: The word/token
    - study_freq: Frequency in study corpus
    - ref_freq: Frequency in reference corpus
    - ll: Log-likelihood ratio (Rayson)
    - or: Odds ratio
    - pd: Percent difference
    - bf: Bayes factor

Examples:
    # Basic comparison
    python -m jargon.keyness.compare_freqs \\
        -s pubmed.json -r leipzig.json -o comparison.tsv

    # Custom smoothing and specific metrics
    python -m jargon.keyness.compare_freqs \\
        -s pubmed.json -r leipzig.json -o comparison.tsv \\
        --smoothing 0.1 --metrics ll,or

    # With transformations (lowercase and strip whitespace)
    python -m jargon.keyness.compare_freqs \\
        -s pubmed.json -r leipzig.json -o comparison.tsv \\
        --lowercase --strip

    # Without UNK_ref row
    python -m jargon.keyness.compare_freqs \\
        -s medical.json -r general.json -o results.tsv \\
        --no-include-unk
        """
    )

    # Required arguments
    parser.add_argument(
        '-s', '--study',
        type=str,
        required=True,
        help='Path to study corpus JSON frequency file'
    )
    parser.add_argument(
        '-r', '--reference',
        type=str,
        required=True,
        help='Path to reference corpus JSON frequency file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output TSV file path'
    )

    # Optional arguments
    parser.add_argument(
        '--smoothing',
        type=float,
        default=0.1,
        help='Smoothing parameter for contingency tables (default: 0.1)'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        default='ll,or,pd,bf',
        help='Comma-separated metrics to compute (default: ll,or,pd,bf). '
             'Available: ll, or, pd, bf'
    )
    parser.add_argument(
        '--lowercase',
        action='store_true',
        help='Convert all tokens to lowercase and combine counts '
             '(e.g., "Name" + "name" -> "name")'
    )
    parser.add_argument(
        '--strip',
        action='store_true',
        help='Strip leading/trailing whitespace from tokens and combine counts '
             '(e.g., " word " + "word" -> "word")'
    )
    parser.add_argument(
        '--include-unk',
        dest='include_unk',
        action='store_true',
        default=True,
        help='Include UNK_ref row for reference-only tokens (default: True)'
    )
    parser.add_argument(
        '--no-include-unk',
        dest='include_unk',
        action='store_false',
        help='Exclude UNK_ref row'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    return parser.parse_args()


def validate_metrics(metrics_str: str) -> list[str]:
    """
    Validate and parse metrics string.

    Parameters
    ----------
    metrics_str : str
        Comma-separated metric names.

    Returns
    -------
    list[str]
        List of valid metric names.

    Raises
    ------
    ValueError
        If any metric is invalid.
    """
    metrics = [m.strip() for m in metrics_str.split(',')]
    invalid = [m for m in metrics if m not in AVAILABLE_METRICS]

    if invalid:
        raise ValueError(
            f"Invalid metrics: {', '.join(invalid)}. "
            f"Available: {', '.join(AVAILABLE_METRICS.keys())}"
        )

    return metrics


def compute_metrics(
    contingency_table,
    metrics: list[str],
    verbose: bool = False
) -> dict[str, float]:
    """
    Compute requested metrics from contingency table.

    Parameters
    ----------
    contingency_table : np.ndarray
        2x2 contingency table.
    metrics : list[str]
        Metrics to compute.
    verbose : bool, optional
        Print computation details.

    Returns
    -------
    dict[str, float]
        Dictionary of metric names to values.
    """
    results = {}

    for metric in metrics:
        if metric in AVAILABLE_METRICS:
            metric_name, metric_func = AVAILABLE_METRICS[metric]
            try:
                value = metric_func(contingency_table)
                results[metric] = value
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to compute {metric_name}: {e}")
                results[metric] = float('nan')

    return results


def main():
    """Main function."""
    args = parse_args()

    # Validate arguments
    try:
        metrics = validate_metrics(args.metrics)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.smoothing < 0:
        print(f"Error: Smoothing must be >= 0, got {args.smoothing}", file=sys.stderr)
        sys.exit(1)

    # Convert paths
    study_path = Path(args.study)
    ref_path = Path(args.reference)
    output_path = Path(args.output)

    if args.verbose:
        print(f"Study corpus: {study_path}")
        print(f"Reference corpus: {ref_path}")
        print(f"Output: {output_path}")
        print(f"Smoothing: {args.smoothing}")
        print(f"Metrics: {', '.join(metrics)}")
        if args.lowercase or args.strip:
            transforms = []
            if args.strip:
                transforms.append("strip")
            if args.lowercase:
                transforms.append("lowercase")
            print(f"Transformations: {', '.join(transforms)}")
        print()

    # Load frequency files
    try:
        if args.verbose:
            print("Loading study corpus...")
        study_freqs = load_frequency_json(study_path)

        if args.verbose:
            print("Loading reference corpus...")
        ref_freqs = load_frequency_json(ref_path)
    except Exception as e:
        print(f"Error loading frequency files: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply transformations if requested
    if args.lowercase or args.strip:
        if args.verbose:
            print("Applying transformations...")
            print(f"  Study corpus before: {len(study_freqs):,} unique tokens")
            print(f"  Reference corpus before: {len(ref_freqs):,} unique tokens")
            print()

        study_freqs = transform_frequencies(
            study_freqs,
            lowercase=args.lowercase,
            strip=args.strip,
            show_progress=args.verbose,
            desc="Transforming study corpus"
        )
        ref_freqs = transform_frequencies(
            ref_freqs,
            lowercase=args.lowercase,
            strip=args.strip,
            show_progress=args.verbose,
            desc="Transforming reference corpus"
        )

        if args.verbose:
            print(f"\n  Study corpus after: {len(study_freqs):,} unique tokens")
            print(f"  Reference corpus after: {len(ref_freqs):,} unique tokens")
            print()

    # Compute corpus statistics
    study_total, study_vocab = compute_corpus_stats(study_freqs)
    ref_total, ref_vocab = compute_corpus_stats(ref_freqs)

    if args.verbose:
        print(f"Final corpus statistics:")
        print(f"  Study corpus: {study_total:,} tokens, {study_vocab:,} unique")
        print(f"  Reference corpus: {ref_total:,} tokens, {ref_vocab:,} unique")
        print()

    # Process all tokens (union of both corpora)
    all_tokens = set(study_freqs.keys()) | set(ref_freqs.keys())

    if args.verbose:
        print(f"Processing {len(all_tokens)} unique tokens...")

    tokens_data = []

    for token in tqdm(all_tokens, desc="Computing metrics", disable=not args.verbose):
        # Build contingency table
        ct = build_contingency_table(
            token,
            study_freqs,
            ref_freqs,
            study_total,
            ref_total,
            study_vocab,
            ref_vocab,
            smoothing=args.smoothing
        )

        # Compute requested metrics
        metrics_dict = compute_metrics(ct, metrics, verbose=args.verbose)

        # Compute relative frequencies
        study_freq_abs = study_freqs.get(token, 0)
        ref_freq_abs = ref_freqs.get(token, 0)
        study_rel_freq = study_freq_abs / study_total if study_total > 0 else 0.0
        ref_rel_freq = ref_freq_abs / ref_total if ref_total > 0 else 0.0

        # Add to results
        tokens_data.append({
            'token': token,
            'study_freq': study_freq_abs,
            'ref_freq': ref_freq_abs,
            'study_rel_freq': study_rel_freq,
            'ref_rel_freq': ref_rel_freq,
            **metrics_dict
        })

    # Write output
    try:
        write_comparison_tsv(output_path, tokens_data, metrics)
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print(f"\nComparison complete!")
    print(f"  Tokens processed: {len(all_tokens):,}")
    print(f"  Total rows: {len(tokens_data):,}")
    print(f"  Output written to: {output_path}")


if __name__ == '__main__':
    main()
