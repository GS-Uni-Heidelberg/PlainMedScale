"""
Contingency table utilities for word frequency comparison.

This module provides functions for loading word frequency data and building
contingency tables for statistical comparison between corpora.
"""

import json
import numpy as np
from pathlib import Path
from typing import Union
from tqdm import tqdm


def load_frequency_json(filepath: Union[str, Path]) -> dict[str, int]:
    """
    Load word frequencies from JSON file.

    Supports multiple JSON formats:
    - Nested: {"word_freqs_lemma": {token: freq, ...}}
    - Nested: {"word_freqs_wordform": {token: freq, ...}}
    - Flat: {token: freq, ...}

    Parameters
    ----------
    filepath : str or Path
        Path to JSON file containing word frequencies.

    Returns
    -------
    dict[str, int]
        Dictionary mapping tokens to their frequencies.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    JSONDecodeError
        If the file is not valid JSON.
    ValueError
        If the JSON format is not recognized.

    Examples
    --------
    >>> freqs = load_frequency_json("pubmed_lemma.json")
    >>> freqs['plasma']
    262440
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Frequency file not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in {filepath}: {e.msg}",
            e.doc, e.pos
        )

    # Validate and extract frequency dictionary
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected dictionary in {filepath}, got {type(data).__name__}"
        )

    # Check for nested format with 'word_freqs_lemma' or 'word_freqs_wordform'
    if 'word_freqs_lemma' in data:
        freq_dict = data['word_freqs_lemma']
    elif 'word_freqs_wordform' in data:
        freq_dict = data['word_freqs_wordform']
    else:
        # Assume flat format - validate all values are integers
        if all(isinstance(v, int) for v in data.values()):
            freq_dict = data
        else:
            raise ValueError(
                f"Unrecognized JSON format in {filepath}. "
                f"Expected nested format with 'word_freqs_lemma' or 'word_freqs_wordform', "
                f"or flat format with {token: frequency} pairs."
            )

    # Validate frequency dictionary
    if not freq_dict:
        raise ValueError(f"Empty frequency dictionary in {filepath}")

    # Ensure all frequencies are integers
    try:
        freq_dict = {k: int(v) for k, v in freq_dict.items()}
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Non-integer frequency values found in {filepath}: {e}"
        )

    return freq_dict


def compute_corpus_stats(freq_dict: dict[str, int]) -> tuple[int, int]:
    """
    Compute total token count and vocabulary size from frequency dictionary.

    Parameters
    ----------
    freq_dict : dict[str, int]
        Dictionary mapping tokens to their frequencies.

    Returns
    -------
    tuple[int, int]
        (total_tokens, vocab_size) where:
        - total_tokens: sum of all token frequencies
        - vocab_size: number of unique tokens

    Examples
    --------
    >>> freqs = {'the': 1000, 'a': 500, 'cat': 10}
    >>> total, vocab = compute_corpus_stats(freqs)
    >>> total, vocab
    (1510, 3)
    """
    total_tokens = sum(freq_dict.values())
    vocab_size = len(freq_dict)
    return total_tokens, vocab_size


def transform_frequencies(
    freq_dict: dict[str, int],
    lowercase: bool = False,
    strip: bool = False,
    show_progress: bool = False,
    desc: str = "Transforming"
) -> dict[str, int]:
    """
    Apply transformations to frequency dictionary.

    Transformations are applied in order: strip, then lowercase.

    Parameters
    ----------
    freq_dict : dict[str, int]
        Original frequency dictionary.
    lowercase : bool, optional
        Convert all tokens to lowercase and combine counts (default: False).
        Example: 'Name': 100, 'name': 50 -> 'name': 150
    strip : bool, optional
        Strip leading/trailing whitespace from tokens (default: False).
        Example: ' word ': 100, 'word': 50 -> 'word': 150
    show_progress : bool, optional
        Show progress bar using tqdm (default: False).
    desc : str, optional
        Description for progress bar (default: "Transforming").

    Returns
    -------
    dict[str, int]
        Transformed frequency dictionary with combined counts.

    Examples
    --------
    >>> freqs = {'Name': 100, 'name': 50, ' word ': 30}
    >>> transform_frequencies(freqs, lowercase=True, strip=True)
    {'name': 150, 'word': 30}
    """
    if not lowercase and not strip:
        # No transformations needed
        return freq_dict.copy()

    # Use tqdm if show_progress is True
    items = tqdm(freq_dict.items(), desc=desc, disable=not show_progress)

    transformed = {}

    for token, freq in items:
        # Apply transformations in order
        new_token = token

        if strip:
            new_token = new_token.strip()

        if lowercase:
            new_token = new_token.lower()

        # Combine counts for tokens that map to the same transformed form
        if new_token in transformed:
            transformed[new_token] += freq
        else:
            transformed[new_token] = freq

    return transformed


def build_contingency_table(
    token: str,
    study_freqs: dict[str, int],
    ref_freqs: dict[str, int],
    study_total: int,
    ref_total: int,
    study_vocab: int,
    ref_vocab: int,
    smoothing: float = 0.5
) -> np.ndarray:
    """
    Create smoothed 2x2 contingency table for a single token.

    The contingency table has the following structure:
        [[study_token_freq, ref_token_freq],
         [study_other_freqs, ref_other_freqs]]

    where study_other_freqs is the count of all other tokens in the study corpus,
    and ref_other_freqs is the count of all other tokens in the reference corpus.

    Smoothing is applied to avoid zero frequencies:
    - Token frequencies: add smoothing
    - Other frequencies: add smoothing * vocabulary_size

    Parameters
    ----------
    token : str
        The token to build the contingency table for.
    study_freqs : dict[str, int]
        Study corpus token frequencies.
    ref_freqs : dict[str, int]
        Reference corpus token frequencies.
    study_total : int
        Total tokens in study corpus.
    ref_total : int
        Total tokens in reference corpus.
    study_vocab : int
        Vocabulary size of study corpus.
    ref_vocab : int
        Vocabulary size of reference corpus.
    smoothing : float, optional
        Smoothing parameter (default: 0.5, Laplace smoothing).

    Returns
    -------
    np.ndarray
        2x2 contingency table as numpy array.

    Examples
    --------
    >>> ct = build_contingency_table(
    ...     'plasma', med_freqs, ref_freqs,
    ...     med_total, ref_total, med_vocab, ref_vocab,
    ...     smoothing=0.1
    ... )
    >>> ct.shape
    (2, 2)
    """
    if smoothing < 0:
        raise ValueError(f"Smoothing must be >= 0, got {smoothing}")

    contingency_table = np.zeros((2, 2))

    # First row: observed token frequencies (with smoothing)
    contingency_table[0, 0] = study_freqs.get(token, 0) + smoothing
    contingency_table[0, 1] = ref_freqs.get(token, 0) + smoothing

    # Second row: all other tokens (with vocabulary-scaled smoothing)
    contingency_table[1, 0] = (
        study_total - study_freqs.get(token, 0)
        + smoothing * study_vocab
    )
    contingency_table[1, 1] = (
        ref_total - ref_freqs.get(token, 0)
        + smoothing * ref_vocab
    )

    return contingency_table


def compute_unk_contingency(
    study_freqs: dict[str, int],
    ref_freqs: dict[str, int],
    study_total: int,
    ref_total: int,
    study_vocab: int,
    ref_vocab: int,
    smoothing: float = 0.5
) -> np.ndarray:
    """
    Create contingency table for UNK_ref (tokens only in reference corpus).

    This represents the aggregate mass of reference corpus tokens that do not
    appear in the study corpus. Useful for understanding the overall vocabulary
    difference between corpora.

    Parameters
    ----------
    study_freqs : dict[str, int]
        Study corpus token frequencies.
    ref_freqs : dict[str, int]
        Reference corpus token frequencies.
    study_total : int
        Total tokens in study corpus.
    ref_total : int
        Total tokens in reference corpus.
    study_vocab : int
        Vocabulary size of study corpus.
    ref_vocab : int
        Vocabulary size of reference corpus.
    smoothing : float, optional
        Smoothing parameter (default: 0.5).

    Returns
    -------
    np.ndarray
        2x2 contingency table for reference-only tokens.

    Examples
    --------
    >>> unk_ct = compute_unk_contingency(
    ...     med_freqs, ref_freqs,
    ...     med_total, ref_total,
    ...     med_vocab, ref_vocab,
    ...     smoothing=0.1
    ... )
    """
    if smoothing < 0:
        raise ValueError(f"Smoothing must be >= 0, got {smoothing}")

    # Calculate total frequency of reference tokens not in study corpus
    unk_ref_med_freq = 0
    unk_ref_ref_freq = sum(
        freq for token, freq in ref_freqs.items()
        if token not in study_freqs
    )

    contingency_table = np.array([
        [unk_ref_med_freq + smoothing, unk_ref_ref_freq + smoothing],
        [
            study_total - unk_ref_med_freq + smoothing * study_vocab,
            ref_total - unk_ref_ref_freq + smoothing * ref_vocab
        ]
    ])

    return contingency_table
