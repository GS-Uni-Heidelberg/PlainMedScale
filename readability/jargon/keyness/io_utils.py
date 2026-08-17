"""
Input/output utilities for word frequency comparison.

This module provides functions for file I/O, naming conventions, and data formatting
for word frequency comparison workflows.
"""

import pandas as pd
from pathlib import Path
from typing import Union


def derive_comparison_name(
    study_path: Union[str, Path],
    ref_path: Union[str, Path]
) -> str:
    """
    Auto-derive comparison name from file paths.

    Strips common suffixes and file extensions to create a clean comparison name
    in the format "study-v-reference".

    Common suffixes stripped:
    - _lemma, _wordform, _lower, _freqs, _freq, _tokens

    Parameters
    ----------
    study_path : str or Path
        Path to study corpus file.
    ref_path : str or Path
        Path to reference corpus file.

    Returns
    -------
    str
        Comparison name in format "study-v-reference".

    Examples
    --------
    >>> derive_comparison_name("pubmed_lemma.json", "leipzig_lemma.json")
    'pubmed-v-leipzig'
    >>> derive_comparison_name("/path/to/medical_freqs.json", "general_freqs.json")
    'medical-v-general'
    """
    # Common suffixes to strip
    suffixes = ['_lemma', '_wordform', '_lower', '_freqs', '_freq', '_tokens']

    def clean_name(path: Union[str, Path]) -> str:
        """Remove common suffixes and extension from filename."""
        path = Path(path)
        name = path.stem  # Remove extension

        # Strip known suffixes
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break  # Only strip one suffix

        return name

    study_name = clean_name(study_path)
    ref_name = clean_name(ref_path)

    return f"{study_name}-v-{ref_name}"


def write_comparison_tsv(
    output_path: Union[str, Path],
    tokens_data: list[dict],
    metrics: list[str]
):
    """
    Write comparison results to TSV file.

    Parameters
    ----------
    output_path : str or Path
        Output file path.
    tokens_data : list[dict]
        List of dictionaries containing token data.
        Each dict should have keys: 'token', 'study_freq', 'ref_freq',
        and metric keys like 'll', 'or', 'pd', 'bf'.
    metrics : list[str]
        Which metrics to include in output.
        Options: 'll', 'or', 'pd', 'bf'.

    Raises
    ------
    ValueError
        If tokens_data is empty or has invalid format.

    Examples
    --------
    >>> data = [
    ...     {'token': 'plasma', 'study_freq': 262440, 'ref_freq': 2200,
    ...      'll': 561341.59, 'or': 244.53},
    ... ]
    >>> write_comparison_tsv('output.tsv', data, ['ll', 'or'])
    """
    output_path = Path(output_path)

    if not tokens_data:
        raise ValueError("tokens_data is empty")

    # Create dataframe
    df = pd.DataFrame(tokens_data)

    # Ensure required columns exist
    required_cols = ['token', 'study_freq', 'ref_freq', 'study_rel_freq', 'ref_rel_freq']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Select columns to output
    output_cols = ['token', 'study_freq', 'ref_freq', 'study_rel_freq', 'ref_rel_freq']
    for metric in metrics:
        if metric in df.columns:
            output_cols.append(metric)

    df = df[output_cols]

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to TSV
    df.to_csv(output_path, sep='\t', index=False)


def extract_comparison_name(filepath: Union[str, Path]) -> str:
    """
    Extract comparison name from filename.

    Parameters
    ----------
    filepath : str or Path
        Path to comparison file.

    Returns
    -------
    str
        Comparison name extracted from filename.

    Examples
    --------
    >>> extract_comparison_name("pubmed-v-leipzig.tsv")
    'pubmed-v-leipzig'
    >>> extract_comparison_name("/path/to/medical-v-general_lemma.tsv")
    'medical-v-general'
    """
    filepath = Path(filepath)
    name = filepath.stem  # Remove extension

    # Strip common suffixes that might have been added
    suffixes = ['_lemma', '_wordform', '_lower', '_comparison', '_stats']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    return name


def parse_comparison_name(name: str) -> tuple[str, str]:
    """
    Parse comparison name into corpus names.

    Expected format: "study-v-reference"

    Parameters
    ----------
    name : str
        Comparison name in format "study-v-reference".

    Returns
    -------
    tuple[str, str]
        (study_corpus_name, reference_corpus_name)

    Raises
    ------
    ValueError
        If name doesn't contain '-v-' separator.

    Examples
    --------
    >>> parse_comparison_name("pubmed-v-leipzig")
    ('pubmed', 'leipzig')
    >>> parse_comparison_name("medical-v-general")
    ('medical', 'general')
    """
    if '-v-' not in name:
        raise ValueError(
            f"Comparison name '{name}' doesn't follow 'study-v-reference' format"
        )

    parts = name.split('-v-')
    if len(parts) != 2:
        raise ValueError(
            f"Comparison name '{name}' has multiple '-v-' separators"
        )

    study_name, ref_name = parts
    return study_name, ref_name


def rename_columns_with_comparison(
    df: pd.DataFrame,
    comparison_name: str
) -> pd.DataFrame:
    """
    Rename DataFrame columns to include comparison name.

    Transforms:
    - 'study_freq' → 'abs-freq_{study}'
    - 'ref_freq' → 'abs-freq_{reference}'
    - 'll' → '{comparison}_ll'
    - 'or' → '{comparison}_or'
    - etc.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: study_freq, ref_freq, ll, or, etc.
    comparison_name : str
        Comparison name in format "study-v-reference".

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'study_freq': [100, 200],
    ...     'ref_freq': [10, 20],
    ...     'll': [123.45, 234.56],
    ...     'or': [5.5, 6.6]
    ... })
    >>> renamed = rename_columns_with_comparison(df, "pubmed-v-leipzig")
    >>> list(renamed.columns)
    ['abs-freq_pubmed', 'abs-freq_leipzig', 'pubmed-v-leipzig_ll', 'pubmed-v-leipzig_or']
    """
    # Parse comparison name to get corpus names
    try:
        study_name, ref_name = parse_comparison_name(comparison_name)
    except ValueError:
        # If parsing fails, use generic names
        study_name = "study"
        ref_name = "reference"

    # Create column mapping
    column_mapping = {}

    # Rename frequency columns
    if 'study_freq' in df.columns:
        column_mapping['study_freq'] = f'abs-freq_{study_name}'
    if 'ref_freq' in df.columns:
        column_mapping['ref_freq'] = f'abs-freq_{ref_name}'

    # Rename metric columns
    metric_cols = ['ll', 'or', 'pd', 'bf']
    for metric in metric_cols:
        if metric in df.columns:
            column_mapping[metric] = f'{comparison_name}_{metric}'

    # Apply renaming
    df_renamed = df.rename(columns=column_mapping)

    return df_renamed


def validate_comparison_files(filepaths: list[Union[str, Path]]):
    """
    Validate that all comparison files exist and have compatible formats.

    Parameters
    ----------
    filepaths : list[str or Path]
        List of file paths to validate.

    Raises
    ------
    FileNotFoundError
        If any file doesn't exist.
    ValueError
        If files have incompatible formats.

    Examples
    --------
    >>> validate_comparison_files(['comp1.tsv', 'comp2.tsv'])
    """
    filepaths = [Path(fp) for fp in filepaths]

    # Check all files exist
    missing = [fp for fp in filepaths if not fp.exists()]
    if missing:
        missing_str = '\n  '.join(str(fp) for fp in missing)
        raise FileNotFoundError(
            f"The following comparison files were not found:\n  {missing_str}"
        )

    # Check all files are readable and have 'token' column
    for filepath in filepaths:
        try:
            df = pd.read_csv(filepath, sep='\t', nrows=1)
            if 'token' not in df.columns:
                raise ValueError(
                    f"File {filepath} missing required 'token' column"
                )
        except Exception as e:
            raise ValueError(
                f"Error reading {filepath}: {e}"
            )
