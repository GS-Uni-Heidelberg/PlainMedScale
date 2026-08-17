"""
Simple Medical Jargon Detector using only SomaJo tokenization.

This module provides a simpler alternative to the tiktoken-based detector:
1. Tokenize text using SomaJo
2. Look up keyness scores directly for SomaJo tokens (no tiktoken encoding)
3. Classify as jargon based on threshold

This is faster and more straightforward than the tiktoken-based approach,
but doesn't account for subword tokenization.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Literal, Optional
from somajo import SoMaJo


class SimpleJargonDetector:
    """
    Detects medical jargon using direct SomaJo token keyness lookup.
    """

    def __init__(
        self,
        keyness_dir: Path | str,
        comparison: Literal["drks-v-deleipzig", "drks-v-dewiki", "pubmed-v-enleipzig", "pubmed-v-enwiki"] = "drks-v-deleipzig",
        preprocessing: Literal["default", "lower"] = "default"
    ):
        """
        Initialize the simple jargon detector.

        Args:
            keyness_dir: Path to directory containing keyness TSV files (e.g., 'keys/somajo')
            comparison: Which comparison to use for keyness scores
            preprocessing: Token preprocessing strategy (default or lower)
        """
        self.keyness_dir = Path(keyness_dir)
        self.comparison = comparison
        self.preprocessing = preprocessing

        # Load keyness dictionary
        self.keyness_dict = self._load_keyness_dict()

        # Initialize SomaJo tokenizer
        self.somajo = SoMaJo("de_CMC", xml_sentences=False)

    def _load_keyness_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Load keyness values from TSV file.

        Returns:
            Dictionary mapping token -> {metric: value}
        """
        keyness_file = self.keyness_dir / self.preprocessing / f"{self.comparison}.tsv"

        if not keyness_file.exists():
            raise FileNotFoundError(f"Keyness file not found: {keyness_file}")

        df = pd.read_csv(keyness_file, sep="\t")

        # Create dictionary: token -> {or, ll, pd, bf, study_freq, ref_freq}
        keyness_dict = {}
        for _, row in df.iterrows():
            token = str(row['token'])

            keyness_dict[token] = {
                'or': row.get('or', 1.0),
                'll': row.get('ll', 0.0),
                'pd': row.get('pd', 0.0),
                'bf': row.get('bf', 0.0),
                'study_freq': row.get('study_freq', 0),
                'ref_freq': row.get('ref_freq', 0),
                'study_rel_freq': row.get('study_rel_freq', 0.0),
                'ref_rel_freq': row.get('ref_rel_freq', 0.0)
            }

        return keyness_dict

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using SomaJo.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        sentences = self.somajo.tokenize_text([text])
        tokens = []
        for sentence in sentences:
            for token in sentence:
                tokens.append(token.text)
        return tokens

    def get_token_score(
        self,
        token: str,
        metric: Literal["or", "ll", "pd", "bf"] = "or"
    ) -> float:
        """
        Get keyness score for a token.

        Args:
            token: Input token
            metric: Which keyness metric to use

        Returns:
            Keyness score (default 1.0 for OR if not found, 0.0 for others)
        """
        # Apply preprocessing if needed
        if self.preprocessing == "lower":
            token = token.lower()

        if token in self.keyness_dict:
            return self.keyness_dict[token][metric]
        else:
            # Default to neutral score
            return 1.0 if metric == "or" else 0.0

    def identify_jargon(
        self,
        word: str,
        metric: Literal["or", "ll", "pd", "bf"] = "or"
    ) -> Dict[str, any]:
        """
        Check if a word is medical jargon.

        Args:
            word: Input word
            metric: Which keyness metric to use

        Returns:
            Dictionary with:
                - word: original word
                - score: keyness score
                - metric: metric used
                - in_dict: whether word was found in keyness dictionary
        """
        score = self.get_token_score(word, metric=metric)

        # Check if word was in dictionary
        lookup_word = word.lower() if self.preprocessing == "lower" else word
        in_dict = lookup_word in self.keyness_dict

        return {
            'word': word,
            'score': score,
            'metric': metric,
            'in_dict': in_dict
        }

    def analyze_text(
        self,
        text: str,
        metric: Literal["or", "ll", "pd", "bf"] = "or",
        threshold: Optional[float] = None
    ) -> List[Dict[str, any]]:
        """
        Analyze entire text for jargon terms.

        Args:
            text: Input text
            metric: Which keyness metric to use
            threshold: Optional threshold for jargon classification

        Returns:
            List of analysis results for each unique token, sorted by score (descending)
        """
        # Tokenize with SomaJo
        tokens = self.tokenize(text)

        # Get unique tokens
        unique_tokens = list(set(tokens))

        # Analyze each unique token
        results = []
        for token in unique_tokens:
            result = self.identify_jargon(token, metric=metric)

            # Add threshold classification if provided
            if threshold is not None:
                result['is_jargon'] = result['score'] > threshold

            # Add frequency in text
            result['freq_in_text'] = tokens.count(token)

            results.append(result)

        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)

        return results

    def get_jargon_terms(
        self,
        text: str,
        threshold: float,
        metric: Literal["or", "ll", "pd", "bf"] = "or",
        ref_freq_threshold: float = 0.001
    ) -> List[str]:
        """
        Get list of jargon terms from text.

        Args:
            text: Input text
            threshold: Threshold for jargon classification
            metric: Which keyness metric to use
            ref_freq_threshold: Reference relative frequency threshold.
                Words with ref_rel_freq > this value are filtered out
                even if they have high keyness (e.g., "Patient").
                Default: 0.001 (0.1%)

        Returns:
            List of jargon terms (sorted by score, descending)
        """
        results = self.analyze_text(text, metric=metric, threshold=threshold)

        jargon_terms = []
        for r in results:
            if not r.get('is_jargon', False):
                continue

            # Check if token is in dictionary
            lookup_word = r['word'].lower() if self.preprocessing == "lower" else r['word']

            if lookup_word not in self.keyness_dict:
                # Token not in dictionary -> treat as jargon (unknown medical term)
                jargon_terms.append(r['word'])
            else:
                # Token in dictionary -> check reference frequency
                ref_rel_freq = self.keyness_dict[lookup_word]['ref_rel_freq']
                if ref_rel_freq <= ref_freq_threshold:
                    # Low reference frequency -> is jargon
                    jargon_terms.append(r['word'])
                # else: high reference frequency -> skip (too common)

        return jargon_terms

    def compare_words(
        self,
        words: List[str],
        metric: Literal["or", "ll", "pd", "bf"] = "or"
    ) -> pd.DataFrame:
        """
        Compare keyness scores for a list of words.

        Args:
            words: List of words to compare
            metric: Which keyness metric to use

        Returns:
            DataFrame with words and their scores
        """
        results = []
        for word in words:
            result = self.identify_jargon(word, metric=metric)
            results.append(result)

        df = pd.DataFrame(results)
        df = df.sort_values('score', ascending=False)
        return df
