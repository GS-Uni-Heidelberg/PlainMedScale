import numpy as np
import math
from scipy import stats
from typing import Callable
import pandas as pd
import sys


def percent_difference(cont_table):
    """
    Calculate the percent difference of a word between two corpora.

    Parameters
    ----------
    word : str
        The word to calculate the percent difference for.
    study_corpus : Corpus
        The corpus to calculate the percent difference for.
    ref_corpus : Corpus
        The reference corpus to calculate the percent difference for.

    Returns
    -------
    percent_difference : float
        The percent difference of the word between the two corpora.
    """
    # Calculate the percent difference

    normal_freq_study = (
        cont_table[0, 0] /
        (cont_table[0, 0] + cont_table[1, 0])
    )
    normal_freq_ref = (
        cont_table[0, 1]
        / (cont_table[0, 1] + cont_table[1, 1])
    )

    try:
        percent_difference = (
            100 *
            (normal_freq_study - normal_freq_ref)
            / normal_freq_ref
        )
    except ZeroDivisionError:
        percent_difference = float('inf')

    return percent_difference


def log_likelihood_scipy(contingency_table):
    """
    Calculate the log likelihood of a contingency table.

    Parameters
    ----------
    contingency_table : array_like
        A 2x2 contingency table.

    Returns
    -------
    log_likelihood : float
        The log likelihood of the contingency table.
    """
    # Calculate the log likelihood
    log_likelihood = stats.chi2_contingency(
        contingency_table, lambda_="log-likelihood"
    )[0]

    # Calculate corpus sizes
    study_corpus_size = contingency_table[:, 0].sum()
    ref_corpus_size = contingency_table[:, 1].sum()

    # Change to negative log likelihood if more frequent
    # (relatively) in ref corpus
    if (contingency_table[0, 0] / study_corpus_size) < (
        contingency_table[0, 1] / ref_corpus_size
    ):
        log_likelihood = -log_likelihood

    return log_likelihood


def log_likelihood_rayson(contingency_table):
    study_corpus_size = contingency_table[:, 0].sum()
    ref_corpus_size = contingency_table[:, 1].sum()
    full_corpus_size = study_corpus_size + ref_corpus_size

    frequency_sum = contingency_table[0, :].sum()

    e1 = study_corpus_size * frequency_sum / full_corpus_size
    e2 = ref_corpus_size * frequency_sum / full_corpus_size

    log_likelihood = 2 * (
        contingency_table[0, 0] * math.log(contingency_table[0, 0] / e1)
        + contingency_table[0, 1] * math.log(contingency_table[0, 1] / e2)
    )

    # Change to negative log likelihood if more frequent
    # (relatively) in ref corpus
    if (contingency_table[0, 0] / study_corpus_size) < (
        contingency_table[0, 1] / ref_corpus_size
    ):
        log_likelihood = -log_likelihood

    return log_likelihood


def bayes_factor(contingency_table, ll_function=log_likelihood_rayson):

    ll = ll_function(contingency_table)

    bic = ll - math.log(np.sum(contingency_table))

    return bic


def odds_ratio(contingency_table):
    """
    Calculate the odds ratio of a contingency table.

    Parameters
    ----------
    contingency_table : array_like
        A 2x2 contingency table.

    Returns
    -------
    odds_ratio : float
        The odds ratio of the contingency table.
    """
    # Calculate the odds ratio
    odds_ratio = (
        (contingency_table[0, 0] * contingency_table[1, 1])
        / (contingency_table[0, 1] * contingency_table[1, 0])
    )

    return odds_ratio





def string_to_function(function_name: str) -> Callable:
    """Take a string and find the corresponding function in this module.

    Parameters:
        function_name (str): The name of the function to find.

    Returns:
        Callable: The function corresponding to the string.
    """
    # Get the current module
    current_module = sys.modules[__name__]

    # Check if the function name exists in the current module
    if hasattr(current_module, function_name):
        func = getattr(current_module, function_name)
        if callable(func):
            return func
        else:
            raise ValueError(
                f"{function_name} exists in the module but is not callable."
            )
    else:
        raise ValueError(f"Function {function_name} not found in module.")
