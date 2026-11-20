from typing import Callable, Dict
import math
import numpy as np
from data.dataset import Dataset
from model_selection._search_utils import (k_fold_indices, cross_validate_configuration)


def log_uniform(low: float, high: float):
    """Create a logarithmic uniform sampling function.

    Parameters
    ----------
    low : float
        Lower bound of the sampling range.
    high : float
        Upper bound of the sampling range.

    Returns
    -------
    Callable
        Function that samples a value from a log-uniform distribution
        using a random number generator.
    """
    return lambda rng: 10 ** rng.uniform(math.log10(low), math.log10(high))


def sample_configuration(hyperparameter_distributions: Dict,
                         rng: np.random.Generator):
    """Randomly sample a hyperparameter configuration.

    Parameters
    ----------
    hyperparameter_distributions : Dict
        Dictionary containing hyperparameter names and their possible values or sampling functions.
    rng : np.random.Generator
        Random number generator used for sampling.

    Returns
    -------
    dict
        Dictionary containing the sampled hyperparameter values.
    """
    configuration = {}

    for name, distribution in hyperparameter_distributions.items():

        if callable(distribution):
            configuration[name] = distribution(rng)
        else:
            configuration[name] = rng.choice(distribution)

    return configuration


def randomized_search_cv(model, dataset, hyperparameter_distributions,
                         scoring=None, cv=5, n_iter=10, seed=None):
    """Perform randomized hyperparameter search using cross-validation.

    Parameters
    ----------
    model : callable
        Model builder function that creates a new model.
    dataset : Dataset
        Training dataset used for hyperparameter optimization.
    hyperparameter_distributions : Dict
        Dictionary containing hyperparameter names and their possible values or sampling functions.
    scoring : callable, optional
        Scoring function used to evaluate the model. If None, uses the model's score method.
    cv : int, default=5
        Number of folds used in cross-validation.
    n_iter : int, default=10
        Number of random configurations to evaluate.
    seed : int, optional
        Random seed used to generate folds and sample configurations.

    Returns
    -------
    dict
        Dictionary containing the tested hyperparameters, their mean validation scores, 
        the best hyperparameter configuration and its corresponding score.
    """
    # Create the k folds once and a random generator.
    folds = k_fold_indices(len(dataset.X), cv, seed)
    rng = np.random.default_rng(seed)

    hyperparameters = []
    scores = []

    for n in range(n_iter):

        # Sample one value per hyperparameter.
        configuration = sample_configuration(
            hyperparameter_distributions, rng
        )

        # Cross-validate and save the mean score.
        mean_score = cross_validate_configuration(
            model, dataset, configuration, folds, scoring
        )

        hyperparameters.append(configuration)
        scores.append(mean_score)

    # Select the configuration with the highest mean score.
    best_index = np.argmax(scores)

    return {
        "hyperparameters": hyperparameters,
        "scores": scores,
        "best_hyperparameters": hyperparameters[best_index],
        "best_score": scores[best_index]
    }