import itertools
import numpy as np
from data.dataset import Dataset
from model_selection._search_utils import k_fold_indices, cross_validate_configuration


def grid_search_cv(model, dataset: Dataset, hyperparameter_grid: dict,
                   scoring: callable = None, cv: int = 5, seed: int = None):
    """Perform grid search using cross-validation.

    Parameters
    ----------
    model : callable
        Model builder function that creates a new model.
    dataset : Dataset
        Training dataset used for hyperparameter optimization.
    hyperparameter_grid : dict
        Dictionary containing hyperparameter names and lists of values to evaluate.
    scoring : callable, optional
        Scoring function used to evaluate the model. If None, uses the model's score method.
    cv : int, default=5
        Number of folds used in cross-validation.
    seed : int, optional
        Random seed used to create the folds.

    Returns
    -------
    dict
        Dictionary containing all tested hyperparameter configurations, 
        their mean validation scores, 
        the best configuration and its corresponding score.
    """
    # Create the k folds once.
    folds = k_fold_indices(len(dataset.X), cv, seed)

    # Compute all combinations of hyperparameter values.
    configurations = list(
        itertools.product(*hyperparameter_grid.values())
    )

    hyperparameters = []
    scores = []

    for combination in configurations:

        configuration = dict(
            zip(hyperparameter_grid.keys(), combination)
        )

        # Cross-validate the current configuration.
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