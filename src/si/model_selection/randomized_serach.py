from typing import Callable, Dict
import math
from random import uniform
import numpy as np
from data.dataset import Dataset
from model_selection._search_utils import k_fold_indices, cross_validate_configuration


def log_unifrom(low: float, high:float):
    return  10 ** uniform(math.log10(low), math.log10(high))


def sample_configuration(hyperparametre_distributions : Dict, rng: np.random.Generator):
    return None


def randomized_search_cv(model, dataset, hyperparameter_distributions, scoring=None, cv=5, n_iter=10, seed=None):
    #create the k folds once and a random generator with the seed
    # k_fold_indices FUNCTION
    for n in range(n_iter):
        None
        # sample one value per hyperparameter (list → uniform choice; function → call it with the generator)
        # cross-validate and save the mean score
        # cross_validate_configuration FUNCTION

    #Return the results dictionary -> best_hyperparameters and best_score – the configuration with the highest mean score.
    return None
