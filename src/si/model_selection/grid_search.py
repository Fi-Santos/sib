import itertools
import numpy as np
from data.dataset import Dataset
from model_selection._search_utils import k_fold_indices, cross_validate_configuration
from model_selection.randomized_serach import log_unifrom, sample_configuration,randomized_search_cv
from model_selection.Builder_function import build_network


def grid_search_cv(model, dataset: Dataset, hyperparameter_grid: dict[str, list], scoring: callable = None, cv: int=5, seed: int=None):
    """
    - hyperparameters – list with every configuration tested;
    - scores – mean validation score of each configuration;
    - best_hyperparameters and best_score – the configuration with the highest mean score.
    """
    #cv – number of folds;
    #- seed –seed used to create the folds.
    # -> Create the k folds once

    #Compute all combinations of the hyperparameter values (itertools.product)
    hyperparameters = list(itertools.product(*hyperparameter_grid.values()))

    for combination in hyperparameters:
        for fold in range(len(cv)):
            model = build_network(combination[0], combination[1])
    #       fit it on the training folds and score it on the validation fold
        configuration = dict(zip(hyperparameter_grid.keys(), combination))
        # Salvar a media dos k scores


    # Select the configuration with the highest mean score
    # scores e hyperparametros

    #Return the results dictionary -> best_hyperparameters and best_score – the configuration with the highest mean score.