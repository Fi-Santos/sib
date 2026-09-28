from typing import Callable, Union

import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.models.mse import mse

"""
Testar com um dataset pequeno pq este metodo demora muito tempo a correr
"""

class RidgeRegressionLeastSquares:
    """
    Comentatarios e explicação

    Parametros
    ---------
    
    l2_penalty: ...
        ....
    scale: ...
        .... 

    Atributos
    ---------
    ...
    """
    
    def __init__(self, l2_penalty, scale):
        pass
    def fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Comentatarios e explicação
        """
        return None
    def predict(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Comentatarios e explicação
        """
        return None
    def score(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Comentatarios e explicação
        """
        return None

    
if __name__ == '__main__': # o construtor
    None