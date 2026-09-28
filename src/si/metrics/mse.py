from typing import Callable, Union

import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset

class RidgeRegression(Model):
    """
    Comentatarios e explicação

    Parametros
    ---------
    
    l2_penalty: ...
        ....
    alpha: ...
        .... 
    max_iter: ...
        ....
    pacience: ...
        ....
    scale: ...
        .... 

    Atributos
    ---------
    ...

    """
    def __init__(self, l2_penalty, alpha, max_iter, pacience, scale):
        return None
    
    def _fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
            Comentatarios e explicação
        """
        return None
    
    def _predict(self, dataset: Dataset) -> 'RidgeRegression':
        """
            Comentatarios e explicação
        """
        return None
    
    def _score(self, dataset: Dataset) -> 'RidgeRegression':
        """
            Comentatarios e explicação
        """
        return None
    
    def _cost(self, dataset: Dataset) -> 'RidgeRegression':
        """
            Comentatarios e explicação
        """
        return None


if __name__ == '__main__': # o construtor
    None