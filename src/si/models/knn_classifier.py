from typing import Callable, Union

import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.acuraçy import accuraçy
from si.statistics.euclidean.distance import eclidean_distance

class KNNClassifier(Model):
    """
    Comentatarios e explicação

    Parametros
    ---------
    k: ...
        ....
    distance: ...
        .... 

    Atributos
    ---------
    ...

    """
    def __init__(self, k, distance):
        return None 
    
    def _det_closest_label(self, dataset: Dataset):
        """
            Comentatarios e explicação
        """
        return None #a label das amostras mais proximas #return label
    
    def _predicte(self, dataset: Dataset):
        """
            Comentatarios e explicação
        """
        return None #prediction
    
    def _score(self, dataset: Dataset):
        """
            Comentatarios e explicação
        """
        return None

    def Manhattan_distance(self, dataset: Dataset):
        """
            Comentatarios e explicação
        """
        return None

    def Minkowski_distance(self, dataset: Dataset):
        """
            Comentatarios e explicação
        """
        return None
        

if __name__ == '__main__': # o construtor
    None


