from typing import Tuple
from si.data.dataset import Dataset
from scipy.stats import f_oneway
import numpy as np

def f_classification(dataset: Dataset):

    classes = dataset.get_classes()

    f_values = []
    p_values = []

    for feature in range(dataset.X.shape[1]):

        groups = []

        for class_label in classes:
            groups.append(dataset.X[dataset.y == class_label, feature])

        f, p = f_oneway(*groups)

        f_values.append(f)
        p_values.append(p)

    return tuple(f_values), tuple(p_values)
