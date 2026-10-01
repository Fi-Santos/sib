import numpy as np
from scipy.spatial import distance

def Minkowski_distance(sample, dataset_samples):
    """
    Calculates the Minkowski distance between a sample and all samples in the dataset.

    Parameters
    ----------
    sample : array-like
        Sample for which the distances are calculated.
    dataset_samples : array-like
        Samples from the dataset used to calculate the distances.

    Returns
    -------
    distances : numpy.ndarray
        Array containing the Minkowski distance between the sample and each sample in the dataset.
    """

    distances = []
    point_1 = np.array(sample)

    for sample_2 in dataset_samples:
        point_2 = np.array(sample_2)
        dis = distance.minkowski(point_1, point_2)
        distances.append(dis)

    distances = np.array(distances)
    return distances
