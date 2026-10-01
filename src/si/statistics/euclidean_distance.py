import numpy as np
import math

def euclidean_distance(sample, dataset_samples):
    """
    Calculates the Euclidean distance between a sample and all samplesin the dataset.

    Parameters
    ----------
    sample : array-like
        Sample for which the distances are calculated.
    dataset_samples : array-like
        Samples from the dataset used to calculate the distances.

    Returns
    -------
    distances : numpy.ndarray
        Array containing the Euclidean distance between the sample and each sample in the dataset.
    """

    distances = []

    for sample_2 in dataset_samples:
        dis = math.dist(sample, sample_2)
        distances.append(dis)

    distances = np.array(distances)
    return distances
