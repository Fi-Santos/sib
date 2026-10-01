from typing import Callable, Union

import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance
from si.statistics.Manhattan_distance import Manhattan_distance
from si.statistics.Minkowski_distance import Minkowski_distance


class KNNClassifier(Model):
    """
    K-nearest neighbors classifier that estimates the class of a sample based on the classes of its k nearest training examples.

    Parameters
    ----------
    k : int
        Number of nearest training examples used to make the prediction.
    distance : Callable
        Function used to calculate the distance between samples.

    Attributes
    ----------
    k : int
        Number of nearest training examples considered.
    distance : Callable
        Function used to calculate the distance between samples.
    dataset : Dataset or None
        Training dataset stored during fitting.
    """

    def __init__(self, k, distance):
        """
        Initializes the KNN classifier.
        """
        super().__init__()
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset):
        """
        Stores the training dataset.

        Parameters
        ----------
        dataset : Dataset
            Training dataset used by the KNN classifier.

        Returns
        -------
        self : KNNClassifier
            Fitted KNN classifier.
        """
        self.dataset = dataset
        return self

    def _predict(self, dataset: Dataset):
        """
        Predicts the classes of the samples in the dataset using the k nearest training examples.

        Parameters
        ----------
        dataset : Dataset
            Test dataset used to make predictions.

        Returns
        -------
        predictions : numpy.ndarray
            Array containing the predicted classes for each test sample.
        """

        predictions = []

        for sample in dataset.X:
            distances = self.distance(sample, self.dataset.X)
            nearest_indexes = np.argsort(distances)[:self.k]
            nearest_labels = self.dataset.y[nearest_indexes]
            labels, counts = np.unique(nearest_labels, return_counts=True)
            predicted_class = labels[np.argmax(counts)]
            predictions.append(predicted_class)

        return np.array(predictions)

    def _score(self, dataset: Dataset):
        """
        Calculates the accuracy between the predicted and actual classes.

        Parameters
        ----------
        dataset : Dataset
            Test dataset used to calculate the score.

        Returns
        -------
        accuracy : float
            Accuracy of the predictions.
        """
        y_pred = self._predict(dataset)
        return accuracy(dataset.y, y_pred)

if __name__ == '__main__':
    X = np.array([
        [1, 1],
        [1, 2],
        [4, 4],
        [4, 5]
    ])

    y = np.array([0, 0, 1, 1])

    dataset = Dataset(X, y)

    knn_euclidean = KNNClassifier(k=3, distance=euclidean_distance)
    knn_manhattan = KNNClassifier(k=3, distance=Manhattan_distance)
    knn_minkowski = KNNClassifier(k=3, distance=Minkowski_distance)

    knn_euclidean.fit(dataset)
    knn_manhattan.fit(dataset)
    knn_minkowski.fit(dataset)

    print("Euclidean predictions:", knn_euclidean.predict(dataset))
    print("Euclidean accuracy:", knn_euclidean.score(dataset))

    print("Manhattan predictions:", knn_manhattan.predict(dataset))
    print("Manhattan accuracy:", knn_manhattan.score(dataset))

    print("Minkowski predictions:", knn_minkowski.predict(dataset))
    print("Minkowski accuracy:", knn_minkowski.score(dataset))

