
from abc import ABCMeta, abstractmethod
import numpy as np


class LossFunction(metaclass=ABCMeta):
    """Abstract base class for loss functions.

    Loss functions measure the difference between the predicted values and the ground-truth values. 
    Their derivatives are used during backpropagation to update the network parameters.
    """

    @abstractmethod
    def loss(self, y_true, y_pred):
        """Calculate the loss between true and predicted values.

        Parameters
        ----------
        y_true : numpy.ndarray
            Ground-truth values.
        y_pred : numpy.ndarray
            Predicted values.

        Returns
        -------
        float
            Loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        """Calculate the derivative of the loss with respect to predictions.

        Parameters
        ----------
        y_true : numpy.ndarray
            Ground-truth values.
        y_pred : numpy.ndarray
            Predicted values.

        Returns
        -------
        numpy.ndarray
            Gradient of the loss with respect to the predicted values.
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """Mean Squared Error (MSE) loss function.

    Calculates the average squared difference between true and predicted values. 
    Commonly used for regression tasks.
    """

    def loss(self, y_true, y_pred):
        """Calculate the mean squared error.

        Parameters
        ----------
        y_true : numpy.ndarray
            Ground-truth values.
        y_pred : numpy.ndarray
            Predicted values.

        Returns
        -------
        float
            Mean squared error.
        """
        return np.sum((y_true - y_pred) ** 2) / y_true.size

    def derivative(self, y_true, y_pred):
        """Calculate the gradient of the mean squared error.

        Parameters
        ----------
        y_true : numpy.ndarray
            Ground-truth values.
        y_pred : numpy.ndarray
            Predicted values.

        Returns
        -------
        numpy.ndarray
            Gradient of the loss with respect to the predicted values.
        """
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """Binary Cross-Entropy (BCE) loss function.

    Measures the difference between true binary labels and predicted probabilities. 
    Predictions are clipped to avoid numerical issues when computing logarithms.
    """

    def loss(self, y_true, y_pred):
        """Calculate the binary cross-entropy loss.

        Parameters
        ----------
        y_true : numpy.ndarray
            Ground-truth binary labels.
        y_pred : numpy.ndarray
            Predicted probabilities.

        Returns
        -------
        float
            Binary cross-entropy loss.
        """
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true, y_pred):
        """Calculate the gradient of the binary cross-entropy loss.

        Parameters
        ----------
        y_true : numpy.ndarray
            Ground-truth binary labels.
        y_pred : numpy.ndarray
            Predicted probabilities.

        Returns
        -------
        numpy.ndarray
            Gradient of the loss with respect to the predicted probabilities.
        """
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / p + (1 - y_true) / (1 - p)


class CategoricalCrossEntropy(LossFunction):
    """Categorical Cross-Entropy (CCE) loss function.

    Measures the difference between true class labels and predicted class probabilities. 
    Typically used with one-hot encoded labels.
    Predictions are clipped to avoid numerical issues when computing logarithms.
    """

    def loss(self, y_true, y_pred):
        """Calculate the categorical cross-entropy loss.

        Parameters
        ----------
        y_true : numpy.ndarray
            Ground-truth class labels, typically one-hot encoded.
        y_pred : numpy.ndarray
            Predicted class probabilities.

        Returns
        -------
        float
            Categorical cross-entropy loss.
        """
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p))

    def derivative(self, y_true, y_pred):
        """Calculate the gradient of the categorical cross-entropy loss.

        Parameters
        ----------
        y_true : numpy.ndarray
            Ground-truth class labels, typically one-hot encoded.
        y_pred : numpy.ndarray
            Predicted class probabilities.

        Returns
        -------
        numpy.ndarray
            Gradient of the loss with respect to the predicted probabilities.
        """
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / p
