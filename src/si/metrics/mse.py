import numpy as np

def mse(y_pred, y_true):
    """
    Calculates the mean squared error between the predicted and true values.

    Parameters
    ----------
    y_true : array-like
        Real values of y.
    y_pred : array-like
        Predicted values of y.

    Returns
    -------
    mse : float
        Mean squared error between the predicted and true values.
    """
    squared_errors = (y_true - y_pred) ** 2
    return np.mean(squared_errors)