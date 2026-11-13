import numpy as np
def accuracy(y_true, y_pred):
    """
    Calculates the accuracy by comparing the true labels with the predicted labels.
    """
    y_pred = (y_pred >= 0.5).astype(int)
    return np.mean(y_true == y_pred)