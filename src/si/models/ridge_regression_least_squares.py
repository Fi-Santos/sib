import numpy as np
from si.data.dataset import Dataset
from si.metrics.mse import mse

class RidgeRegressionLeastSquares:
    """
    Ridge Regression model using the Least Squares method.

    Parameters
    ----------
    l2_penalty : float
        Regularization parameter used to penalize large coefficients.
    scale : bool
        Whether to standardize the input features before fitting the model.

    Attributes
    ----------
    theta : numpy.ndarray
        Coefficients estimated for each input feature.
    theta_zero : float
        Intercept of the model.
    mean : numpy.ndarray
        Mean of each feature, used when scaling is enabled.
    std : numpy.ndarray
        Standard deviation of each feature, used when scaling is enabled.
    """

    def __init__(self, l2_penalty, scale):
        self.l2_penalty = l2_penalty
        self.scale = scale

        self.theta = None
        self.theta_zero = 0
        self.mean = None
        self.std = None

    def fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Estimates the model parameters using Ridge Least Squares.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the input features and target values.

        Returns
        -------
        RidgeRegressionLeastSquares
            The fitted model.
        """
        if self.scale == True:
            self.mean = np.mean(dataset.X, axis=0)
            self.std = np.std(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        X = np.c_[np.ones(X.shape[0]), X]

        penalty = self.l2_penalty * np.eye(X.shape[1])
        penalty[0, 0] = 0

        parameters = np.linalg.inv(X.T @ X + penalty) @ X.T @ dataset.y

        self.theta_zero = parameters[0]
        self.theta = parameters[1:]

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the target values using the estimated model parameters.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the input features to predict.

        Returns
        -------
        numpy.ndarray
            Predicted target values.
        """
        X = dataset.X

        if self.scale == True:
            X = (X - self.mean) / self.std
        else:
            X = dataset.X

        X = np.c_[np.ones(X.shape[0]), X]

        parameters = np.r_[self.theta_zero, self.theta]
        y_pred = X @ parameters

        return y_pred

    def score(self, dataset: Dataset) -> float:
        """
        Calculates the mean squared error between the real and predicted values.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to evaluate the model.

        Returns
        -------
        float
            Mean squared error between the real and predicted values.
        """
        y_pred = self.predict(dataset)
        score = mse(dataset.y, y_pred)

        return score


    
if __name__ == '__main__':
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ])

    y = np.array([3, 5, 7, 9])

    dataset = Dataset(X, y)

    model = RidgeRegressionLeastSquares(
        l2_penalty=0.1,
        scale=False
    )

    model.fit(dataset)

    print("Theta zero:", model.theta_zero)
    print("Theta:", model.theta)
    print("Predictions:", model.predict(dataset))
    print("Score:", model.score(dataset))