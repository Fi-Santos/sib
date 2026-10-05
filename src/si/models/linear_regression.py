import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

class RidgeRegression(Model):
    """
    Ridge Regression model using gradient descent.

    Parameters
    ----------
    l2_penalty : float
        L2 regularization parameter.
    alpha : float
        Learning rate used to update the model coefficients.
    max_iter : int
        Maximum number of iterations.
    pacience : int
        Maximum number of iterations without improvement allowed.
    scale : bool
        Whether to scale the input data or not.

    Attributes
    ----------
    l2_penalty : float
        L2 regularization parameter.
    alpha : float
        Learning rate used to update the model coefficients.
    max_iter : int
        Maximum number of iterations.
    patience : int
        Maximum number of iterations without improvement allowed.
    scale : bool
        Whether the input data is scaled or not.
    theta : numpy.ndarray
        Coefficients of the model for each feature.
    theta_zero : float
        Intercept of the model.
    mean : numpy.ndarray or None
        Mean of each feature used for scaling.
    std : numpy.ndarray or None
        Standard deviation of each feature used for scaling.
    cost_history : dict
        Cost function value for each iteration.
    """

    def __init__(self, l2_penalty, alpha, max_iter, patience, scale):
        super().__init__()

        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience #Ainda n esta a ser usado
        self.scale = scale

        self.theta = None
        self.theta_zero = 0
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        Estimates the model coefficients using gradient descent.

        Parameters
        ----------
        dataset : Dataset
            Training dataset used to fit the model.

        Returns
        -------
        self : RidgeRegression
            Fitted Ridge Regression model.
        """
        if self.scale == True:
            self.mean = np.mean(dataset.X, axis=0)
            self.std = np.std(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        self.theta = np.zeros(X.shape[1])
        self.theta_zero = 0

        for iteration in range(self.max_iter):
            y_pred = self.theta_zero + X @ self.theta

            error = y_pred - dataset.y

            gradient = (2 / len(dataset.y)) * (X.T @ error)
            gradient_zero = (2 / len(dataset.y)) * np.sum(error)

            gradient += self.l2_penalty * self.theta

            self.theta = self.theta - self.alpha * gradient
            self.theta_zero = self.theta_zero - self.alpha * gradient_zero

            cost = self.cost(dataset)
            self.cost_history[iteration] = cost

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the dependent variable y using the model coefficients.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the samples to predict.

        Returns
        -------
        y_pred : numpy.ndarray
            Predicted values of y.
        """
        X = dataset.X

        if self.scale == True:
            X = (X - self.mean) / self.std
        else:
            X = dataset.X

        y_pred = self.theta_zero + X @ self.theta

        return y_pred

    def _score(self, dataset: Dataset) -> float:
        """
        Calculates the mean squared error between real and predicted values.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to evaluate the model.

        Returns
        -------
        score : float
            Mean squared error between the real and predicted values.
        """
        y_pred = self._predict(dataset)
        score = mse(dataset.y, y_pred)

        return score

    def cost(self, dataset: Dataset) -> float:
        """
        Calculates the Ridge Regression cost function.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to calculate the cost.

        Returns
        -------
        cost : float
            Cost of the model, including the L2 regularization term.
        """
        y_pred = self._predict(dataset)

        m = len(dataset.y)
        squared_errors = (y_pred - dataset.y) ** 2
        regularization = self.l2_penalty * np.sum(self.theta ** 2)

        cost = (1 / (2 * m)) * (np.sum(squared_errors) + regularization)

        return cost


if __name__ == '__main__':
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ])

    y = np.array([3, 5, 7, 9])
    dataset = Dataset(X, y)

    model = RidgeRegression(
        l2_penalty=0.1,
        alpha=0.01,
        max_iter=1000,
        patience=10,
        scale=False
    )

    model.fit(dataset)

    print("Predictions:", model.predict(dataset))
    print("Score:", model.score(dataset))
    print("Cost:", model.cost(dataset))