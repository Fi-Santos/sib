import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function

class LogisticRegression(Model):
    """
        Logistic Regression model using gradient descent.
    
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
        self.patience = patience 
        self.scale = scale
    
        self.theta = None
        self.theta_zero = 0
        self.mean = None
        self.std = None
        self.cost_history = {}
    
    def _fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Estimates the model coefficients using gradient descent.
        
        Parameters
                    
        ----------
        dataset : Dataset
        Training dataset used to fit the model.
        
        Returns
        -------
        self : LogisticRegression
        Fitted Logistic Regression model.
        """
        if self.scale == True:
            self.mean = np.mean(dataset.X, axis=0)
            self.std = np.std(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
    
        else:
            X = dataset.X
    
        self.theta = np.zeros(X.shape[1])
        self.theta_zero = 0

        best_cost = float('inf')
        index_patience = 0
        self.cost_history = {}
    
        for iteration in range(self.max_iter):
            y_pred = sigmoid_function(self.theta_zero + X @ self.theta)

            error = y_pred - dataset.y
    
            gradient = (1 / len(dataset.y)) * (X.T @ error)
            gradient_zero = (1 / len(dataset.y)) * np.sum(error)
    
            gradient += (self.l2_penalty / len(dataset.y)) * self.theta
    
            self.theta = self.theta - self.alpha * gradient
            self.theta_zero = self.theta_zero - self.alpha * gradient_zero
    
            cost = self.cost(dataset)
            self.cost_history[iteration] = cost

            if cost < best_cost:
                best_cost = cost
                index_patience = 0
            else:
                index_patience += 1

            if index_patience >= self.patience:
                break
    
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
    
        y_pred = sigmoid_function(self.theta_zero + X @ self.theta)
        y_pred = np.where(y_pred >= 0.5, 1, 0)

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
            Accuracy between the real and predicted values.
        """

        y_pred = self._predict(dataset)
        score = accuracy(dataset.y, y_pred)
    
        return score
    
    def cost(self, dataset: Dataset) -> float:
        """
        Calculates the Logistic Regression cost function.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to calculate the cost.

        Returns
        -------
        cost : float
            Cost of the model, including the L2 regularization term.
        """

        X = dataset.X

        if self.scale:
            X = (X - self.mean) / self.std

        y_pred = sigmoid_function(self.theta_zero + X @ self.theta)
    
        m = len(dataset.y)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        term1 = -(1 / m) * np.sum(
        dataset.y * np.log(y_pred) + (1 - dataset.y) * np.log(1 - y_pred))

        term2 = (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)

        return term1 + term2
    


if __name__ == '__main__':
    X = np.array([
        [1, 2],
        [2, 1],
        [4, 5],
        [5, 4]
    ])

    y = np.array([0, 0, 1, 1])
    dataset = Dataset(X, y)

    model = LogisticRegression(
    l2_penalty=0.1,
    alpha=0.1,
    max_iter=1000,
    patience=10,
    scale=False
    )

    model.fit(dataset)

    print("Predictions:", model.predict(dataset))
    print("Score:", model.score(dataset))
    print("Cost:", model.cost(dataset))