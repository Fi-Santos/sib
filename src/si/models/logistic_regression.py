import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.acuracy import acuracy
from si.statistics.sigmoid_function import sigmoid_function

class LogisticRegression(Model):
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
    
    def _fit(self, dataset: Dataset):
        if self.scale == True:
            self.mean = np.mean(dataset.X, axis=0)
            self.std = np.std(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
    
        else:
            X = dataset.X
    
        self.theta = np.zeros(X.shape[1])
        self.theta_zero = 0
    
        for iteration in range(self.max_iter):
            #y_pred = self.theta_zero + X @ self.theta
            """
            Estimates the value of Y using the sigmoid_function

            y_pred = 
            """

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
        X = dataset.X
    
        if self.scale == True:
            X = (X - self.mean) / self.std
        else:
            X = dataset.X
    
        y_pred = self.theta_zero + X @ self.theta
    
        return y_pred
    
    def _score(self, dataset: Dataset) -> float:
        y_pred = self._predict(dataset)
        score = acuracy(dataset.y, y_pred)
    
        return score
    
    def cost(self, dataset: Dataset) -> float:
        y_pred = self._predict(dataset)
    
        m = len(dataset.y)
        squared_errors = (y_pred - dataset.y) ** 2
        regularization = self.l2_penalty * np.sum(self.theta ** 2)
    
        cost = (1 / (2 * m)) * (np.sum(squared_errors) + regularization)
    
        return cost
    