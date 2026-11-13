
from abc import ABCMeta, abstractmethod
import numpy as np


class Optimizer(metaclass=ABCMeta):
    """Abstract base class for optimization algorithms.

    Optimizers update the model parameters based on the gradients
    calculated during backpropagation.
    
    Parameters
    ----------
    learning_rate : float
        Step size used to update the model parameters.
    """

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w, grad_loss_w):
        """Update the model parameters using the calculated gradients.

        Parameters
        ----------
        w : numpy.ndarray
            Current model parameters.
        grad_loss_w : numpy.ndarray
            Gradient of the loss with respect to the parameters.

        Returns
        -------
        numpy.ndarray
            Updated model parameters.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer with momentum.

    Momentum uses a fraction of the previous gradient to smooth parameter
    updates and accelerate convergence.

    Parameters
    ----------
    learning_rate : float, optional
        Step size used to update the parameters. Default is 0.01.
    momentum : float, optional
        Contribution of the previous retained gradient. Default is 0.90.
    """

    def __init__(self, learning_rate=0.01, momentum=0.90):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w, grad_loss_w):
        """Update the parameters using SGD with momentum.

        Parameters
        ----------
        w : numpy.ndarray
            Current model parameters.
        grad_loss_w : numpy.ndarray
            Gradient of the loss with respect to the parameters.

        Returns
        -------
        numpy.ndarray
            Updated model parameters.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros_like(w)
        self.retained_gradient = (self.momentum * self.retained_gradient
                                  + (1 - self.momentum) * grad_loss_w)
        return w - self.learning_rate * self.retained_gradient


class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) optimizer.

    Adam combines the advantages of momentum and adaptive learning rates
    by maintaining estimates of the first and second moments of the gradients.

    Parameters
    ----------
    learning_rate : float, optional
        Step size used to update the parameters. Default is 0.001.
    beta_1 : float, optional
        Exponential decay rate for the first moment estimates. Default is 0.9.
    beta_2 : float, optional
        Exponential decay rate for the second moment estimates. Default is 0.999.
    epsilon : float, optional
        Small constant added to the denominator for numerical stability.
        Default is 1e-8.
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, grad_loss_w):
        """Update the parameters using the Adam optimization algorithm.

        Parameters
        ----------
        w : numpy.ndarray
            Current model parameters.
        grad_loss_w : numpy.ndarray
            Gradient of the loss with respect to the parameters.

        Returns
        -------
        numpy.ndarray
            Updated model parameters.
        """
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        self.t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * grad_loss_w ** 2
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)
        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
