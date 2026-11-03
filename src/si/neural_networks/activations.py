
from abc import ABCMeta, abstractmethod
from neural_networks.layers import Layer
import numpy as np


class ActivationLayer(Layer):
    """
    Abstract base class for activation layers.

    Applies an activation function element-wise to the input and
    propagates gradients during backpropagation.
    """

    def __init__(self):
        """Initialize the activation layer."""
        super().__init__()

    @abstractmethod
    def activation_function(self, input):
        """
        Apply the activation function to the input.

        Parameters
        ----------
        input : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Activated output.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        """
        Compute the derivative of the activation function.

        Parameters
        ----------
        input : numpy.ndarray
            Input data used to calculate the derivative.

        Returns
        -------
        numpy.ndarray
            Derivative evaluated at the input values.
        """
        raise NotImplementedError

    def parameters(self):
        """Return zero, since activation layers have no trainable parameters."""
        return 0

    def forward_propagation(self, input, training=None):
        """
        Apply the activation function during forward propagation.

        Parameters
        ----------
        input : numpy.ndarray
            Input data.
        training : bool, optional
            Whether the layer is being used during training.

        Returns
        -------
        numpy.ndarray
            Activated output.
        """
        self.input = input
        self.output = self.activation_function(input)

        return self.output

    def backward_propagation(self, output_error):
        """
        Propagate the error through the activation function.

        The gradient is calculated using element-wise multiplication,
        since each output depends only on its corresponding input.

        Parameters
        ----------
        output_error : numpy.ndarray
            Gradient of the loss with respect to the layer output.

        Returns
        -------
        numpy.ndarray
            Gradient of the loss with respect to the layer input.
        """
        derivative = self.derivative(self.input)
        input_error = output_error * derivative

        return input_error

    def output_shape(self):
        """Return the input shape, since activation layers preserve dimensions."""
        return self.input_shape()




class ReLUActivation(ActivationLayer):
    """
    Rectified Linear Unit (ReLU) activation function.

    Returns zero for negative inputs and the input value otherwise.
    """

    def activation_function(self, input):
        """Apply the ReLU function element-wise."""
        return np.maximum(0, input)

    def derivative(self, input):
        """Compute the derivative of the ReLU function element-wise."""
        return np.where(input > 0, 1, 0)





class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.

    Maps input values to the range (0, 1).
    """

    def activation_function(self, input):
        """Apply the sigmoid function element-wise."""
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        """Compute the sigmoid derivative using its output values."""
        s = self.activation_function(input)
        return s * (1 - s)





class LeakyReLUActivation(ActivationLayer):
    """
    Leaky ReLU activation function.

    Allows a small, non-zero gradient for negative input values.

    Parameters
    ----------
    alpha : float, optional
        Slope for negative input values. Defaults to 0.01.
    """

    def __init__(self, alpha=0.01):
        """Initialize the negative slope parameter."""
        super().__init__()
        self.alpha = alpha

    def activation_function(self, input):
        """Apply the Leaky ReLU function element-wise."""
        return np.where(input > 0, input, self.alpha * input)

    def derivative(self, input):
        """Compute the derivative of the Leaky ReLU function."""
        return np.where(input > 0, 1, self.alpha)





class TanhActivation(ActivationLayer):
    """
    Hyperbolic tangent activation function.

    Maps input values to the range (-1, 1).
    """

    def activation_function(self, input):
        """Apply the hyperbolic tangent function element-wise."""
        return np.tanh(input)

    def derivative(self, input):
        """Compute the derivative of the hyperbolic tangent function."""
        t = self.activation_function(input)
        return 1 - t**2





class SoftmaxActivation(ActivationLayer):
    """
    Softmax activation function.

    Converts input values into a probability distribution along
    the last axis.
    """

    def activation_function(self, input):
        """
        Apply the numerically stabilized Softmax function.

        Subtracting the maximum input value helps prevent
        numerical overflow.

        Parameters
        ----------
        input : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Normalized output values along the last axis.
        """
        ex = np.exp(input - np.max(input, axis=-1, keepdims=True))
        return ex / np.sum(ex, axis=-1, keepdims=True)

    def derivative(self, input):
        """
        Compute the element-wise Softmax derivative approximation.

        Uses the diagonal expression s * (1 - s).
        """
        s = self.activation_function(input)
        return s * (1 - s)
