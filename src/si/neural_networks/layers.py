
from abc import ABCMeta, abstractmethod
import numpy as np
import copy


class Layer(metaclass=ABCMeta):
    """
    Abstract base class for neural network layers.

    Defines the common interface for all layers and provides methods
    for handling input shapes and retrieving layer information.
    """

    @abstractmethod
    def forward_propagation(self, input, training):
        """
        Compute the layer output during forward propagation.

        Parameters
        ----------
        input : numpy.ndarray
            Input data passed to the layer.
        training : bool
            Whether the layer is being used during training.

        Returns
        -------
        numpy.ndarray
            Output produced by the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error):
        """
        Compute the gradient of the loss with respect to the layer input.

        Parameters
        ----------
        output_error : numpy.ndarray
            Gradient of the loss with respect to the layer output.

        Returns
        -------
        numpy.ndarray
            Gradient of the loss with respect to the layer input.
        """
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        """Return the shape of the layer output."""
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        """Return the number of trainable parameters in the layer."""
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        """
        Set the expected input shape of the layer.

        Parameters
        ----------
        input_shape : tuple
            Shape of a single input example.
        """
        self._input_shape = input_shape

    def input_shape(self):
        """Return the shape of a single input example."""
        return self._input_shape

    def layer_name(self):
        """Return the name of the layer class."""
        return self.__class__.__name__





class DenseLayer(Layer):
    """
    Fully connected neural network layer.

    Each neuron is connected to every input feature.

    Parameters
    ----------
    n_units : int
        Number of neurons in the layer.
    input_shape : tuple
        Shape of a single input example.
    """

    def __init__(self, n_units, input_shape = None):
        """Initialize the layer configuration."""
        super().__init__()
        self.n_units = n_units
        self.set_input_shape(input_shape)

    def initialize(self, optimizer=None):
        """
        Initialize the weights, biases, and optimizers.

        Weights are initialized randomly between -0.5 and 0.5,
        while biases are initialized to zero.

        Parameters
        ----------
        optimizer : object, optional
            Optimizer used to update the weights and biases.

        Returns
        -------
        DenseLayer
            The initialized layer.
        """
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        self.bias = np.zeros((1, self.n_units))

        # Create independent optimizer instances for weights and biases.
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)

        return self

    def parameters(self):
        """Return the total number of weights and biases."""
        return self.weights.size + self.bias.size

    def forward_propagation(self, input, training=None):
        """
        Compute the layer output using a linear transformation.

        Parameters
        ----------
        input : numpy.ndarray
            Input data with shape (n_samples, n_features).
        training : bool, optional
            Indicates whether the layer is being used during training.

        Returns
        -------
        numpy.ndarray
            Layer output with shape (n_samples, n_units).
        """
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias

        return self.output

    def backward_propagation(self, output_error):
        """
        Compute gradients and update the layer parameters.

        Parameters
        ----------
        output_error : numpy.ndarray
            Gradient of the loss with respect to the layer output.

        Returns
        -------
        numpy.ndarray
            Gradient of the loss with respect to the layer input.
        """
        # Calculate gradients using the current weights.
        input_error = output_error @ self.weights.T
        weights_error = self.input.T @ output_error
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # Update weights and biases using their respective optimizers.
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.bias = self.b_opt.update(self.bias, bias_error)

        return input_error

    def output_shape(self):
        """Return the output shape for a single input example."""
        return (self.n_units,)





class Dropout(Layer):
    """
    Dropout regularization layer.

    Randomly disables a proportion of input units during training
    to help prevent overfitting.

    Parameters
    ----------
    probability : float
        Probability of setting an input unit to zero.
        Must be in the range [0, 1).
    """

    def __init__(self, probability):
        """Initialize the dropout probability."""
        super().__init__()

        if not 0 <= probability < 1:
            raise ValueError("probability must be between 0 and 1")

        self.probability = probability

    def forward_propagation(self, input, training=None):
        """
        Apply dropout during training or pass the input unchanged during inference.

        Inverted dropout scaling is used to maintain the expected
        activation values during training.

        Parameters
        ----------
        input : numpy.ndarray
            Input data to the layer.
        training : bool, optional
            If True, randomly mask input units. Otherwise, return
            the input unchanged.

        Returns
        -------
        numpy.ndarray
            Output after applying dropout, or the original input
            during inference.
        """
        if training:
            self.input = input

            scaling_factor = 1 / (1 - self.probability)
            self.mask = np.random.binomial(
                1, 1 - self.probability, input.shape
            )

            self.output = input * self.mask * scaling_factor
            return self.output

        # During inference, dropout is disabled.
        return input

    def backward_propagation(self, output_error):
        """
        Propagate the error through the dropout mask.

        Uses the same mask generated during forward propagation.

        Parameters
        ----------
        output_error : numpy.ndarray
            Gradient of the loss with respect to the layer output.

        Returns
        -------
        numpy.ndarray
            Gradient of the loss with respect to the layer input.
        """
        scaling_factor = 1 / (1 - self.probability)
        input_error = output_error * self.mask * scaling_factor

        return input_error

    def output_shape(self):
        """Return the input shape, since dropout does not change dimensions."""
        return self.input_shape()

    def parameters(self):
        """Return zero, since dropout has no trainable parameters."""
        return 0
