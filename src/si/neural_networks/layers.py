from abc import ABCMeta, abstractmethod
import numpy as np


class Layer(metaclass=ABCMeta):
    """
    Base class for all neural network layers.

    Defines the abstract methods that specific layers must implement and
    provides shared methods for handling input shapes and layer names.
    """

    @abstractmethod
    def forward_propagation(self, input, training):
        """Compute the layer output for a given input."""
        raise NotImplementedError

    # @abstractmethod
    # def backward_propagation(self):
    #     """Compute the gradient of the loss with respect to the layer input."""
    #     raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        """Return the shape of the layer output."""
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        """Return the number of trainable parameters in the layer."""
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        """Set the shape of the layer input."""
        self._input_shape = input_shape

    def input_shape(self):
        """Return the shape of the layer input."""
        return self._input_shape

    def layer_name(self):
        """Return the class name of the layer."""
        return self.__class__.__name__


class DenseLayer(Layer):
    """
    Fully connected neural network layer.

    Parameters
    ----------
    n_units : int
        Number of neurons in the layer.
    input_shape : tuple
        Shape of the input for a single example.
    """

    def __init__(self, n_units, input_shape):
        """Initialize the layer configuration."""
        super().__init__()
        self.n_units = n_units
        self.set_input_shape(input_shape)

    def initialize(self, optimizer=None):
        """
        Initialize the weights and biases of the layer.

        Weights are randomly initialized between -0.5 and 0.5,
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
        #Initialize the optimizer for both weights and biases.
        #self.w_opt = copy.deepcopy(optimizer) 
        #self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self):
        """Return the total number of weights and biases."""
        return self.weights.size + self.bias.size

    def forward_propagation(self, input, training=None):
        """
        Compute the layer output using the input, weights, and biases.

        Parameters
        ----------
        input : numpy.ndarray
            Input data for the layer.
        training : bool, optional
            Indicates whether the layer is being used during training.

        Returns
        -------
        numpy.ndarray
            Output of the layer.
        """
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output
    
    
    def backward_propagation(self):
        #input_error = dE/dX (ANTES DE AUTALISAR OS ERROS)
        #wiegths_error = dE/dW
        #bias_erros = dE/dB
        #autualisar os pesos e os bias com self.w_opt e self.b_opt
        # devolver input_error  
        return None


    def output_shape(self):
        """Return the output shape for a single example."""
        return (self.n_units,)

"""
if __name__ == "__main__":
    layer = DenseLayer(4, (3,)).initialize()
    out = layer.forward_propagation(np.random.rand(10, 3))

    assert out.shape == (10, 4)
    assert layer.parameters() == 16
    assert layer.output_shape() == (4,)
"""

class ActivationLayer(Layer):
    def __init__(self, n_units, input_shape):
        super().__init__()
        self.n_units = n_units
        self.set_input_shape(input_shape)

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError 

    def parameters(self):
        return 0 #activation layers do not have learnable parameters

    def forward_propagation(self, input, training=None):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output
    
    def backward_propagation(self):
        #input_error = dE/dX (ANTES DE AUTALISAR OS ERROS)
        #wiegths_error = dE/dW
        #bias_erros = dE/dB
        #autualisar os pesos e os bias com self.w_opt e self.b_opt
        # devolver input_error  
        return None

    def output_shape(self):
        return self.input_shape #activation layers do not change the shape of the input


class ReLUActivation(ActivationLayer):
    def __init__(self):
        super().__init__()
    def activation_function(self):
        return None #applies the ReLU function on the input (argument)
    
    def derivative(self):
        return None #applies the ReLU derivative on the input (argument)


class SigmoidActivation(ActivationLayer):
    def __init__(self):
        super().__init__()
    def activation_function(self): 
        return None #applies the sigmoid function on the input (argument)
    
    def derivative(self):
        return None #applies the sigmoid derivative on the input (argument)



class Dropout(Layer):
    """
    - mask - binomial mask that sets some inputs to 0 based on the probability;
    - input - the input of the layer;
    - output - the output of the layer;
    """
    def __init__(self, probability):
        super().__init__(probability)
        self.probability = probability

    def forward_propagation(self, input, training=None):
        #If we are in training mode:
        self.input = input
        size = len(self.input)
        scaling_factor = (1 / (1 - self.probability)) 
        self.mask = (1 - self.probability) @ size 
        self.output = input * self.mask * scaling_factor
        #return self.output
        
        #If we are in inference mode:
        #return input #input is not changed during inference
    
    def backward_propagation(self, output_error):
        return output_error * self.mask
 
    def output_shape(self):
        return self.input_shape #dropout does not change the shape of the data

    def parameters(self):
        return 0 #dropout layers do not have learnable parameters