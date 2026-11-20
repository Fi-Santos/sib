import numpy as np
from neural_networks.neural_network import NeuralNetwork
from neural_networks.optimizers import SGD
from metrics.accuracy import accuracy
from neural_networks.losses import BinaryCrossEntropy
from neural_networks.activations import ReLUActivation, SigmoidActivation
from neural_networks.layers import DenseLayer


def build_network(learning_rate, n_hidden):
    """Build a neural network with the specified hyperparameters.

    Parameters
    ----------
    learning_rate : float
        Learning rate used by the optimizer.
    n_hidden : int
        Number of neurons in the hidden layer.

    Returns
    -------
    NeuralNetwork
        A new neural network with the specified architecture and hyperparameters.
    """
    np.random.seed(0)

    net = NeuralNetwork(
        epochs=100,
        batch_size=16,
        optimizer=SGD,
        learning_rate=learning_rate,
        momentum=0.9,
        loss=BinaryCrossEntropy,
        metric=accuracy
    )

    net.add(DenseLayer(n_hidden, (2,)))
    net.add(ReLUActivation())
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())

    return net