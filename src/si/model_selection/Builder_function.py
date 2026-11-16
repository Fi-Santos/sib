import numpy as np
from neural_networks.neural_network import NeuralNetwork
from neural_networks.optimizers import SGD
from metrics.accuracy import accuracy
from neural_networks.losses import BinaryCrossEntropy
from neural_networks.activations import ReLUActivation, SigmoidActivation
from neural_networks.layers import DenseLayer

def build_network(learning_rate, n_hidden):
    np.random.seed(0)   # same initial weights for every configuration
    net = NeuralNetwork(epochs=100, batch_size=16, optimizer=SGD,
    learning_rate=learning_rate, momentum=0.9,
    loss=BinaryCrossEntropy, metric=accuracy)
    net.add(DenseLayer(n_hidden, (2,)))
    net.add(ReLUActivation())
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())
    return net