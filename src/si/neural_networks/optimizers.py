from abc import ABCMeta, abstractmethod
import numpy as np

class Optimizer:
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self):
    #to update the parameters (weights or biases) of the 
    #model, it receives the current parameters of a layer 
    # and the gradient of the loss function with respect to the parameters
        return None


class SGD(Optimizer):
    def __init__(self, learning_rate, momentum):
        super().__init__(learning_rate, momentum)
        self.momentum = momentum
        self.learning_rate = learning_rate

    def update(self, w, grad_loss_w):
        #Update the retained_gradient, computes and returns the updated weights
        if not self.retained_gradient:
            self.retained_gradient = np.zeros((1, len(self.w)))

        retained_gradient = None
        new_weights = None
        return None


class Adam(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)