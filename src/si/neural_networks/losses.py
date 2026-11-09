from abc import ABCMeta, abstractmethod

class LossFunction: 
    @abstractmethod
    def loss(self):
        return None #computes the loss based on the true and predicted labels

    @abstractmethod
    def derivative(sef):
        return None #computes the derivative of the loss based on the true and predicted labels


class MeanSquaredError(LossFunction):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def loss(self):
        return None #Formula no PP 

    def derivative(sef): 
        return None #Formula no PP 


class BinaryCrossEntropy(LossFunction):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def loss(self):
        return None #Formula no PP 

    def derivative(sef): 
        return None #Formula no PP 


class CategoricalCrossEntropy(LossFunction):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def loss(self):
        return None #Formula no PP 

    def derivative(sef): 
        return None #Formula no PP 