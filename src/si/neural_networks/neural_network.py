

class NeuralNetwork():
    def __init__(self, epochs, batch_size, optimizer, learning_rate, verbose, loss, metric, kwargs):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.loss = loss
        self.metric = metric
        self.kwargs = kwargs

    def add(self):
        return None

    def fit(self):
        return None

    def predict(self):
        return None

    def score(self):
        return None