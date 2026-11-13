
import numpy as np
from neural_networks.losses import MeanSquaredError
from neural_networks.optimizers import SGD


class NeuralNetwork:
    """Neural network that connects layers and manages the training process.

    Estimated Parameters
    --------------------
    layers : list
        List containing the layers of the neural network.
    history : dict
        Dictionary containing the loss and metric at each epoch.
    """

    def __init__(self, epochs=100, batch_size=128, optimizer=SGD, learning_rate=0.01,
                 verbose=False, loss=MeanSquaredError, metric=None, **kwargs):
        """Initialize the neural network.

        Parameters
        ----------
        epochs : int, optional
            Number of epochs used for training. Default is 100.
        batch_size : int, optional
            Number of samples in each mini-batch. Default is 128.
        optimizer : Optimizer, optional
            Optimization algorithm used to update the weights. Default is SGD.
        learning_rate : float, optional
            Learning rate used by the optimizer. Default is 0.01.
        verbose : bool, optional
            Whether to display the loss and metric at each epoch.
            Default is False.
        loss : LossFunction, optional
            Loss function used during training. Default is MeanSquaredError.
        metric : callable, optional
            Metric used to evaluate the network. Default is None.
        **kwargs
            Additional keyword arguments passed to the optimizer.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer(learning_rate=learning_rate, **kwargs)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric
        self.layers = []
        self.history = {}

    def add(self, layer):
        """Add a layer to the neural network.

        Sets the input shape based on the output shape of the previous layer,
        initializes the layer with the optimizer if required, and appends it
        to the network. The input shape of the first layer must be provided
        beforehand.

        Parameters
        ----------
        layer : Layer
            Layer to add to the neural network.

        Returns
        -------
        NeuralNetwork
            The current neural network instance.
        """
        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())

        if hasattr(layer, "initialize"):
            layer.initialize(self.optimizer)

        self.layers.append(layer)
        return self

    def _mini_batches(self, X, y):
        """Shuffle the dataset and divide it into mini-batches.

        Parameters
        ----------
        X : numpy.ndarray
            Input features.
        y : numpy.ndarray
            Target values.

        Yields
        ------
        tuple of numpy.ndarray
            Mini-batch containing the input features and corresponding targets.
        """
        indices = np.random.permutation(X.shape[0])

        for start in range(0, X.shape[0], self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            yield X[batch_indices], y[batch_indices]

    def _forward(self, X, training):
        """Perform forward propagation through all network layers.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        training : bool
            Whether the network is in training mode.

        Returns
        -------
        numpy.ndarray
            Output produced by the final layer.
        """
        output = X

        for layer in self.layers:
            output = layer.forward_propagation(output, training=training)

        return output

    def fit(self, dataset):
        """Train the neural network using mini-batch gradient descent.

        For each epoch, the dataset is shuffled and divided into mini-batches.
        Each mini-batch undergoes forward propagation, loss derivative
        calculation and backward propagation. The weights are updated after
        every mini-batch.

        At the end of each epoch, the loss and metric are computed and stored
        in the history dictionary.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the neural network.

        Returns
        -------
        NeuralNetwork
            The trained neural network instance.
        """
        X, y = dataset.X, dataset.y

        if np.ndim(y) == 1:
            y = y.reshape(-1, 1)

        self.history = {}

        for epoch in range(1, self.epochs + 1):
            epoch_outputs = []
            epoch_targets = []

            # Shuffle the data and split it into mini-batches.
            for X_batch, y_batch in self._mini_batches(X, y):

                # Forward propagation through all layers.
                output = self._forward(X_batch, training=True)

                # Calculate the initial error using the loss derivative.
                error = self.loss.derivative(y_batch, output)

                # Backward propagation through all layers in reverse order.
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error)

                epoch_outputs.append(output)
                epoch_targets.append(y_batch)

            # Combine predictions and targets from all mini-batches.
            predictions = np.concatenate(epoch_outputs, axis=0)
            targets = np.concatenate(epoch_targets, axis=0)

            # Calculate the loss and metric for the epoch.
            epoch_loss = self.loss.loss(targets, predictions)
            epoch_metric = self.metric(targets, predictions) if self.metric else None

            # Store the epoch results.
            self.history[epoch] = {
                "loss": epoch_loss,
                "metric": epoch_metric
            }

            if self.verbose:
                print(
                    f"Epoch {epoch}/{self.epochs} - "
                    f"loss: {epoch_loss:.4f} - metric: {epoch_metric}"
                )

        return self

    def predict(self, dataset):
        """Make predictions on new data using forward propagation.

        Forward propagation is performed with training set to False.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the input features.

        Returns
        -------
        numpy.ndarray
            Predictions produced by the neural network.
        """
        return self._forward(dataset.X, training=False)

    def score(self, dataset):
        """Evaluate the neural network on a given dataset.

        Uses predict() to obtain the predictions and computes the selected
        evaluation metric.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to evaluate the neural network.

        Returns
        -------
        float
            Metric value calculated from the true and predicted values.
        """
        y = dataset.y.reshape(-1, 1) if np.ndim(dataset.y) == 1 else dataset.y
        predictions = self.predict(dataset)

        return self.metric(y, predictions)
