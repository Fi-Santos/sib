import numpy as np

def sigmoid_function(X):
    """
    Calculate the sigmoid function for the given input.

    The sigmoid function transforms input values into probabilities
    between 0 and 1.

    Parameters:
        X: Input value or array of values.

    Returns:
        The sigmoid probability for each input value.
    """
    probability = 1 / (1 + np.exp(-X))

    return probability





if __name__ == '__main__':
    X = np.array([-2, 0, 2])
    print(sigmoid_function(X))