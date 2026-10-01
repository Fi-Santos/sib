import pandas as pd

def read_csv(filename, separador, features, label):
    """
    Reads a CSV file and returns the dataset according to the specified feature and label options.
    """
    if features == "y":
        df = pd.read_csv(filename, sep=separador, header=0)
    else:
        df = pd.read_csv(filename, sep=separador, header=None)

        df = df.iloc[:, 1:]

        if label == "y":
            dataset = df.iloc[:, :-1]
        else:
            dataset = df

        return dataset


def write_csv(filename, dataset, separador, label):
    """
    Writes a dataset to a CSV file using the specified separator.
    """
    return dataset.to_csv(filename, sep=separador, index=False)