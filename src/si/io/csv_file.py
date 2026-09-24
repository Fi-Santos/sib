import pandas as pd
def read_csv(filename, separador, features, label):

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


def write_csv(filename, dataset,separador, label):
	return dataset.to_csv(filename, sep=separador, index = False)
