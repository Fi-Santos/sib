import numpy as np

def read_data_file(filename, sep, label):
	df = np.genfromtxt(filename, delimiter = sep, skip_header = 1)
	header = np.genfromtxt(filename, delimiter=sep, max_rows=1, dtype=str)
	df = np.vstack([header, df])
	if label == "y":
		df = df[:,:-1]
	else:
		None
	return df

def write_data_file(filename, dataset, sep):
	return np.savetxt(filename, dataset, delimiter = sep, fmt="%s")
