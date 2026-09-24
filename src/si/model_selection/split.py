import numpy as np
def train_test_split(dataset, test_size, random_state):
	dataset_size = len(dataset)
	test_dataset = int(dataset_size * test_size)
	np.random.seed(random_state)
	permutações = np.random.permutation(dataset_size)

	test = []
	test_permu = permutações[:test_dataset]
	test = dataset.iloc[test_permu]

	train = []
	train_permu = permutações[test_dataset:]
	train = dataset.iloc[train_permu]
	return (train, test)


def stratified_train_test_split(dataset, test_size, random_state):

	y = dataset.iloc[:, -1]
	labels = y.unique()

	train_indices = []
	test_indices = []

	np.random.seed(random_state)

	for label in labels:

		class_indices = []

		for i in range(len(dataset)):
			if dataset.iloc[i, -1] == label:
				class_indices.append(i)

		class_size = len(class_indices)
		test_class_size = int(class_size * test_size)

		permutações = np.random.permutation(class_size)

		test_permu = permutações[:test_class_size]
		train_permu = permutações[test_class_size:]

		test_class_indices = [class_indices[i] for i in test_permu]
		train_class_indices = [class_indices[i] for i in train_permu]

		test_indices.extend(test_class_indices)
		train_indices.extend(train_class_indices)

	train = dataset.iloc[train_indices]
	test = dataset.iloc[test_indices]

	return train, test
