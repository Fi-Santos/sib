from si.base.transformer import Transformer
from si.data.dataset import Dataset
import numpy as np

class Variancethreshold(Transformer):
	"""
    Transformer that removes features with a variance below a given threshold.

	Parameters
    ----------
    threshold : float, default=0.05
        Minimum variance required for a feature to be kept.

    Attributes
    ----------
    threshold : float
        Variance threshold used to select features.
    variance : numpy.ndarray or None
        Variance calculated for each feature during fitting.

    """
	def __init__(self, threshold : float = 0.05):
		"""Initializes the Variancethreshold transformer."""
		super().__init__()
		self.threshold = threshold
		self.variance = None

	def _fit(self, dataset: Dataset) -> Dataset:
		"""Calculates the variance of each feature in the dataset."""
		self.variance = np.var(dataset.X, axis = 0)
		return self

	def _transform(self, dataset: Dataset) -> Dataset:
		"""Removes features whose variance is below the specified threshold."""
		features_to_keep = self.variance > self.threshold
		X = dataset.X[:, features_to_keep]
		features = np.array(dataset.features)[features_to_keep] if dataset.features is not None else None
		return Dataset(X, dataset.y, features, dataset.label)



if __name__ == '__main__':
    # Testar o construtor e o funcionamento fora da classe
    X = np.array([[0, 2], [0, 3], [0, 4]])
    y = np.array([0, 1, 0])
    features = ["f1", "f2"]
    
    dataset = Dataset(X, y, features=features)
    
    # Chamada do construtor
    vt = Variancethreshold(threshold=0.1)
    print("Threshold definido no construtor:", vt.threshold)
    
    # Testar fit e transform
    vt.fit(dataset)
    print("Variâncias calculadas:", vt.variance)
    
    dataset_transformed = vt.transform(dataset)
    print("X após o transform:", dataset_transformed.X)
	


