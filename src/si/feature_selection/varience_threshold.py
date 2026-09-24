from si.base.transformer import Transformer
from si.data.dataset import Dataset
import numpy as np

class VarianceThreshould(Transformer):
	def __init__(self, threshould):
        	self.threshould = threshould

	def _fit(self, dataset: Dataset) -> Dataset:
		self.variance = np.var(dataset.X, axis = 0)
		return self

	def _transform(self, dataset: Dataset) -> Dataset:
		features_to_keep = self.variance > self.threshould
		X = dataset.X[:, features_to_keep]
		features = np.array(dataset.features)[features_to_keep]
		return Dataset(X, dataset.y, features, dataset.label)
