import numpy as np

def euclidean_distance (a, b):
	a = np.array(a)
	b = np.array(b)
	return np.linalg.norm(a - b)


def manhattan_distance (a, b):
	a = np.array(a)
	b = np.array(b)
	return np.sum(np.abs(a - b))


def hamming_distance (a, b):
	if len(a) != len(b):
		raise ValueError("Sequences must be of the same length")
	return sum(el1 != el2 for el1, el2 in zip(a, b))


class KNN:
	def __init__ (self, k=5, indexing="euclidean"):
		if k % 2 == 0:
			raise ValueError("K must be an odd number")
		self.k = k
		self.indexing = indexing
		self.X_train = None
		self.y_train = None
	
	def fit (self, X, y):
		self.X_train = np.array(X)
		self.y_train = np.array(y)
	
	def predict (self, X):
		X = np.array(X)
		predictions = [self.__predict(x) for x in X]
		return np.array(predictions)
	
	def __predict (self, x):
		match self.indexing:
			case "euclidean":
				distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
			case "manhattan":
				distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
			case "hamming":
				distances = [hamming_distance(x, x_train) for x_train in self.X_train]
			case _:
				raise ValueError("Unknown indexing method")
		
		
		k_indices = np.argsort(distances)[:self.k]
		k_nearest_labels = [self.y_train[i] for i in k_indices]
		most_common = np.bincount(k_nearest_labels).argmax()
		return most_common

