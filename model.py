import numpy as np
from typing import List, Union


class my_KNN:
    """
    K-Nearest Neighbors (KNN) classifier.

    :param k: The number of neighbors to use for classification. Must be an odd number.
    :param indexing: The distance metric to use ('euclidean', 'manhattan', or 'hamming').

    :type k: int
    :type indexing: str
    """
    
    def __init__ (self, k: int = 5, indexing: str = "euclidean"):
        """
        Initialize the KNN classifier.

        :param k: The number of neighbors. Must be an odd number.
        :type k: int
        :param indexing: The distance metric to use ('euclidean', 'manhattan', or 'hamming').
        :type indexing: str

        :raises ValueError: If k is an even number.
        """
        if k % 2 == 0:
            raise ValueError("K must be an odd number")
        self.k = k
        self.indexing = indexing
        self.X_train = None
        self.y_train = None
    
    def fit (self, X: Union[np.ndarray, List[List[float]]], y: Union[np.ndarray, List[int]]):
        """
        Fit the KNN model with training data.

        :param X: Training data.
        :type X: Union[np.ndarray, List[List[float]]]
        :param y: Labels for the training data.
        :type y: Union[np.ndarray, List[int]]
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict (self, X: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Predict the labels for the input data.

        :param X: Input data for prediction.
        :type X: Union[np.ndarray, List[List[float]]]

        :return: Predicted labels.
        :rtype: np.ndarray
        """
        X = np.array(X)
        predictions = [self.__predict(x) for x in X]
        return np.array(predictions)
    
    @staticmethod
    def euclidean_distance (a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the Euclidean distance between two points.

        :param a: First point.
        :type a: np.ndarray
        :param b: Second point.
        :type b: np.ndarray

        :return: Euclidean distance.
        :rtype: float
        """
        return float(np.linalg.norm(a - b))
    
    @staticmethod
    def manhattan_distance (a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the Manhattan distance between two points.

        :param a: First point.
        :type a: np.ndarray
        :param b: Second point.
        :type b: np.ndarray

        :return: Manhattan distance.
        :rtype: float
        """
        return np.sum(np.abs(a - b))
    
    @staticmethod
    def hamming_distance (a: np.ndarray, b: np.ndarray) -> int:
        """
        Compute the Hamming distance between two sequences.

        :param a: First sequence.
        :type a: np.ndarray
        :param b: Second sequence.
        :type b: np.ndarray

        :return: Hamming distance.
        :rtype: int

        :raises ValueError: If the sequences have different lengths.
        """
        if len(a) != len(b):
            raise ValueError("Sequences must be of the same length")
        return np.sum(a != b)
    
    def __predict (self, x: np.ndarray) -> int:
        """
        Predict the label for a single data point.

        :param x: Data point to classify.
        :type x: np.ndarray

        :return: Predicted label.
        :rtype: int

        :raises ValueError: If an unknown indexing method is specified.
        """
        if self.indexing == "euclidean":
            distances = np.linalg.norm(self.X_train - x, axis=1)
        elif self.indexing == "manhattan":
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        elif self.indexing == "hamming":
            if x.shape[0] != self.X_train.shape[1]:
                raise ValueError("Feature length of input and training data must be the same for Hamming distance")
            distances = np.sum(self.X_train != x, axis=1)
        else:
            raise ValueError("Unknown indexing method")
        
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return int(most_common)
