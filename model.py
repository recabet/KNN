import numpy as np
from typing import List, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
    
    def predict (self, X: Union[np.ndarray, List[float]]) -> int:
        """
        Predict the label for a single data point.

        :param X: Data point to classify.
        :type X: Union[np.ndarray, List[float]]

        :return: Predicted label.
        :rtype: int

        :raises ValueError: If an unknown indexing method is specified or distance computation is invalid.
        """
        X = np.array(X)
        
        if self.indexing == "euclidean":
            distances = np.linalg.norm(self.X_train - X, axis=1)
        elif self.indexing == "manhattan":
            distances = np.sum(np.abs(self.X_train - X), axis=1)
        elif self.indexing == "hamming":
            if X.shape[0] != self.X_train.shape[1]:
                raise ValueError("Feature length of input and training data must be the same for Hamming distance")
            distances = np.sum(self.X_train != X, axis=1)
        else:
            raise ValueError("Unknown indexing method")
        
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        
        return int(most_common)
    
    def find_optimal_k (self, X: Union[np.ndarray, List[List[float]]], y: Union[np.ndarray, List[int]],
                        k_values: List[int] = None, test_size: float = 0.2) -> int:
        """
        Find the optimal value of k using a range of k values.

        :param X: Training data.
        :type X: Union[np.ndarray, List[List[float]]]
        :param y: Labels for the training data.
        :type y: Union[np.ndarray, List[int]]
        :param k_values: List of k values to try. If None, a default range from 1 to 20 is used.
        :type k_values: List[int]
        :param test_size: Proportion of data to use for testing.
        :type test_size: float

        :return: The optimal value of k based on accuracy.
        :rtype: int
        """
        if k_values is None:
            k_values = list(range(1, 21, 2))
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
        
        accuracies = []
        
        for k in k_values:
            self.k = k
            self.fit(X_train, y_train)
            predictions = [self.predict(x) for x in X_val]
            accuracy = accuracy_score(y_val, predictions)
            accuracies.append((k, accuracy))
        
        best_k = max(accuracies, key=lambda item: item[1])[0]
        
        print(f"Optimal k found: {best_k} with accuracy: {max(accuracies, key=lambda item: item[1])[1]:.2f}")
        self.k = best_k
        
        return best_k
