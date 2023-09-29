import numpy as np
from scipy.sparse import lil_matrix


def iqr_threshold_method(scores, margin):
    q1 = np.percentile(scores, 25, interpolation="midpoint")
    q3 = np.percentile(scores, 75, interpolation="midpoint")
    iqr = q3 - q1
    lower_range = q1 - (1.5 * iqr)
    upper_range = q3 + (1.5 * iqr)
    lower_range = lower_range - margin
    upper_range = upper_range + margin
    return lower_range, upper_range


class KNN:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first n_neighbors neighbors
        k_indices = np.argsort(distances)[: self.n_neighbors]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

    def kneighbors_graph(self):
        n_samples = self.X_train.shape[0]
        graph = lil_matrix((n_samples, n_samples))
        dists = np.linalg.norm(self.X_train[:, None] - self.X_train, axis=2).T

        for i in range(n_samples):
            indices = np.argsort(dists[i])
            for j in range(1, self.n_neighbors + 1):
                graph[i, indices[j]] = 1

        return graph


def get_odin_outliers(data, n_neighbors, margin=0):
    knn = KNN(n_neighbors)
    knn.fit(data, None)
    graph = knn.kneighbors_graph()
    indegree = graph.sum(axis=0)

    scores = (indegree.max() - indegree) / indegree.max()
    scores = np.array(scores)[0]

    lower_range, upper_range = iqr_threshold_method(scores, margin)

    anomaly_points = []
    for i in range(len(scores)):
        if scores[i] < lower_range or scores[i] > upper_range:
            anomaly_points.append(i)

    return anomaly_points


def get_knn_outliers(x_train, y_train, n_neighbors):
    train_len = x_train.shape[0]
    knn_outliers = []
    for i, image in enumerate(x_train):
        other_indicies = np.arange(train_len) != i
        x_train_without_i = x_train[other_indicies]
        y_train_without_i = y_train[other_indicies]
        model = KNN(n_neighbors)
        model.fit(x_train_without_i, y_train_without_i)
        prediction = model.predict(image.reshape(1, -1))[0]
        if prediction != y_train[i]:
            knn_outliers.append(i)
    return knn_outliers
