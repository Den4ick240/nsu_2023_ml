import numpy as np
from utils.get_mnist import get_mnist
from utils.test_model import test_model
from models.knn import KNN, get_knn_outliers, get_odin_outliers

n_neighbors = 3
mnist = get_mnist()
acc = test_model(KNN(n_neighbors), *mnist)
print(f"Accuracy: {acc}")

acc = test_model(KNN(n_neighbors), *get_mnist(noise_level=0.2))
print(f"Noisy accuracy: {acc}")

x_train, y_train, _, _ = mnist
knn_outliers = get_knn_outliers(x_train, y_train, n_neighbors)
print("KNN outliers: ")
print(knn_outliers)

odin_outliers = get_odin_outliers(x_train, n_neighbors)
print("ODIN outliers: ")
print(odin_outliers)

intersection = np.intersect1d(knn_outliers, odin_outliers)
print("Intersections")
print(intersection)
