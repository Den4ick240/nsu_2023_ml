import numpy as np


class PolynomRegressor:
    def __init__(self, M, alpha):
        self.M = M
        self.alpha = alpha

    def fit(self, x, y):
        X = np.vander(x, self.M + 1, increasing=True)
        A = X.T @ X + self.alpha * np.identity(self.M + 1)
        B = X.T @ y
        self.theta = np.linalg.inv(A) @ B

    def predict(self, x):
        X = np.vander(x, self.M + 1, increasing=True)
        return X @ self.theta
