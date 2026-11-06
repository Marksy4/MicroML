import numpy as np


class LinearRegression:
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None


    def fit(self, x, y):
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float).ravel()
        ones = np.ones((x.shape[0], 1))
        x_b = np.hstack((ones, x))
        weights, residuals, rank, s = np.linalg.lstsq(x_b, y, rcond=None)

        self.intercept_ = weights[0]
        self.coef_ = weights[1:]
        return self


    def predict(self, x):
        X = np.asarray(x, dtype=float)
        return X @ self.coef_ + self.intercept_


class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.intercept_ = None
        self.coef_ = None

    def fit(self, x, y):
        X = np.array(x, dtype=float)
        y = np.array(y, dtype=float).ravel()

        n_features = X.shape[1]

        x_mean = X.mean(axis=0)
        y_mean = y.mean()
        x_centered = X - x_mean
        y_centered = y - y_mean

        I = np.identity(n_features)
        A = x_centered.T @ x_centered + self.alpha * I

        self.coef_ = np.linalg.solve(A, x_centered.T @ y_centered)
        self.intercept_ = y_mean - x_mean @ self.coef_
        return self

    def predict(self, x):
        X = np.asarray(x, dtype=float)
        return X @ self.coef_ + self.intercept_