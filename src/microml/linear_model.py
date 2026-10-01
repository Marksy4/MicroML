import numpy as np


class LinearRegression:
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None


    def fit(self, x, y):
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float).ravel()
        ones = np.ones((x.shape[0], 1))
        x_b =np.hstack((ones, x))
        weights, residuals, rank, s = np.linalg.lstsq(x_b, y, rcond=None)

        self.intercept_ = weights[0]
        self.coef_ = weights[1:]
        return self


    def predict(self, x):
        X = np.asarray(x, dtype=float)
        return X @ self.coef_ + self.intercept_