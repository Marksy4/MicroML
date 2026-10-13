import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None


    def fit(self, x):
        x = np.asarray(x, dtype=float)

        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0, ddof=0)

        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)

        return self


    def transform(self, x):
        x = np.asarray(x, dtype=float)
        return (x - self.mean_) / self.scale_


    def fit_transform(self, x):
        return self.fit(x).transform(x)