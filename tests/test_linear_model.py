import numpy as np
from sklearn.linear_model import Ridge as SkRidge
from sklearn.linear_model import LinearRegression as SkLinearRegression
from microml.linear_model import LinearRegression, Ridge



def test_linear_regression_matches_sklearn():
    X = np.array([
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3],
        [5, 6],
    ])
    y = np.array([5, 4, 11, 10, 17])

    mine = LinearRegression().fit(X, y)
    sk = SkLinearRegression().fit(X, y)

    assert np.allclose(mine.coef_, sk.coef_)
    assert np.allclose(mine.intercept_, sk.intercept_, atol=1e-8)


def test_ridge_matches_sklearn_across_alphas():
    X = np.array([
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3],
        [5, 6],
    ])
    y = np.array([5, 4, 11, 10, 17])

    for alpha in [0.0, 1.0, 10.0]:
        mine = Ridge(alpha=alpha).fit(X, y)
        sk = SkRidge(alpha=alpha).fit(X, y)
        assert np.allclose(mine.coef_, sk.coef_)
        assert np.allclose(mine.intercept_, sk.intercept_)