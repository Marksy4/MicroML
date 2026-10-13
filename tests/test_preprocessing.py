import numpy as np
from microml.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler as SkStandardScaler


def test_scaler():
    X_train = np.array([
        [2.0, 100.0],
        [4.0, 200.0],
        [6.0, 300.0]
    ])

    scaler = StandardScaler()
    scaler.fit(X_train)
    expected_mu = np.array([4.0, 200.0])
    expected_sigma = np.array([1.63299316, 81.64965809])

    assert np.allclose(scaler.mean_, expected_mu)
    assert np.allclose(scaler.scale_, expected_sigma)


def test_transform_output():
    X_train = np.array([
        [2.0, 100.0],
        [4.0, 200.0],
        [6.0, 300.0],
    ])

    scaler = StandardScaler()
    result = scaler.fit_transform(X_train)

    assert np.allclose(result.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(result.std(axis=0), 1, atol=1e-8)


def test_fit_transformer():
    X_train = np.array([
        [2.0, 0],
        [2.0, 0],
        [2.0, 0]
    ])

    scaler = StandardScaler()
    expected_mu = np.array([2, 0])
    expected_sigma = np.array([1, 1])
    result = scaler.fit_transform(X_train)

    assert np.all(np.isfinite(result))
    assert np.allclose(result, 0)
    assert np.allclose(scaler.mean_, expected_mu)
    assert np.allclose(scaler.scale_, expected_sigma)


def test_fit_transformer_sklearn():
    X_train = np.array([
        [2.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.0]
    ])

    sklearn_scaler = SkStandardScaler()
    sklearn_result = sklearn_scaler.fit_transform(X_train)

    micro_scaler = StandardScaler()
    micro_result = micro_scaler.fit_transform(X_train)

    assert np.allclose(micro_scaler.mean_, sklearn_scaler.mean_)
    assert np.allclose(micro_scaler.scale_, sklearn_scaler.scale_)
    assert np.allclose(micro_result, sklearn_result)