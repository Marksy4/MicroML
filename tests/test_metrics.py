import numpy as np
from microml.metrics import mean_squared_error, r2_score


def test_mse_zero_when_predictions_match_exactly():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    assert mean_squared_error(y_true, y_pred) == 0


def test_mse_known_value():
    # ошибки: (1,2)->1, (2,4)->4 → среднее (1+4)/2 = 2.5
    y_true = np.array([1, 2])
    y_pred = np.array([2, 4])
    assert np.isclose(mean_squared_error(y_true, y_pred), 2.5)


def test_r2_perfect_prediction_is_one():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])
    assert np.isclose(r2_score(y_true, y_pred), 1.0)


def test_r2_mean_prediction_is_zero():
    # если предсказывать средним по y_true, R² должен быть ровно 0
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.full_like(y_true, fill_value=np.mean(y_true), dtype=float)
    assert np.isclose(r2_score(y_true, y_pred), 0.0)