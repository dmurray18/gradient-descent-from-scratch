"""
test_gradient_descent.py -- unit tests for the core implementation.

Run from the repo root: pytest tests/
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gradient_descent import (
    predict, mse, standardize_train, compute_gradients, gradient_descent,
    numerical_gradient_check,
)


def test_predict():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = np.array([2.0, 3.0])
    b = 1.0
    expected = np.array([9.0, 19.0])
    assert np.allclose(predict(X, w, b), expected)


def test_mse():
    predictions = np.array([1.0, 2.0, 3.0])
    actual = np.array([1.0, 3.0, 5.0])
    assert np.isclose(mse(predictions, actual), 5 / 3)


def test_standardize_train_produces_zero_mean_unit_std():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_scaled, mean, std = standardize_train(X)
    assert np.allclose(np.mean(X_scaled, axis=0), 0)
    assert np.allclose(np.std(X_scaled, axis=0), 1)


def test_compute_gradients_shape():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 3))
    y = rng.normal(size=10)
    w = np.zeros(3)
    b = 0.0
    db, dw = compute_gradients(X, y, w, b)
    assert np.isscalar(db)
    assert dw.shape == (3,)


def test_gradient_descent_reduces_cost():
    rng = np.random.default_rng(18)
    X = rng.normal(size=(50, 3))
    true_w = np.array([2.0, -1.0, 0.5])
    y = X @ true_w + rng.normal(scale=0.1, size=50)

    w = np.zeros(3)
    b = 0.0
    _, _, cost_history = gradient_descent(X, y, w, b, alpha=0.1, iterations=500)

    assert len(cost_history) == 500
    assert cost_history[-1] < cost_history[0]


def test_numerical_gradient_check_passes_within_relative_tolerance():
    rng = np.random.default_rng(18)
    X = rng.normal(size=(100, 5))
    y = rng.normal(loc=200000, scale=50000, size=100)  # mimics real target scale
    w = np.zeros(5)
    b = 0.0

    analytical_dw, numerical_dw, analytical_db, numerical_db, weight_error, bias_error = (
        numerical_gradient_check(X, y, w, b)
    )

    weight_rel_error = weight_error / (np.max(np.abs(analytical_dw)) + 1e-10)
    bias_rel_error = bias_error / (abs(analytical_db) + 1e-10)

    assert weight_rel_error < 1e-3
    assert bias_rel_error < 1e-3
