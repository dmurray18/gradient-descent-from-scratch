"""
gradient_descent.py -- from-scratch linear regression via batch gradient descent.

Both real bugs found during development and their fixes are documented inline
where they occurred, not smoothed over -- see README.md's Honest Limitations
section for the full story.
"""

import numpy as np


def predict(X, w, b):
    """Linear prediction: X @ w + b.

    Parameters
    ----------
    X : ndarray, shape (m, n)
    w : ndarray, shape (n,)
    b : float

    Returns
    -------
    ndarray, shape (m,)
    """
    return np.dot(X, w) + b


def mse(predictions, actual):
    """Mean squared error."""
    return np.mean((predictions - actual) ** 2)


def rmse(predictions, actual):
    """Root mean squared error."""
    return np.sqrt(mse(predictions, actual))


def r_squared(predictions, actual):
    """Coefficient of determination."""
    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (ss_res / ss_tot)


def compute_gradients(X, y, w, b):
    """Analytical gradient of MSE with respect to w and b.

    BUG FIX: an earlier version of this function computed (1/m)*X.T@error,
    missing the factor of 2 that comes from differentiating the squared
    error term in MSE = mean((Xw+b-y)^2). The true gradient is
    (2/m)*X.T@error. This was caught by numerical_gradient_check() below --
    every component was off by exactly 2x before the fix. See README.md.
    """
    m = X.shape[0]
    predictions = predict(X, w, b)
    errors = predictions - y
    dw = (2 * X.T @ errors) / m
    db = 2 * np.mean(errors)
    return db, dw


def gradient_descent(X, y, w, b, alpha=0.01, iterations=2000):
    """Batch gradient descent. Returns final (w, b, cost_history)."""
    cost_history = []
    for _ in range(iterations):
        db, dw = compute_gradients(X, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        cost_history.append(mse(predict(X, w, b), y))
    return w, b, cost_history


def train_test_split(X, y, test_size=0.2, seed=42):
    """Simple random train/test split."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    test_count = int(len(X) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def standardize_train(X):
    """Z-score standardization, fit on training data. Returns (X_scaled, mean, std)."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0, 1, std)
    return (X - mean) / std, mean, std


def standardize_test(X, mean, std):
    """Apply training-set mean/std to test data -- never re-fit on test data."""
    return (X - mean) / std


def k_fold_cross_validation(X, y, k=5, alpha=0.01, iterations=2000, seed=42):
    """K-fold CV. Standardization is re-fit on each fold's training split only,
    to avoid leaking validation-fold statistics into training."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    folds = np.array_split(indices, k)
    scores = []

    for i in range(k):
        validation_indices = folds[i]
        training_indices = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, X_valid = X[training_indices], X[validation_indices]
        y_train, y_valid = y[training_indices], y[validation_indices]

        X_train, mean, std = standardize_train(X_train)
        X_valid = standardize_test(X_valid, mean, std)

        w = np.zeros(X_train.shape[1])
        b = 0.0
        w, b, _ = gradient_descent(X_train, y_train, w, b, alpha, iterations)

        scores.append(mse(predict(X_valid, w, b), y_valid))

    return np.array(scores)


def numerical_gradient_check(X, y, w, b, epsilon=1e-5):
    """Finite-difference gradient check against the analytical gradient.

    This is what caught the factor-of-2 bug during development -- every
    component of the analytical gradient was off from the numerical
    approximation by a consistent ratio of exactly 2x before the fix.
    """
    analytical_db, analytical_dw = compute_gradients(X, y, w, b)

    numerical_dw = np.zeros(len(w))
    for j in range(len(w)):
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[j] += epsilon
        w_minus[j] -= epsilon
        cost_plus = mse(predict(X, w_plus, b), y)
        cost_minus = mse(predict(X, w_minus, b), y)
        numerical_dw[j] = (cost_plus - cost_minus) / (2 * epsilon)

    cost_plus_b = mse(predict(X, w, b + epsilon), y)
    cost_minus_b = mse(predict(X, w, b - epsilon), y)
    numerical_db = (cost_plus_b - cost_minus_b) / (2 * epsilon)

    weight_error = np.max(np.abs(analytical_dw - numerical_dw))
    bias_error = abs(analytical_db - numerical_db)

    return analytical_dw, numerical_dw, analytical_db, numerical_db, weight_error, bias_error


def baseline_prediction(y_train, y_test):
    """Baseline: always predict the training mean."""
    return np.full(len(y_test), np.mean(y_train))


def residuals(predictions, actual):
    return actual - predictions


def evaluate_model(predictions, actual):
    return {
        "MSE": mse(predictions, actual),
        "RMSE": rmse(predictions, actual),
        "R2": r_squared(predictions, actual),
    }


def run_learning_rate_experiment(X_train, y_train, learning_rates, iterations=2000):
    """Train with several learning rates, return final MSE and history for each."""
    results = {}
    for alpha in learning_rates:
        w = np.zeros(X_train.shape[1])
        b = 0.0
        w, b, history = gradient_descent(X_train, y_train, w, b, alpha=alpha, iterations=iterations)
        results[alpha] = {"final_mse": history[-1], "history": history, "weights": w, "bias": b}
    return results
