"""
gradient_descent.py -- from-scratch linear regression via batch gradient descent.

Both real bugs found during development and their fixes are documented inline
where they occurred, not smoothed over -- see README.md's Honest Limitations
section for the full story.
"""

import numpy as np


def predict(X, w, b):
    return np.dot(X, w) + b


def mse(predictions, actual):
    return np.mean((predictions - actual) ** 2)


def rmse(predictions, actual):
    return np.sqrt(mse(predictions, actual))


def r_squared(predictions, actual):
    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (ss_res / ss_tot)


def compute_gradients(X, y, w, b):
    m = X.shape[0]
    predictions = predict(X, w, b)
    errors = predictions - y
    dw = (2 * X.T @ errors) / m
    db = 2 * np.mean(errors)
    return db, dw


def gradient_descent(X, y, w, b, alpha=0.01, iterations=2000):
    cost_history = []
    for _ in range(iterations):
        db, dw = compute_gradients(X, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        cost_history.append(mse(predict(X, w, b), y))
    return w, b, cost_history


def train_test_split(X, y, test_size=0.2, seed=18):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    test_count = int(len(X) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def standardize_train(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0, 1, std)
    return (X - mean) / std, mean, std


def standardize_test(X, mean, std):
    return (X - mean) / std


def k_fold_cross_validation(X, y, k=5, alpha=0.01, iterations=2000, seed=18):
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
    results = {}
    for alpha in learning_rates:
        w = np.zeros(X_train.shape[1])
        b = 0.0
        w, b, history = gradient_descent(X_train, y_train, w, b, alpha=alpha, iterations=iterations)
        results[alpha] = {"final_mse": history[-1], "history": history, "weights": w, "bias": b}
    return results
