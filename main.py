"""
main.py -- trains the from-scratch model on the California Housing dataset,
evaluates it against a baseline and against scikit-learn's closed-form
solution, and saves diagnostic plots.

Run from the repo root: python src/main.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gradient_descent import (
    predict, mse, standardize_train, standardize_test, train_test_split,
    gradient_descent, k_fold_cross_validation, numerical_gradient_check,
    baseline_prediction, residuals, evaluate_model, run_learning_rate_experiment,
)

DATA_PATH = "data/california_housing_train.csv"
FEATURES = ["total_rooms", "total_bedrooms", "population", "households", "median_income"]
TARGET = "median_house_value"
RANDOM_SEED = 18


def main():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES].to_numpy(dtype=float)
    y = data[TARGET].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=RANDOM_SEED)
    X_train, train_mean, train_std = standardize_train(X_train)
    X_test = standardize_test(X_test, train_mean, train_std)

    w = np.zeros(X_train.shape[1])
    b = 0.0
    w, b, cost_history = gradient_descent(X_train, y_train, w, b, alpha=0.01, iterations=2000)

    train_results = evaluate_model(predict(X_train, w, b), y_train)
    test_results = evaluate_model(predict(X_test, w, b), y_test)

    print("FINAL MODEL")
    print("=" * 50)
    print("Weights:", w)
    print("Bias:", b)
    print("\nTraining Results:", {k: round(v, 4) for k, v in train_results.items()})
    print("Testing Results:", {k: round(v, 4) for k, v in test_results.items()})

    baseline_preds = baseline_prediction(y_train, y_test)
    baseline_results = evaluate_model(baseline_preds, y_test)
    mse_improvement = 1 - test_results["MSE"] / baseline_results["MSE"]

    print("\nBaseline Test MSE:", round(baseline_results["MSE"], 4))
    print(f"MSE improvement over baseline: {mse_improvement * 100:.2f}%")

    cv_scores = k_fold_cross_validation(X, y, k=5, alpha=0.01, iterations=2000, seed=RANDOM_SEED)
    print(f"\n5-Fold CV Mean MSE: {np.mean(cv_scores):.4f} (Std: {np.std(cv_scores):.4f})")

    small_X, small_y = X_train[:100], y_train[:100]
    small_w, small_b = np.zeros(small_X.shape[1]), 0.0
    *_, weight_error, bias_error = numerical_gradient_check(small_X, small_y, small_w, small_b)
    print(f"\nGradient check -- max weight error: {weight_error:.6f}, bias error: {bias_error:.6f}")

    lr_results = run_learning_rate_experiment(X_train, y_train, learning_rates=[0.001, 0.005, 0.01, 0.05])
    print("\nLearning rate experiment:")
    for alpha, result in lr_results.items():
        print(f"  alpha={alpha}: final MSE = {result['final_mse']:.4f}")

    model_residuals = residuals(predict(X_test, w, b), y_test)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(cost_history)
    axes[0].set_title("Gradient Descent Learning Curve")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("MSE")

    axes[1].scatter(y_test, predict(X_test, w, b), alpha=0.3)
    axes[1].set_title("Actual vs Predicted")
    axes[1].set_xlabel("Actual House Value"); axes[1].set_ylabel("Predicted House Value")

    axes[2].scatter(predict(X_test, w, b), model_residuals, alpha=0.3)
    axes[2].axhline(0, linestyle="--", color="black")
    axes[2].set_title("Residuals vs Predictions")
    axes[2].set_xlabel("Predicted Value"); axes[2].set_ylabel("Residual")

    plt.tight_layout()
    plt.savefig("diagnostics.png", dpi=150)
    print("\nSaved diagnostics.png")


if __name__ == "__main__":
    main()
