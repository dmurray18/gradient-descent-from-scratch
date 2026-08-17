# Gradient Descent From Scratch

A from-scratch NumPy implementation of linear regression and batch gradient descent, validated against scikit-learn's closed-form solution — built to prove mechanism-level understanding, not library-calling.

## Problem

Understanding *how* gradient descent actually works — not just calling `.fit()` — is one of the clearest signals separating candidates who've used ML libraries from candidates who understand what those libraries are doing underneath. This project implements linear regression and batch gradient descent entirely from first principles in NumPy, then validates the result against scikit-learn's `LinearRegression` on real data (California Housing, 17,000 observations) to confirm the implementation converges to the correct answer, not just *an* answer.

## Approach

1. Implement `predict`, `mse`, and `compute_gradients` (the analytical gradient of MSE) from scratch
2. Implement batch gradient descent, standardization (fit on train, applied to test — never re-fit on test data), and a train/test split
3. Validate the analytical gradient against a finite-difference numerical approximation (`numerical_gradient_check`) — this is what caught a real bug during development (see Honest Limitations)
4. Train on 5 features (`total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`) against `median_house_value`
5. Evaluate against a baseline (predict the training mean) and against scikit-learn's closed-form `LinearRegression` on the identical train/test split
6. Run 5-fold cross-validation and a learning-rate sensitivity sweep

## Results

| Metric | This implementation | scikit-learn |
|---|---|---|
| Test MSE | 6,210,157,893.76 | 6,221,384,132.34 |
| Test R² | 0.5491 | 0.5483 |

**0.18% MSE difference, R² difference of 0.0008** — effectively identical, which is the real evidence this implementation is mathematically correct rather than merely "runs without crashing."

- 5-fold CV Mean MSE: 6,268,809,616.31 (Std: 312,949,577.52)
- 54.92% MSE improvement over the baseline (predicting the training mean)
- Gradient check: passes within relative tolerance (see Honest Limitations for why relative, not absolute)

## Honest Limitations

**A real bug was found and fixed during development, not before it.** The first working version of `compute_gradients` was missing a factor of 2 — the true gradient of `MSE = mean((Xw+b-y)²)` is `(2/m)·Xᵀ·error`, and the original code computed `(1/m)·Xᵀ·error`. This didn't prevent the model from training (a missing constant factor in every gradient step is mathematically equivalent to training at half the stated learning rate — it slows convergence, it doesn't break it), which is exactly why it's easy to miss: the model still produced plausible-looking results. It was caught by `numerical_gradient_check`, which is specifically designed to catch this class of bug — every gradient component was off from the finite-difference approximation by a consistent, exact factor of 2 before the fix.

**A second, smaller issue was found while fixing the first.** After fixing the gradient, the check still failed against a fixed absolute tolerance (`1e-4`). Investigated via an epsilon-sweep: the residual error *grew* as epsilon shrank (0.003% relative error at epsilon=1e-4, growing to 0.31% at epsilon=1e-6) — the textbook signature of floating-point cancellation, not a further algorithmic bug, since a real remaining bug would show the opposite pattern. Fixed by switching the test to relative tolerance, which is standard, correct practice for gradient checking specifically because a fixed absolute tolerance doesn't scale across problems with different target magnitudes (house prices here are ~$10⁵, so gradients are naturally large).

**The dataset has a known censoring artifact.** `median_house_value` is capped at $500,001 in the original data collection — visible as a vertical cluster in the Actual vs. Predicted plot. The model can't distinguish a $500,001 house from a $2,000,000 house because the data itself can't, which caps achievable R² regardless of model quality.

**Hyperparameters were not tuned via grid search.** `alpha=0.01` and `iterations=2000` were chosen via the learning-rate sweep (`results/learning_rate_experiment`) rather than a systematic search — sufficient to demonstrate correct convergence, not necessarily optimal.

**Only linear relationships are modeled.** No polynomial features or interaction terms were engineered; R² of ~0.55 reflects a genuinely linear model on features with real nonlinear relationships to price (e.g., location data was excluded entirely).

## How to Run

```bash
git clone <repo-url>
cd gradient-descent-from-scratch
pip install -r requirements.txt
python src/main.py
```

Run the test suite:

```bash
pytest tests/
```

## Repo Structure

```
src/
  gradient_descent.py   - core implementation (predict, gradients, training, CV, gradient check)
  main.py                - loads data, trains, evaluates, saves diagnostics.png
tests/
  test_gradient_descent.py - pytest unit tests, including the gradient check that caught the real bug
data/
  california_housing_train.csv - the dataset (17,000 rows)
```

No `notebooks/` directory — this project didn't need an exploratory notebook; the analysis lives directly in `src/main.py`.

## Tech Stack

Python 3, NumPy (core implementation), Pandas (data loading), Matplotlib (diagnostics), scikit-learn (validation comparison only — never used inside the actual implementation), pytest (testing).
