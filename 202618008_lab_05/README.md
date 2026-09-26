## Lab Assignment 5: Machine Learning with Scikit-learn and From Scratch

- **Student Name:** Nandini Sanjaybhai Pipaliya
- **Student ID:** 202618008
- **Dataset:** [UCI Productivity Prediction of Garment Employees](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees)

Comparing a Scikit-learn implementation against a from-scratch NumPy/Pandas implementation of Linear Regression and Logistic Regression, using the UCI Garment Employee Productivity dataset.

## Project Structure

- **Part A — Scikit-learn Implementation**: preprocessing via `ColumnTransformer`/`Pipeline`, `LinearRegression`, `LogisticRegression`.
- **Part B — From-Scratch Implementation**: manual missing-value imputation, one-hot encoding, standardization, closed-form Linear Regression (Normal Equation), and Logistic Regression via sigmoid + gradient descent — NumPy/Pandas only, no ML libraries.
- **Part C — Comparison & Optimization**: side-by-side metric/timing tables, plus manual-model improvements via convergence tracking, learning-rate tuning, L2 regularization, and correlation-based feature selection.

## Targets

- **Regression**: `actual_productivity` (continuous)
- **Classification**: `MeetsTarget` = 1 if `actual_productivity >= targeted_productivity`, else 0 (`actual_productivity` excluded from classification inputs to avoid leakage)


## Key Results

| Metric | Scikit-learn | Manual v1 | Manual v2 (optimized) |
|---|---|---|---|
| MAE | 0.1085 | 0.1085 | 0.1083 |
| RMSE | 0.1486 | 0.1486 | 0.1485 |
| R² | 0.1682 | 0.1682 | 0.1695 |
| Accuracy | 0.775 | 0.775 | 0.775 |
| Precision | 0.781 | 0.781 | 0.781 |
| Recall | 0.966 | 0.966 | 0.966 |
| F1 | 0.864 | 0.864 | 0.864 |

Regularization (L2, λ=1.0) slightly improved regression metrics and shifted classification probabilities (by up to ~0.05) without flipping any final predictions on the test set.

## Notable Implementation Notes

- The Normal Equation initially failed with `LinAlgError: Singular matrix` due to the **dummy variable trap** (one-hot encoding every category + a bias term creates a linearly dependent feature matrix). Fixed by dropping one reference category per categorical variable; also independently resolved by adding L2 regularization (`λI` guarantees invertibility).
- Manual metric functions (`mae`, `accuracy`, etc.) were named identically to Scikit-learn result variables in early drafts, silently overwriting them — resolved by renaming Scikit-learn's stored scores (e.g., `mae_sklearn`).
- Manual implementation was faster on regression (closed-form solve, no pipeline overhead) but slower on classification training (plain gradient descent vs. Scikit-learn's `lbfgs` solver).
