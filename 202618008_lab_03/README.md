# DS605: Fundamentals of Machine Learning
## Lab Assignment 3: Scikit-learn Data Preprocessing and Model Performance Evaluation

- **Student Name:** [Nandini Sanjaybhai Pipaliya]
- **Student ID:** [202618008]
- **Dataset:** [Kaggle Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

---

## 1. Project Overview
The objective of this assignment is to build and evaluate robust Scikit-learn preprocessing pipelines to predict hotel booking cancellations (`is_canceled`)[cite: 1]. We compare two distinct preprocessing pipelines across two classification algorithms (Logistic Regression and Decision Tree Classifier) using a stratified train-test split[cite: 1].

---

## 2. Preprocessing & Data Cleaning Decisions

1. **Target Selection & Leakage Prevention:**
   - Target variable: `is_canceled` (0 = Not Canceled, 1 = Canceled)[cite: 1].
   - Dropped `reservation_status` and `reservation_status_date` because they directly disclose the final booking outcome, causing target leakage[cite: 1].

2. **Handling Missing Values:**
   - The `company` column was dropped due to extreme missingness (>94% null values)[cite: 1].
   - Numerical columns: Missing values are imputed using `KNNImputer(n_neighbors=5)`[cite: 1].
   - Categorical columns: Missing values are imputed using `SimpleImputer(strategy='most_frequent')`[cite: 1].

3. **Outlier Treatment:**
   - Removed logically impossible records (rows where `adults`, `children`, and `babies` are all 0).
   - Filtered out negative or unrealistic `adr` values (`adr >= 0` and `adr < 5000`)[cite: 1].
   - Applied IQR filtering ($3 \times \text{IQR}$) to trim extreme anomalies in `adr` and `lead_time`[cite: 1].

4. **Pipeline Architectures:**
   - **Pipeline A:** `KNNImputer(5)` + `StandardScaler` for numerical features; `SimpleImputer` + `OneHotEncoder(handle_unknown='ignore')` for categorical features[cite: 1].
   - **Pipeline B:** `KNNImputer(5)` + `MinMaxScaler` for numerical features; `SimpleImputer` + `OneHotEncoder(handle_unknown='ignore')` for categorical features[cite: 1].
   - `ColumnTransformer` and `Pipeline` were used to ensure all transformations were fitted strictly on the training set[cite: 1].

---

## 3. Model Performance Comparison Table

| Model & Pipeline | Train Accuracy | Test Accuracy | Overfitting Gap (Train - Test) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (StandardScaler)** | 0.8187 | 0.8152 | 0.0035 | 0.8038 | 0.6622 | 0.7262 |
| **Logistic Regression (MinMaxScaler)** | 0.8148 | 0.8110 | 0.0038 | 0.8003 | 0.6519 | 0.7186 |
| **Decision Tree (StandardScaler)** | 0.9960 | 0.8604 | 0.1357 | 0.8053 | 0.8212 | 0.8132 |
| **Decision Tree (MinMaxScaler)** | 0.9960 | 0.8605 | 0.1356 | 0.8054 | 0.8214 | 0.8133 |

---

## 4. Final Observations

1. **Best Overall Result:** The **Decision Tree Classifier** achieved the highest overall performance, with a test accuracy of **86.05%** and an F1-score of **0.8133**. It captures non-linear feature interactions much more effectively than Logistic Regression.
2. **Impact of Scaling on Logistic Regression:** **StandardScaler** yielded slightly better results than **MinMaxScaler** for Logistic Regression (test accuracy of 81.52% vs. 81.10%, and F1-score of 0.7262 vs. 0.7186). Standardizing zero-centered distributions with unit variance provided better optimization stability for gradient convergence.
3. **Impact of Scaling on Decision Trees:** Scaling had **no operational effect** on the Decision Tree (performance across StandardScaler and MinMaxScaler differed by only 0.0001). Decision trees split nodes based on rank order and feature thresholds, making them scale-invariant.
4. **Overfitting Analysis:** The unpruned **Decision Tree exhibits significant overfitting**, achieving 99.60% accuracy on training data but dropping to 86.05% on testing data (a 13.56% gap). In contrast, **Logistic Regression generalizes consistently**, exhibiting a negligible train-test gap of ~0.35%.

---

