## Lab Assignment 4: End-to-End Machine Learning Project: Airbnb Price Prediction 

- **Student Name:** Nandini Sanjaybhai Pipaliya
- **Student ID:** 202618008
- **Dataset:** [Kaggle New York City Airbnb Open Data (AB_NYC_2019.csv)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

## Project Overview
This project cleans and analyzes the AB_NYC_2019 dataset, engineers
location- and activity-based features, trains and compares multiple
regression models, and deploys the best model as an interactive
Streamlit application.

## Repository Structure
- `notebooks/airbnb_price_prediction.ipynb` — full analysis: data cleaning,
  feature engineering, model training, tuning, and evaluation
- `app/app.py` — Streamlit application for interactive price estimation
- `models/` — saved preprocessing + model pipeline (joblib)
- `outputs/` — model comparison results, plots, and app screenshots
- `requirements.txt` — Python dependencies


## Key Findings
- Price is heavily right-skewed; modeled as `log1p(price)`
- Strongest price drivers: `room_type`, `neighbourhood_group` (borough),
  and engineered `distance_to_center_km`

## Model Comparison

| Model | Train R² | Train RMSE | Test R² | Test RMSE | Test MAE (log) |
|---|---|---|---|---|---|
| Linear Regression | 0.5728 | 0.4424 | 0.5484 | 0.4576 | 0.3351 |
| Ridge | 0.5725 | 0.4425 | 0.5488 | 0.4574 | 0.3351 |
| Random Forest | 0.8204 | 0.2868 | 0.6101 | 0.4252 | 0.3075 |
| **Gradient Boosting** | 0.6309 | 0.4112 | **0.5954** | 0.4331 | 0.3163 |

*RMSE and MAE are reported in log-price space (model trained on `log1p(price)`); see Final Model Evaluation for dollar-scale metrics.*

### Observations
- **Linear Regression and Ridge** perform almost identically (Ridge's regularization has minimal effect here), and both show a small train/test gap — indicating **underfitting** rather than overfitting. Neither model captures the non-linear interactions in the data.
- **Random Forest** achieves the highest raw test R² (0.610), but with the largest train/test gap (0.820 → 0.610, a drop of ~0.21) — a clear sign of **overfitting**, consistent with the learning curve analysis.
- **Gradient Boosting** has the best balance: a smaller train/test gap (0.631 → 0.595, ~0.04) than Random Forest, meaning it generalizes more reliably despite a marginally lower raw test R². This makes it the preferred candidate for hyperparameter tuning and final selection.

## How to Run

### Notebook
```bash
pip install -r requirements.txt
jupyter notebook notebooks/airbnb_price_prediction.ipynb
```

### Streamlit App
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Deployed Application
[Live app link](https://your-app-name.streamlit.app) *(add once deployed — Step 5 below)*

## Limitations
- No image, review-text, or seasonal data — pricing psychology and
  photo quality aren't captured
- `neighbourhood` (221 categories) can be noisy for linear models
- Model reflects 2019 NYC pricing patterns; not adjusted for inflation
  or post-pandemic market shifts

