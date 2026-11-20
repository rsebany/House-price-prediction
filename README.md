# House Prices Prediction — Practical Work 1

Kaggle: House Prices — Advanced Regression Techniques (Ames, Iowa)

**Team:** SEBANY Romualdo & Ait ameur Abdenafaa
**Model:** tuned ElasticNet on log(SalePrice), submitted prediction refit on all 1,458 labelled rows
**Results (hold-out):** MAE $13,442 (7.41%) · RMSE $18,874 (10.41%) · R² 0.936 · MAPE 8.0% · RMSLE 0.1105
**Honest estimate (CV):** RMSE ≈ $21,482 (≈11.9% of mean price)

## Contents

- `house-prediction_SEBANY_Romualdo_AitAmeur_Abdenafaa.ipynb` — full pipeline: EDA → missing values → feature engineering → preprocessing → model comparison → tuning → hold-out evaluation → submission. Runs on Kaggle (competition data auto-detected) or locally (uses `./data/`).
- `submission.csv` — Kaggle submission, tuned ElasticNet refit on all labelled rows.
- `results.json` — machine-readable results: all models (CV + hold-out, $ and %), tuning outcome, metrics.
- `training_history.csv`, `training_history.json` — train/validation learning curve and the full cross-validation/tuning history.
- `holdout_predictions.csv` — 292 hold-out predictions (row_id, actual, predicted, residual).
- `house_price_model.joblib`, `app_meta.json` — trained model + app metadata for the Streamlit app.
- `app.py` — Streamlit showcase: Overview / Model comparison / Final model evaluation / Price predictor / Submission (with submission upload & comparison).
- `data/` — train.csv, test.csv, sample_submission.csv, data_description.txt (for local re-runs; on Kaggle the notebook finds the competition data automatically).
- `screenshots/` — Streamlit app screenshots: overview, model comparison, final evaluation, price predictor.
- `docs/` — the PW1 assignment brief.

## How to run

Notebook: Kaggle → Add data (competition attached) → Run all. Local → `pip install pandas scikit-learn==1.6.1 matplotlib seaborn xgboost` → open Jupyter, run all (it finds `./data/train.csv`).

Streamlit app: `pip install streamlit matplotlib joblib` → `streamlit run app.py` → http://localhost:8501

Note: keep scikit-learn at **1.6.1** — `house_price_model.joblib` was saved with it.

# House-price-prediction
