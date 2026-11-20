# House Price Prediction

End-to-end machine learning project for the Kaggle "House Prices: Advanced Regression Techniques" competition (Ames, Iowa). It covers data cleaning, exploratory analysis, feature engineering, model comparison, hyperparameter tuning, hold-out evaluation, submission generation, and an interactive Streamlit app.

## Highlights

- Single notebook pipeline from raw data to Kaggle submission.
- Eight regression models compared with cross-validation and a hold-out set.
- Tuned ElasticNet on log(SalePrice) selected as the final model.
- Stacking ensemble evaluated as an additional experiment.
- Streamlit app for model diagnostics, price prediction, and submission validation.
- Saved metrics, tuning history, hold-out predictions, and trained model for reproducibility.

## Dataset

Source: Kaggle House Prices: Advanced Regression Techniques.

- Ames housing data with 1,460 labelled rows and 79 explanatory variables.
- Target variable: SalePrice.
- The notebook detects Kaggle competition data automatically when running on Kaggle, and uses `./data/` locally.

## Results

Final model performance on the hold-out set:

| Metric | Value |
| --- | --- |
| MAE | 13,442 (7.41%) |
| RMSE | 18,874 (10.41%) |
| R2 | 0.936 |
| RMSLE | 0.1105 |

Cross-validation RMSE of the tuned model is about 21,482, roughly 11.9% of the mean sale price. Full model scores, tuning parameters, and ensemble results are in `results.json`.

## Repository layout

```text
main.ipynb                     Full analysis and modelling pipeline
app.py                         Streamlit application
app_meta.json                  App configuration and model metadata
house_price_model.joblib       Trained model used by the app
results.json                   Metrics, tuning outcome, and model comparison
training_history.csv           Training and validation curve data
training_history.json          Full cross-validation and tuning history
holdout_predictions.csv        Hold-out predictions with actual values and residuals
submission.csv                 Prediction file ready for Kaggle upload
data/                          Train, test, sample submission, and data description
screenshots/                   Streamlit app screenshots
```

## Setup

Python 3.10 or newer is recommended.

```bash
pip install pandas numpy scikit-learn==1.6.1 matplotlib seaborn xgboost joblib streamlit
```

The model artifact was serialized with scikit-learn 1.6.1, so keep that version when loading it.

## Usage

### Notebook

On Kaggle, attach the House Prices competition data and run all cells. Locally, keep `data/` in the project root and run all cells in `main.ipynb`. The notebook writes the submission and all result artifacts next to itself.

### Streamlit app

```bash
streamlit run app.py
```

The app has pages for overview, model comparison, final model evaluation, price prediction, and submission upload and checking.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, joblib, Streamlit.

## Acknowledgment

Dataset provided by Kaggle through the House Prices: Advanced Regression Techniques competition.
