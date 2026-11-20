# House-price-prediction

Kaggle House Prices: Advanced Regression Techniques (Ames, Iowa).

Team: SEBANY Romualdo and Ait ameur Abdenafaa

Model: tuned ElasticNet on log(SalePrice), refit on all 1458 labelled rows for the submission.

Validation results: MAE 13442 (7.41%), RMSE 18874 (10.41%), R2 0.936, RMSLE 0.1105.

Contents:
- main.ipynb: full pipeline (EDA, preprocessing, feature engineering, model comparison, tuning, evaluation, submission)
- app.py: Streamlit app (overview, model comparison, evaluation, price predictor, submission check)
- app_meta.json, house_price_model.joblib: files used by the app
- results.json, training_history.csv, training_history.json, holdout_predictions.csv: results and diagnostics
- submission.csv: Kaggle submission
- data/: train.csv, test.csv, sample_submission.csv, data_description.txt
- screenshots/: app screenshots

Run the notebook: attach the Kaggle competition data and run all cells. Locally, run all cells with data/ in place.

Install: pip install pandas scikit-learn==1.6.1 matplotlib seaborn xgboost streamlit joblib

Run the app: streamlit run app.py

Note: the model was saved with scikit-learn 1.6.1.
