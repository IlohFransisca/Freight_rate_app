# Freight Rate Predictor

A Streamlit app that predicts freight `posted_rate` using a two-stage
gradient boosting model.

## Features
- Single Prediction - enter load details, get an instant rate estimate.
- Batch Upload - upload a CSV, download predictions.
- December Chart - view the December 2025 predicted rate curve.
- Auto-fill coordinates - look up pickup/delivery coordinates by city name.

## Model
- Stage 1: HistGradientBoostingRegressor on static lane / equipment / weight features
- Stage 2: market-index slope + 14-day block level adjustment

| Metric | Value |
|---|---|
| MAE  | 57.09 |
| RMSE | 139.11 |
| R2   | 0.989 |
| MAPE | 2.36% |

quote_signal is excluded by design.

## Repo layout

    .
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    ├── .streamlit/
    │   └── config.toml
    ├── models/
    │   └── final_model.joblib
    └── data/
        ├── validation.csv
        └── december_chart_inputs_filled.csv

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

## Deploy on Streamlit Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and click New app.
3. Pick the repo, branch main, main file app.py.
4. Click Deploy.

## Batch CSV format

Required columns:

    pickup, delivery, pickup_lat, pickup_lon,
    delivery_lat, delivery_lon, distance,
    equipment, weight, date, market_index
