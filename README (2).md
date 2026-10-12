# Freight Rate Predictor (Streamlit app)

A small web app around the freight rate model. Pick a lane, equipment, weight, distance and date and it returns the predicted posted rate. It can also sweep a date range (like the December chart) and score a CSV of loads.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `freight_model.py` | The model classes (same as the notebook) and the helper functions the app uses |
| `train_model.py` | Fits the model on all cleaned labeled data and saves `model/bundle.joblib` |
| `model/bundle.joblib` | The trained model plus lookup tables (city coordinates, lane distances, daily market index) |
| `requirements.txt` | Dependencies |

## Run it

```bash
python -m pip install -r requirements.txt

# put train_test.csv and validation.csv in data/, then (about 2 minutes):
python train_model.py

streamlit run app.py
```

Re-running `train_model.py` in your own environment is recommended even though a bundle is included, because a saved scikit-learn model is only guaranteed to load under the same library version it was saved with.

### In Google Colab

Streamlit needs a tunnel in Colab:

```python
!pip -q install streamlit
!npm -q install localtunnel
!streamlit run app.py &>/content/logs.txt &
!npx localtunnel --port 8501      # open the URL it prints; the password is the output of: !curl ipv4.icanhazip.com
```

### Deploy

Push the folder to GitHub and create an app on Streamlit Community Cloud pointing at `app.py`.
Commit `data/train_test.csv` and `data/validation.csv` as well as `model/bundle.joblib`. A saved scikit-learn model only loads safely under the version that saved it, and Streamlit Cloud installs the newest versions. On start the app therefore checks the saved model; if it was saved under a different version it retrains from the CSVs (a minute or so, once per server start) and says so on the About tab. Without the CSVs it shows an error asking for them. Keep the repository private if the data should not be public.

## What the tabs do

- **Single prediction:** one load. Distance defaults to the typical distance for the lane and can be edited. The market index comes from the date automatically and can be overridden.
- **Date sweep:** the same load across a date range, with a chart, summary numbers and a CSV download. December 1 to 31 reproduces the assessment chart (use 360 miles; the lane default is 362).
- **Batch (CSV):** upload `pickup, delivery, distance, equipment, weight, date` (and optionally `market_index`). Download a template from the tab. Rows with missing values are skipped; unknown cities give a clear error.
- **About the model:** how it works and its limits.

## Model in short

Gradient boosting on log(rate) for load, lane and equipment effects, plus a linear market-index term and a carried-forward level for the slow drift in rates. `quote_signal` is not used because its relationship to the rate changes from month to month. Forward-in-time validation (three folds): MAE $57.09, RMSE $139.11, R² 0.989, MAPE 2.36%.

## Things to know

- The market index for a date comes from the November/December data files; dates outside the data use the latest 14-day average.
- Predictions through the app use the daily average market index, while the notebook's validation predictions use each load's own value. The difference is small (about 0.2% on average).
- Eight cities (Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego) appear in the validation data but not in training. They can be selected; the model estimates them from coordinates and the app says so.
- The training data ends in October, so holiday effects in late December are not learned.
