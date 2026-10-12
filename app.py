from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

STATIC_NUM = [
    "distance", "log_distance", "weight", "weight_missing",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "bearing_ew", "bearing_ns",
]
STATIC_CAT = ["pickup", "delivery", "equipment"]
T0 = pd.Timestamp("2025-01-01")


class FreightFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["dow"] = df["date"].dt.dayofweek
        for c in ["distance", "weight", "market_index",
                  "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["weight_missing"] = df["weight"].isna().astype(int)
        df["weight"] = df["weight"].abs().clip(upper=47500)
        day_mean = df.groupby("date")["market_index"].transform("mean")
        df["market_index"] = df["market_index"].fillna(day_mean)
        df["log_distance"] = np.log(df["distance"])
        df["bearing_ew"] = df["delivery_lon"] - df["pickup_lon"]
        df["bearing_ns"] = df["delivery_lat"] - df["pickup_lat"]
        return df


class TwoStageRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, block=14, carry_days=60, iters=2, seed=42):
        self.block = block
        self.carry_days = carry_days
        self.iters = iters
        self.seed = seed

    def _static(self, d):
        return d[STATIC_NUM + STATIC_CAT]

    def fit(self, X, y):
        fe = FreightFeatureEngineer()
        self.fe_ = fe
        d = fe.transform(X)
        y = np.log(np.asarray(y, float))
        self.prep_ = ColumnTransformer([
            ("num", "passthrough", STATIC_NUM),
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1), STATIC_CAT),
        ])
        Z = self.prep_.fit_transform(self._static(d))
        cat_mask = [False] * len(STATIC_NUM) + [True] * len(STATIC_CAT)
        days = (d["date"] - T0).dt.days.to_numpy()
        blk = days // self.block
        B = (blk[:, None] == np.unique(blk)[None, :]).astype(float)
        m = d["market_index"].to_numpy()[:, None]
        A = np.c_[m, B]
        lvl = np.zeros_like(y)
        for _ in range(self.iters):
            self.gbm_ = HistGradientBoostingRegressor(
                categorical_features=cat_mask,
                max_iter=300, learning_rate=0.05,
                min_samples_leaf=40, l2_regularization=1.0,
                early_stopping=False, random_state=self.seed,
            ).fit(Z, y - lvl)
            r = y - self.gbm_.predict(Z)
            coef = np.linalg.lstsq(A, r, rcond=None)[0]
            lvl = A @ coef
        self.slope_ = float(coef[0])
        resid_level = r - self.slope_ * m[:, 0]
        self.offset_ = float(
            resid_level[days >= days.max() - self.carry_days].mean()
        )
        return self

    def predict(self, X):
        d = self.fe_.transform(X)
        Z = self.prep_.transform(self._static(d))
        return np.exp(
            self.gbm_.predict(Z)
            + self.slope_ * d["market_index"].to_numpy()
            + self.offset_
        )


st.set_page_config(page_title="Freight Rate Predictor", page_icon="🚚", layout="wide")

BASE = Path(__file__).parent

MODEL_PATH = BASE / "models" / "final_model.joblib"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE / "final_model.joblib"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE / "freight_rate_model.joblib"

st.sidebar.title("🚚 Freight Rate Predictor")
st.sidebar.caption("Built by **Fransisca Iloh**")

page = st.sidebar.radio("Navigate", ["Predict", "About"])

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        st.sidebar.success("Model loaded")
    except Exception as e:
        model = None
        st.sidebar.error(f"Load failed: {e}")
else:
    model = None
    st.sidebar.error("Model file not found")
    st.sidebar.write("Looked for:", str(MODEL_PATH))


if page == "Predict":
    st.title("Freight Rate Prediction")
    st.caption("Enter load details to estimate the posted rate.")

    c1, c2 = st.columns(2)
    with c1:
        pickup = st.text_input("Pickup city", "Richmond")
        delivery = st.text_input("Delivery city", "Baltimore")
        p_lat = st.number_input("Pickup latitude", value=38.09122, format="%.5f")
        p_lon = st.number_input("Pickup longitude", value=-76.78906, format="%.5f")
        d_lat = st.number_input("Delivery latitude", value=38.16908, format="%.5f")
        d_lon = st.number_input("Delivery longitude", value=-72.74564, format="%.5f")
    with c2:
        distance = st.number_input("Distance (miles)", min_value=1.0, value=274.3)
        equipment = st.selectbox(
            "Equipment",
            ["Dry Van", "Reefer", "Flatbed", "Tanker", "Step Deck", "Power Only"],
        )
        weight = st.number_input("Weight (lbs)", min_value=0.0, value=30658.0)
        market_index = st.number_input("Market index", value=0.95684, format="%.5f")
        load_date = st.date_input("Load date", value=date(2025, 5, 15))

    if st.button("Predict rate", type="primary"):
        if model is None:
            st.error("Model not loaded.")
        else:
            row = pd.DataFrame([{
                "pickup": pickup, "delivery": delivery,
                "pickup_lat": p_lat, "pickup_lon": p_lon,
                "delivery_lat": d_lat, "delivery_lon": d_lon,
                "distance": distance, "equipment": equipment,
                "weight": weight, "market_index": market_index,
                "date": pd.to_datetime(load_date),
            }])
            try:
                pred = float(model.predict(row)[0])
                a, b, c = st.columns(3)
                a.metric("Predicted rate", f"${pred:,.2f}")
                b.metric("Rate per mile", f"${pred/distance:,.2f}")
                c.metric("Distance", f"{distance:,.0f} mi")
            except Exception as e:
                st.exception(e)

else:
    st.title("About")
    st.markdown("""
    ### Hi, I'm Fransisca Iloh

    I'm a **Data Scientist** passionate about solving real-world problems
    with data. This app deploys a two-stage freight rate prediction model
    combining gradient boosting with a time-aware market index adjustment.

    **Validation:** MAE $57.09 · RMSE $139.11 · R² 0.989 · MAPE 2.36%

    **Not used:** `quote_signal` (excluded by design to avoid leakage).

    - 🐙 [GitHub](https://github.com/IlohFransisca)
    """)       
