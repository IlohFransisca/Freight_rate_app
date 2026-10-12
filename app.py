from pathlib import Path
from datetime import date
import os

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

# =====================================================================
# Custom classes (must match training)
# =====================================================================
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


def build_model():
    return TwoStageRegressor()


# =====================================================================
# Streamlit config
# =====================================================================
st.set_page_config(
    page_title="Freight Rate Predictor — Fransisca Iloh",
    page_icon="🚚",
    layout="wide",
)

BASE_DIR = Path(__file__).parent


def _find_file(*candidates):
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


DATA_DIR = BASE_DIR / "data" if (BASE_DIR / "data").exists() else BASE_DIR

MODEL_PATH = _find_file(
    BASE_DIR / "models" / "final_model.joblib",
    BASE_DIR / "final_model.joblib",
    BASE_DIR / "models" / "freight_rate_model.joblib",
    BASE_DIR / "freight_rate_model.joblib",
)

VALIDATION_PATH = _find_file(
    DATA_DIR / "validation.csv",
    BASE_DIR / "validation.csv",
)

DECEMBER_PATH = _find_file(
    DATA_DIR / "december_chart_inputs_filled.csv",
    BASE_DIR / "december_chart_inputs_filled.csv",
)

EQUIPMENT_TYPES = ["Dry Van", "Reefer", "Flatbed", "Tanker", "Step Deck", "Power Only"]


# =====================================================================
# Model + data loading
# =====================================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.session_state["_model_load_error"] = repr(e)
        return None


@st.cache_data(show_spinner=False)
def load_city_lookup():
    if not VALIDATION_PATH.exists():
        return {"pickup": {}, "delivery": {}}
    df = pd.read_csv(VALIDATION_PATH)
    p = (df.dropna(subset=["pickup_lat", "pickup_lon"])
           .drop_duplicates("pickup")
           .set_index("pickup")[["pickup_lat", "pickup_lon"]])
    d = (df.dropna(subset=["delivery_lat", "delivery_lon"])
           .drop_duplicates("delivery")
           .set_index("delivery")[["delivery_lat", "delivery_lon"]])
    return {"pickup": p.to_dict("index"), "delivery": d.to_dict("index")}


model = load_model()
city_lookup = load_city_lookup()


def make_input_row(pickup, delivery, pickup_lat, pickup_lon,
                   delivery_lat, delivery_lon, distance, equipment,
                   weight, market_index, load_date):
    return pd.DataFrame([{
        "pickup": pickup, "delivery": delivery,
        "pickup_lat": pickup_lat, "pickup_lon": pickup_lon,
        "delivery_lat": delivery_lat, "delivery_lon": delivery_lon,
        "distance": distance, "equipment": equipment,
        "weight": weight, "market_index": market_index,
        "date": pd.to_datetime(load_date),
    }])


# =====================================================================
# Sidebar
# =====================================================================
st.sidebar.title("🚚 Freight Rate Predictor")
st.sidebar.caption("Built by **Fransisca Iloh** · Data Scientist")

page = st.sidebar.radio(
    "Navigate",
    ["Single Prediction", "Batch Upload", "December Chart", "About Me", "Feedback"],
)

if model is None:
    st.sidebar.error("Model not found. Check the debug panel below.")
    with st.sidebar.expander("🔍 Debug info", expanded=True):
        st.write("**BASE_DIR:**", str(BASE_DIR))
        st.write("**BASE_DIR exists:**", BASE_DIR.exists())
        try:
            files = sorted(os.listdir(BASE_DIR))
            st.write("**Files in repo root:**")
            st.code("\n".join(files) if files else "(empty)")
        except Exception as e:
            st.write("Could not list:", repr(e))
        st.write("**MODEL_PATH attempted:**", str(MODEL_PATH))
        st.write("**MODEL_PATH exists:**", MODEL_PATH.exists())
        err = st.session_state.get("_model_load_error")
        if err:
            st.write("**Load error:**")
            st.code(err)
else:
    st.sidebar.success("Model loaded ✅")
    st.sidebar.caption(
        f"Slope: **{getattr(model, 'slope_', float('nan')):.3f}**  \n"
        f"Offset: **{getattr(model, 'offset_', float('nan')):.4f}**"
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "📧 [Contact](mailto:your.email@example.com) · "
    "💼 [LinkedIn](https://www.linkedin.com/) · "
    "🐙 [GitHub](https://github.com/IlohFransisca)"
)


# =====================================================================
# Page: Single Prediction
# =====================================================================
if page == "Single Prediction":
    st.title("Single Load Rate Prediction")
    st.caption("Enter load details to estimate the posted freight rate.")

    if model is None:
        st.warning(
            "The model file wasn't loaded. Open the **Debug info** panel "
            "in the sidebar to see why."
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Route")
        pickup = st.text_input("Pickup city", "Richmond")
        delivery = st.text_input("Delivery city", "Baltimore")
        p_lat = st.number_input("Pickup latitude", value=38.09122, format="%.5f")
        p_lon = st.number_input("Pickup longitude", value=-76.78906, format="%.5f")
        d_lat = st.number_input("Delivery latitude", value=38.16908, format="%.5f")
        d_lon = st.number_input("Delivery longitude", value=-72.74564, format="%.5f")
        if st.button("Auto-fill coordinates from city names"):
            p = city_lookup.get("pickup", {}).get(pickup)
            d = city_lookup.get("delivery", {}).get(delivery)
            if p:
                p_lat, p_lon = p["pickup_lat"], p["pickup_lon"]
            if d:
                d_lat, d_lon = d["delivery_lat"], d["delivery_lon"]
            if p or d:
                st.success(f"Filled (pickup={bool(p)}, delivery={bool(d)})")
            else:
                st.warning("No lookup match. Enter coordinates manually.")

    with col2:
        st.subheader("Load details")
        distance = st.number_input("Distance (miles)", min_value=1.0, value=274.3, step=1.0)
        equipment = st.selectbox("Equipment", EQUIPMENT_TYPES)
        weight = st.number_input("Weight (lbs)", min_value=0.0, value=30658.0, step=100.0)
        market_index = st.number_input("Market index", value=0.95684, format="%.5f")
        load_date = st.date_input("Load date", value=date(2025, 5, 15))

    if st.button("Predict rate", type="primary"):
        if model is None:
            st.error("Model not loaded — see debug panel in the sidebar.")
        else:
            row = make_input_row(
                pickup, delivery, p_lat, p_lon, d_lat, d_lon,
                distance, equipment, weight, market_index, load_date,
            )
            try:
                pred = float(model.predict(row)[0])
                rpm = pred / distance
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted rate", f"${pred:,.2f}")
                c2.metric("Rate per mile", f"${rpm:,.2f}")
                c3.metric("Distance", f"{distance:,.0f} mi")

                with st.expander("Input row sent to model"):
                    st.dataframe(row, use_container_width=True)
            except Exception as e:
                st.exception(e)


# =====================================================================
# Page: Batch Upload
# =====================================================================
elif page == "Batch Upload":
    st.title("Batch Prediction")
    st.caption(
        "Upload a CSV with columns: pickup, delivery, pickup_lat, pickup_lon, "
        "delivery_lat, delivery_lon, distance, equipment, weight, date, market_index."
    )

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())

        required = ["pickup", "delivery", "pickup_lat", "pickup_lon",
                    "delivery_lat", "delivery_lon", "distance",
                    "equipment", "weight", "date", "market_index"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        elif model is None:
            st.error("Model not loaded — see debug panel in the sidebar.")
        else:
            with st.spinner("Predicting..."):
                preds = model.predict(df[required])
            out = df.copy()
            out["predicted_rate"] = np.round(preds, 2)

            st.success(f"Predicted {len(out):,} rows.")
            st.dataframe(out.head(50), use_container_width=True)

            st.download_button(
                "Download predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )


# =====================================================================
# Page: December Chart
# =====================================================================
elif page == "December Chart":
    st.title("December 2025 Predicted Load Rate")
    st.caption("Forecasted daily freight rate for December 2025.")

    if not DECEMBER_PATH.exists():
        st.warning(
            "Missing `december_chart_inputs_filled.csv`. "
            "Place it in `data/` or at the repo root."
        )
    else:
        dec = pd.read_csv(DECEMBER_PATH)
        dec["date"] = pd.to_datetime(dec["date"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dec["date"], dec["predicted_rate"], marker="o", linewidth=2)
        ax.set_ylabel("rate ($)")
        ax.set_xlabel("date")
        ax.set_title("Candidate: December 2025 Predicted Load Rate")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        st.pyplot(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("Min rate", f"${dec['predicted_rate'].min():,.2f}")
        c2.metric("Mean rate", f"${dec['predicted_rate'].mean():,.2f}")
        c3.metric("Max rate", f"${dec['predicted_rate'].max():,.2f}")

        with st.expander("View raw data"):
            st.dataframe(dec, use_container_width=True)


# =====================================================================
# Page: About Me
# =====================================================================
elif page == "About Me":
    st.title("About Me")
    st.markdown("""
    ### 👋 Hi, I'm Fransisca Iloh

    I'm a **Data Scientist** passionate about turning messy, real-world data
    into decisions people can actually act on. My work sits at the intersection
    of **machine learning**, **product thinking**, and **clear communication** —
    because a model only creates value when someone understands what it says.

    ---

    ### 🎯 What I focus on
    - **Applied machine learning** — regression, forecasting, and end-to-end pipelines
    - **Feature engineering** — building signal from raw, imperfect operational data
    - **Model evaluation** — designing validation that reflects the real deployment setting
    - **Deployment** — shipping models as usable tools, not just notebooks

    ---

    ### 🚚 About this project
    This Streamlit app is the deployment layer for a freight rate prediction model.

    **Problem:** Predict the `posted_rate` of a freight load from route, equipment,
    distance, weight, and market conditions.

    **Approach:** A two-stage regressor that combines
    - a `HistGradientBoostingRegressor` for static lane / equipment / weight effects, and
    - a time-aware market-index adjustment (14-day blocks + carried level) to model
      slow rate drift through the year.

    **Validation (time-based folds):**

    | Metric | Value |
    |---|---:|
    | MAE | $57.09 |
    | RMSE | $139.11 |
    | R² | 0.989 |
    | MAPE | 2.36 % |

    **Design choice:** `quote_signal` was deliberately excluded to avoid leakage
    and keep the model aligned with real, pre-quote prediction settings.

    ---

    ### 🛠️ Tools I work with
    Python · pandas · NumPy · scikit-learn · Streamlit · matplotlib · Jupyter · Git

    ---

    ### 📬 Get in touch
    - 📧 **Email:** your.email@example.com
    - 💼 **LinkedIn:** [linkedin.com/in/your-handle](https://www.linkedin.com/)
    - 🐙 **GitHub:** [github.com/IlohFransisca](https://github.com/IlohFransisca)

    *Open to data science roles, freelance work, and collaborations on
    real-world ML problems.*
    """)


# =====================================================================
# Page: Feedback
# =====================================================================
else:
    st.title("Feedback")
    st.markdown(
        "I'd love to hear what you think about this app. "
        "Whether it's a bug, a feature idea, or just a quick comment — "
        "every note helps me improve."
    )

    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Your name (optional)", placeholder="Jane Doe")
        email = st.text_input("Your email (optional)", placeholder="jane@example.com")
        rating = st.slider("How would you rate this app?", 1, 5, 5)
        category = st.selectbox(
            "What's your feedback about?",
            ["General comment", "Bug report", "Feature request", "Model accuracy", "UI / UX"],
        )
        message = st.text_area(
            "Your feedback",
            placeholder="Tell me what you think...",
            height=150,
        )
        submitted = st.form_submit_button("Submit feedback", type="primary")

        if submitted:
            if not message.strip():
                st.error("Please write a short message before submitting.")
            else:
                st.session_state.setdefault("feedback_log", []).append({
                    "name": name or "Anonymous",
                    "email": email,
                    "rating": rating,
                    "category": category,
                    "message": message,
                })
                st.success(
                    "Thank you for your feedback! 🙏 "
                    "I read every message and it genuinely helps."
                )
                st.balloons()

    st.markdown("---")
    st.caption(
        "Prefer email? Reach me at **your.email@example.com** — "
        "I usually reply within a day."
    )

    with st.expander("Session feedback log (visible only to you)"):
        log = st.session_state.get("feedback_log", [])
        if log:
            st.dataframe(pd.DataFrame(log), use_container_width=True)
        else:
            st.write("No feedback submitted in this session yet.")input("Weight (lbs)", min_value=0.0, value=30658.0, step=100.0)
        market_index = st.number_input("Market index", value=0.95684, format="%.5f")
        load_date = st.date_input("Load date", value=date(2025, 5, 15))

    if st.button("Predict rate", type="primary"):
        if model is None:
            st.error("Model not loaded — see debug panel in the sidebar.")
        else:
            row = make_input_row(
                pickup, delivery, p_lat, p_lon, d_lat, d_lon,
                distance, equipment, weight, market_index, load_date,
            )
            try:
                pred = float(model.predict(row)[0])
                rpm = pred / distance
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted rate", f"${pred:,.2f}")
                c2.metric("Rate per mile", f"${rpm:,.2f}")
                c3.metric("Distance", f"{distance:,.0f} mi")

                with st.expander("Input row sent to model"):
                    st.dataframe(row, use_container_width=True)
            except Exception as e:
                st.exception(e)


# =====================================================================
# Page: Batch Upload
# =====================================================================
elif page == "Batch Upload":
    st.title("Batch Prediction")
    st.caption(
        "Upload a CSV with columns: pickup, delivery, pickup_lat, pickup_lon, "
        "delivery_lat, delivery_lon, distance, equipment, weight, date, market_index."
    )

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())

        required = ["pickup", "delivery", "pickup_lat", "pickup_lon",
                    "delivery_lat", "delivery_lon", "distance",
                    "equipment", "weight", "date", "market_index"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        elif model is None:
            st.error("Model not loaded — see debug panel in the sidebar.")
        else:
            with st.spinner("Predicting..."):
                preds = model.predict(df[required])
            out = df.copy()
            out["predicted_rate"] = np.round(preds, 2)

            st.success(f"Predicted {len(out):,} rows.")
            st.dataframe(out.head(50), use_container_width=True)

            st.download_button(
                "Download predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )


# =====================================================================
# Page: December Chart
# =====================================================================
elif page == "December Chart":
    st.title("December 2025 Predicted Load Rate")
    st.caption("Forecasted daily freight rate for December 2025.")

    if not DECEMBER_PATH.exists():
        st.warning(
            "Missing `december_chart_inputs_filled.csv`. "
            "Place it in `data/` or at the repo root."
        )
    else:
        dec = pd.read_csv(DECEMBER_PATH)
        dec["date"] = pd.to_datetime(dec["date"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dec["date"], dec["predicted_rate"], marker="o", linewidth=2)
        ax.set_ylabel("rate ($)")
        ax.set_xlabel("date")
        ax.set_title("Candidate: December 2025 Predicted Load Rate")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        st.pyplot(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("Min rate", f"${dec['predicted_rate'].min():,.2f}")
        c2.metric("Mean rate", f"${dec['predicted_rate'].mean():,.2f}")
        c3.metric("Max rate", f"${dec['predicted_rate'].max():,.2f}")

        with st.expander("View raw data"):
            st.dataframe(dec, use_container_width=True)


# =====================================================================
# Page: About Me
# =====================================================================
elif page == "About Me":
    st.title("About Me")
    st.markdown("""
    ### 👋 Hi, I'm Fransisca Iloh

    I'm a **Data Scientist** passionate about turning messy, real-world data
    into decisions people can actually act on. My work sits at the intersection
    of **machine learning**, **product thinking**, and **clear communication** —
    because a model only creates value when someone understands what it says.

    ---

    ### 🎯 What I focus on
    - **Applied machine learning** — regression, forecasting, and end-to-end pipelines
    - **Feature engineering** — building signal from raw, imperfect operational data
    - **Model evaluation** — designing validation that reflects the real deployment setting
    - **Deployment** — shipping models as usable tools, not just notebooks

    ---

    ### 🚚 About this project
    This Streamlit app is the deployment layer for a freight rate prediction model.

    **Problem:** Predict the `posted_rate` of a freight load from route, equipment,
    distance, weight, and market conditions.

    **Approach:** A two-stage regressor that combines
    - a `HistGradientBoostingRegressor` for static lane / equipment / weight effects, and
    - a time-aware market-index adjustment (14-day blocks + carried level) to model
      slow rate drift through the year.

    **Validation (time-based folds):**

    | Metric | Value |
    |---|---:|
    | MAE | $57.09 |
    | RMSE | $139.11 |
    | R² | 0.989 |
    | MAPE | 2.36 % |

    **Design choice:** `quote_signal` was deliberately excluded to avoid leakage
    and keep the model aligned with real, pre-quote prediction settings.

    ---

    ### 🛠️ Tools I work with
    Python · pandas · NumPy · scikit-learn · Streamlit · matplotlib · Jupyter · Git

    ---

    ### 📬 Get in touch
    - 📧 **Email:** your.email@example.com
    - 💼 **LinkedIn:** [linkedin.com/in/your-handle](https://www.linkedin.com/)
    - 🐙 **GitHub:** [github.com/IlohFransisca](https://github.com/IlohFransisca)

    *Open to data science roles, freelance work, and collaborations on
    real-world ML problems.*
    """)


# =====================================================================
# Page: Feedback
# =====================================================================
else:
    st.title("Feedback")
    st.markdown(
        "I'd love to hear what you think about this app. "
        "Whether it's a bug, a feature idea, or just a quick comment — "
        "every note helps me improve."
    )

    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Your name (optional)", placeholder="Jane Doe")
        email = st.text_input("Your email (optional)", placeholder="jane@example.com")
        rating = st.slider("How would you rate this app?", 1, 5, 5)
        category = st.selectbox(
            "What's your feedback about?",
            ["General comment", "Bug report", "Feature request", "Model accuracy", "UI / UX"],
        )
        message = st.text_area(
            "Your feedback",
            placeholder="Tell me what you think...",
            height=150,
        )
        submitted = st.form_submit_button("Submit feedback", type="primary")

        if submitted:
            if not message.strip():
                st.error("Please write a short message before submitting.")
            else:
                st.session_state.setdefault("feedback_log", []).append({
                    "name": name or "Anonymous",
                    "email": email,
                    "rating": rating,
                    "category": category,
                    "message": message,
                })
                st.success(
                    "Thank you for your feedback! 🙏 "
                    "I read every message and it genuinely helps."
                )
                st.balloons()

    st.markdown("---")
    st.caption(
        "Prefer email? Reach me at **your.email@example.com** — "
        "I usually reply within a day."
    )

    with st.expander("Session feedback log (visible only to you)"):
        log = st.session_state.get("feedback_log", [])
        if log:
            st.dataframe(pd.DataFrame(log), use_container_width=True)
        else:
            st.write("No feedback submitted in this session yet.")mi")

                with st.expander("Input row sent to model"):
                    st.dataframe(row, use_container_width=True)
            except Exception as e:
                st.exception(e)


# =====================================================================
# Page: Batch Upload
# =====================================================================
elif page == "Batch Upload":
    st.title("Batch Prediction")
    st.caption(
        "Upload a CSV with columns: pickup, delivery, pickup_lat, pickup_lon, "
        "delivery_lat, delivery_lon, distance, equipment, weight, date, market_index."
    )

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())

        required = ["pickup", "delivery", "pickup_lat", "pickup_lon",
                    "delivery_lat", "delivery_lon", "distance",
                    "equipment", "weight", "date", "market_index"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        elif model is None:
            st.error("Model not loaded.")
        else:
            with st.spinner("Predicting..."):
                preds = model.predict(df[required])
            out = df.copy()
            out["predicted_rate"] = np.round(preds, 2)

            st.success(f"Predicted {len(out):,} rows.")
            st.dataframe(out.head(50), use_container_width=True)

            st.download_button(
                "Download predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )


# =====================================================================
# Page: December Chart
# =====================================================================
elif page == "December Chart":
    st.title("December 2025 Predicted Load Rate")
    st.caption("Forecasted daily freight rate for December 2025.")

    if not DECEMBER_PATH.exists():
        st.warning(
            "Missing `december_chart_inputs_filled.csv`. "
            "Place it in `data/` or at the repo root."
        )
    else:
        dec = pd.read_csv(DECEMBER_PATH)
        dec["date"] = pd.to_datetime(dec["date"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dec["date"], dec["predicted_rate"], marker="o", linewidth=2)
        ax.set_ylabel("rate ($)")
        ax.set_xlabel("date")
        ax.set_title("Candidate: December 2025 Predicted Load Rate")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        st.pyplot(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("Min rate", f"${dec['predicted_rate'].min():,.2f}")
        c2.metric("Mean rate", f"${dec['predicted_rate'].mean():,.2f}")
        c3.metric("Max rate", f"${dec['predicted_rate'].max():,.2f}")

        with st.expander("View raw data"):
            st.dataframe(dec, use_container_width=True)


# =====================================================================
# Page: About Me
# =====================================================================
elif page == "About Me":
    st.title("About Me")
    st.markdown("""
    ### 👋 Hi, I'm Fransisca Iloh

    I'm a **Data Scientist** passionate about turning messy, real-world data
    into decisions people can actually act on. My work sits at the intersection
    of **machine learning**, **product thinking**, and **clear communication** —
    because a model only creates value when someone understands what it says.

    ---

    ### 🎯 What I focus on
    - **Applied machine learning** — regression, forecasting, and end-to-end pipelines
    - **Feature engineering** — building signal from raw, imperfect operational data
    - **Model evaluation** — designing validation that reflects the real deployment setting
    - **Deployment** — shipping models as usable tools, not just notebooks

    ---

    ### 🚚 About this project
    This Streamlit app is the deployment layer for a freight rate prediction model.

    **Problem:** Predict the `posted_rate` of a freight load from route, equipment,
    distance, weight, and market conditions.

    **Approach:** A two-stage regressor that combines
    - a `HistGradientBoostingRegressor` for static lane / equipment / weight effects, and
    - a time-aware market-index adjustment (14-day blocks + carried level) to model
      slow rate drift through the year.

    **Validation (time-based folds):**

    | Metric | Value |
    |---|---:|
    | MAE | $57.09 |
    | RMSE | $139.11 |
    | R² | 0.989 |
    | MAPE | 2.36 % |

    **Design choice:** `quote_signal` was deliberately excluded to avoid leakage
    and keep the model aligned with real, pre-quote prediction settings.

    ---

    ### 🛠️ Tools I work with
    Python · pandas · NumPy · scikit-learn · Streamlit · matplotlib · Jupyter · Git

    ---

    ### 📬 Get in touch
    - 📧 **Email:** your.email@example.com
    - 💼 **LinkedIn:** [linkedin.com/in/your-handle](https://www.linkedin.com/)
    - 🐙 **GitHub:** [github.com/IlohFransisca](https://github.com/IlohFransisca)

    *Open to data science roles, freelance work, and collaborations on
    real-world ML problems.*
    """)


# =====================================================================
# Page: Feedback
# =====================================================================
else:
    st.title("Feedback")
    st.markdown(
        "I'd love to hear what you think about this app. "
        "Whether it's a bug, a feature idea, or just a quick comment — "
        "every note helps me improve."
    )

    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Your name (optional)", placeholder="Jane Doe")
        email = st.text_input("Your email (optional)", placeholder="jane@example.com")
        rating = st.slider("How would you rate this app?", 1, 5, 5)
        category = st.selectbox(
            "What's your feedback about?",
            ["General comment", "Bug report", "Feature request", "Model accuracy", "UI / UX"],
        )
        message = st.text_area(
            "Your feedback",
            placeholder="Tell me what you think...",
            height=150,
        )
        submitted = st.form_submit_button("Submit feedback", type="primary")

        if submitted:
            if not message.strip():
                st.error("Please write a short message before submitting.")
            else:
                # Log to session so the user sees it was received
                st.session_state.setdefault("feedback_log", []).append({
                    "name": name or "Anonymous",
                    "email": email,
                    "rating": rating,
                    "category": category,
                    "message": message,
                })
                st.success(
                    "Thank you for your feedback! 🙏 "
                    "I read every message and it genuinely helps."
                )
                st.balloons()

    st.markdown("---")
    st.caption(
        "Prefer email? Reach me at **your.email@example.com** — "
        "I usually reply within a day."
    )

    # Optional: show the current session's feedback (dev-only)
    with st.expander("Session feedback log (visible only to you)"):
        log = st.session_state.get("feedback_log", [])
        if log:
            st.dataframe(pd.DataFrame(log), use_container_width=True)
        else:
            st.write("No feedback submitted in this session yet.")
