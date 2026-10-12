"""Freight rate predictor (Streamlit).  Run:
    streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import freight_model  # noqa: F401  (needed so joblib can unpickle the model classes)
from freight_model import REQUIRED, default_distance, market_for, predict_frame

TYPICAL_ERROR = 0.0236  # mean MAPE over the three forward-in-time validation folds

st.set_page_config(
    page_title="Freight Rate Predictor",
    page_icon="🚚",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_bundle():
    base = Path(__file__).parent

    def find(name):
        # data may sit in data/ or next to app.py
        for p in (base / "data" / name, base / name):
            if p.exists():
                return p
        return base / "data" / name

    with st.spinner(
        "Preparing the model. If it has to retrain this takes a minute "
        "or so, but only on the first load."
    ):
        try:
            return freight_model.load_or_train(
                base / "model" / "bundle.joblib",
                find("train_test.csv"),
                find("validation.csv"),
            )
        except Exception as e:
            st.error(f"Could not prepare the model: {e}")
            st.stop()


bundle, model_status = load_bundle()
cities = bundle["cities"]

st.title("🚚 Freight Rate Predictor")
st.caption(
    "Estimates the posted rate for a truckload from lane, equipment, "
    "weight, distance and date."
)

# ---------------------------------------------------------------------
# Sidebar: the load
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Load details")

    pickup = st.selectbox(
        "Pickup city",
        cities,
        index=cities.index("Lexington") if "Lexington" in cities else 0,
    )
    delivery = st.selectbox(
        "Delivery city",
        cities,
        index=cities.index("Fort Wayne") if "Fort Wayne" in cities else 1,
    )
    equipment = st.selectbox(
        "Equipment",
        bundle["equipment"],
        index=bundle["equipment"].index("Dry Van")
        if "Dry Van" in bundle["equipment"] else 0,
    )

    w_lo, w_hi, w_med = bundle["weight_range"]
    weight = st.number_input(
        "Weight (lb)",
        min_value=1000,
        max_value=int(w_hi),
        value=32000,
        step=500,
    )

    if pickup == delivery:
        st.warning("Pickup and delivery are the same city. Pick two different cities.")
        st.stop()

    new_cities = [c for c in (pickup, delivery) if c not in bundle["trained_cities"]]
    if new_cities:
        st.info(
            f"{', '.join(new_cities)}: not in the training data. The estimate "
            "relies on the city's coordinates and distance, so it may be a "
            "little less accurate."
        )

    dist_default, dist_note = default_distance(bundle, pickup, delivery)
    distance = st.number_input(
        "Distance (miles)",
        min_value=1.0,
        max_value=5000.0,
        value=float(dist_default),
        step=10.0,
        key=f"dist_{pickup}_{delivery}",
    )
    st.caption(f"Default: {dist_note}.")


def one_row(date, market=None):
    row = {
        "pickup": pickup,
        "delivery": delivery,
        "distance": float(distance),
        "equipment": equipment,
        "weight": float(weight),
        "date": pd.Timestamp(date),
    }
    if market is not None:
        row["market_index"] = float(market)
    return pd.DataFrame([row])


tab1, tab2, tab3, tab4 = st.tabs(
    ["Single prediction", "Date sweep", "Batch (CSV)", "About the model"]
)

# ---------------------------------------------------------------------
# Tab 1: single load
# ---------------------------------------------------------------------
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        date = st.date_input(
            "Pickup date",
            value=pd.Timestamp("2025-12-01").date(),
        )
        auto_mkt, n_missing = market_for(bundle, [date])
        override = st.checkbox("Override market index")
        market = (
            st.number_input(
                "Market index",
                min_value=0.5,
                max_value=2.0,
                value=float(round(auto_mkt.iloc[0], 3)),
                step=0.01,
            )
            if override else None
        )
        if n_missing and not override:
            st.info(
                "This date is outside the data, so the latest 14-day average "
                "market index is used."
            )
    with c2:
        pred = float(predict_frame(bundle, one_row(date, market))[0])
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted rate", f"${pred:,.0f}")
        m2.metric("Per mile", f"${pred / distance:,.2f}")
        m3.metric(
            "Market index used",
            f"{market if override else auto_mkt.iloc[0]:.3f}",
        )
        st.caption(
            f"Typical error in validation is about {TYPICAL_ERROR:.1%}, so "
            f"roughly ${pred * (1 - TYPICAL_ERROR):,.0f} to "
            f"${pred * (1 + TYPICAL_ERROR):,.0f}. "
            "This is a guide, not a guarantee."
        )

# ---------------------------------------------------------------------
# Tab 2: date sweep
# ---------------------------------------------------------------------
with tab2:
    st.write(
        f"Same load ({pickup} to {delivery}, {equipment}, "
        f"{weight:,.0f} lb, {distance:,.0f} mi) over a range of dates. "
        "Only the date, and therefore the market index, changes."
    )
    d1, d2 = st.columns(2)
    start = d1.date_input(
        "Start",
        value=pd.Timestamp("2025-12-01").date(),
        key="sweep_start",
    )
    end = d2.date_input(
        "End",
        value=pd.Timestamp("2025-12-31").date(),
        key="sweep_end",
    )

    if end < start:
        st.warning("End date must be on or after the start date.")
    elif (end - start).days > 365:
        st.warning("Please pick a range of at most one year.")
    else:
        days = pd.date_range(start, end, freq="D")
        frame = pd.DataFrame({
            "pickup": pickup,
            "delivery": delivery,
            "distance": float(distance),
            "equipment": equipment,
            "weight": float(weight),
            "date": days,
        })
        out = pd.DataFrame({
            "date": days,
            "predicted_rate": predict_frame(bundle, frame).round(2),
        })
        _, miss = market_for(bundle, days)
        if miss:
            st.info(
                f"{miss} date(s) are outside the data and use the latest "
                "14-day average market index."
            )
        st.line_chart(
            out.set_index("date")["predicted_rate"],
            y_label="Predicted rate ($)",
        )
        a, b, c = st.columns(3)
        a.metric("Lowest", f"${out.predicted_rate.min():,.0f}")
        b.metric("Average", f"${out.predicted_rate.mean():,.0f}")
        c.metric("Highest", f"${out.predicted_rate.max():,.0f}")
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            out.to_csv(index=False),
            "date_sweep.csv",
            "text/csv",
        )

# ---------------------------------------------------------------------
# Tab 3: batch
# ---------------------------------------------------------------------
with tab3:
    st.write(
        "Upload a CSV with columns: "
        + ", ".join(f"`{c}`" for c in REQUIRED)
        + ". An optional `market_index` column overrides the automatic value."
    )
    template = pd.DataFrame({
        "pickup": ["Lexington", "Dallas"],
        "delivery": ["Fort Wayne", "Nashville"],
        "distance": [360, 650],
        "equipment": ["Dry Van", "Reefer"],
        "weight": [32000, 28000],
        "date": ["2025-12-01", "2025-12-02"],
    })
    st.download_button(
        "Download template",
        template.to_csv(index=False),
        "batch_template.csv",
        "text/csv",
    )

    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            missing_cols = [c for c in REQUIRED if c not in df.columns]
            if missing_cols:
                st.error("Missing columns: " + ", ".join(missing_cols))
            else:
                bad = (
                    df[REQUIRED].isna().any(axis=1)
                    | ~df["equipment"].isin(bundle["equipment"])
                )
                if bad.any():
                    st.warning(
                        f"{int(bad.sum())} row(s) have missing values or an "
                        "unknown equipment type and were skipped."
                    )
                good = df.loc[~bad].copy()
                good["predicted_rate"] = predict_frame(bundle, good).round(2)
                st.success(f"Predicted {len(good):,} loads.")
                st.dataframe(
                    good.head(200),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download predictions",
                    good.to_csv(index=False),
                    "predictions.csv",
                    "text/csv",
                )
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Could not process this file: {e}")

# ---------------------------------------------------------------------
# Tab 4: about
# ---------------------------------------------------------------------
with tab4:
    st.markdown(f"""
**What it is.** A gradient-boosting model on log(rate) learns load, lane and
equipment effects. A small linear term adds the effect of the market index,
and a carried-forward level accounts for the slow drift in rates.

**Trained on** {bundle['train_dates'][0]} to {bundle['train_dates'][1]}, with
646 rows of impossible rates removed. `quote_signal` is not used because its
relationship with the rate changes from month to month.

**Model status:** {model_status}.

**Validation** (three forward-in-time folds, mean): MAE $57.09, RMSE $139.11,
R² 0.989, MAPE 2.36%.

**Things to know**
- The market index for a date comes from the November/December data files;
  dates outside the data use the latest 14-day average.
- Distances default to the typical distance for the lane and can be edited.
- 8 cities in the validation data (Allentown, Charlotte, Chicago, Jackson,
  Knoxville, Laredo, Norfolk, San Diego) were never seen in training. The
  model estimates them from coordinates, which in a hold-out test cost a
  small amount of accuracy.
- The data ends in October, so holiday effects in late December are not learned.
- Rates are estimates of the posted rate, not quotes.
""")
