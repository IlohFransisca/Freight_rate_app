"""Model code shared by train_model.py and app.py (same model as the Colab notebook)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

STATIC_NUM = [
    'distance', 'log_distance', 'weight', 'weight_missing',
    'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
    'bearing_ew', 'bearing_ns',
]
STATIC_CAT = ['pickup', 'delivery', 'equipment']
T0 = pd.Timestamp('2025-01-01')
REQUIRED = ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date']


class FreightFeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean raw freight rows. quote_signal is deliberately NOT used."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['dow'] = df['date'].dt.dayofweek
        for c in ['distance', 'weight', 'market_index',
                  'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['weight_missing'] = df['weight'].isna().astype(int)
        df['weight'] = df['weight'].abs().clip(upper=47500)
        day_mean = df.groupby('date')['market_index'].transform('mean')
        df['market_index'] = df['market_index'].fillna(day_mean)
        df['log_distance'] = np.log(df['distance'])
        df['bearing_ew'] = df['delivery_lon'] - df['pickup_lon']
        df['bearing_ns'] = df['delivery_lat'] - df['pickup_lat']
        return df


class TwoStageRegressor(BaseEstimator, RegressorMixin):
    """log(rate) = boosted trees on lane/equipment/weight + slope*market_index + carried-forward level."""

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
            ('num', 'passthrough', STATIC_NUM),
            ('cat', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
                encoded_missing_value=-1), STATIC_CAT),
        ])
        Z = self.prep_.fit_transform(self._static(d))
        cat_mask = [False] * len(STATIC_NUM) + [True] * len(STATIC_CAT)

        days = (d['date'] - T0).dt.days.to_numpy()
        blk = days // self.block
        B = (blk[:, None] == np.unique(blk)[None, :]).astype(float)
        m = d['market_index'].to_numpy()[:, None]
        A = np.c_[m, B]
        lvl = np.zeros(len(d))

        for _ in range(self.iters):
            self.gbm_ = HistGradientBoostingRegressor(
                categorical_features=cat_mask,
                max_iter=300,
                learning_rate=0.07,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=self.seed,
            ).fit(Z, y - lvl)
            r = y - self.gbm_.predict(Z)
            coef = np.linalg.lstsq(A, r, rcond=None)[0]
            lvl = A @ coef

        self.slope_ = float(coef[0])
        resid_level = r - self.slope_ * m[:, 0]
        self.offset_ = float(
            resid_level[days >= days.max() - self.carry_days + 1].mean()
        )
        return self

    def predict(self, X):
        d = self.fe_.transform(X)
        Z = self.prep_.transform(self._static(d))
        return np.exp(
            self.gbm_.predict(Z)
            + self.slope_ * d['market_index'].to_numpy()
            + self.offset_
        )


def drop_label_errors(df, lo=1.0, hi=5.0):
    rpm = df['posted_rate'] / df['distance']
    return df.loc[~((rpm < lo) | (rpm > hi))].reset_index(drop=True)


def haversine_miles(lat1, lon1, lat2, lon2):
    a, b, c, d = map(np.radians, (lat1, lon1, lat2, lon2))
    x = (
        np.sin((c - a) / 2) ** 2
        + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    )
    return 3958.8 * 2 * np.arcsin(np.sqrt(x))


def build_bundle(train: pd.DataFrame, validation: pd.DataFrame,
                 model: TwoStageRegressor) -> dict:
    """Model + lookup tables. City coordinates come from train AND validation:
    validation.csv has 8 cities that never appear in training."""

    def roles(df):
        return pd.concat([
            df[['pickup', 'pickup_lat', 'pickup_lon']]
                .set_axis(['city', 'lat', 'lon'], axis=1),
            df[['delivery', 'delivery_lat', 'delivery_lon']]
                .set_axis(['city', 'lat', 'lon'], axis=1),
        ])

    coords = (
        pd.concat([roles(train), roles(validation)])
        .groupby('city')[['lat', 'lon']]
        .median()
    )
    trained_cities = sorted(set(train['pickup']) | set(train['delivery']))

    lane_dist = (
        train.groupby(['pickup', 'delivery'])['distance']
             .median().round(0).to_dict()
    )

    hv = haversine_miles(
        train.pickup_lat, train.pickup_lon,
        train.delivery_lat, train.delivery_lon,
    )
    road_factor = float((train['distance'] / hv.replace(0, np.nan)).median())

    mk = pd.concat([
        train[['date', 'market_index']],
        validation[['date', 'market_index']],
    ])
    mk['date'] = pd.to_datetime(mk['date'])
    market_s = mk.groupby('date')['market_index'].mean()
    market_by_day = {
        k.strftime('%Y-%m-%d'): float(v) for k, v in market_s.items()
    }

    weights = train['weight'].abs()

    return {
        'model': model,
        'cities': sorted(coords.index),
        'trained_cities': trained_cities,
        'equipment': sorted(train['equipment'].unique()),
        'coords': {c: (float(r.lat), float(r.lon)) for c, r in coords.iterrows()},
        'lane_distance': lane_dist,
        'road_factor': road_factor,
        'market_by_day': market_by_day,
        'market_fallback': float(market_s.iloc[-14:].mean()),
        'weight_range': (
            float(weights.quantile(0.01)), 47500.0, float(weights.median())
        ),
        'train_dates': (
            str(pd.to_datetime(train['date']).min().date()),
            str(pd.to_datetime(train['date']).max().date()),
        ),
    }


def default_distance(bundle, pickup, delivery):
    d = bundle['lane_distance'].get((pickup, delivery))
    if d:
        return float(d), 'median distance for this lane in the training data'
    (la1, lo1), (la2, lo2) = bundle['coords'][pickup], bundle['coords'][delivery]
    return (
        float(round(haversine_miles(la1, lo1, la2, lo2) * bundle['road_factor'])),
        'estimated from city coordinates (this lane is not in the training data)',
    )


def market_for(bundle, dates):
    """Market index for each date; dates outside the data use the latest 14-day average."""
    dates = pd.to_datetime(pd.Series(dates))
    vals = dates.dt.strftime('%Y-%m-%d').map(bundle['market_by_day'])
    missing = int(vals.isna().sum())
    return vals.fillna(bundle['market_fallback']).astype(float), missing


def prepare_frame(bundle, df: pd.DataFrame) -> pd.DataFrame:
    """Add coordinates and market_index (if not supplied) to rows with REQUIRED columns."""
    f = df.copy()
    f['date'] = pd.to_datetime(f['date'])

    unknown = sorted((set(f['pickup']) | set(f['delivery'])) - set(bundle['coords']))
    if unknown:
        raise ValueError(f"Unknown cities: {', '.join(map(str, unknown))}")

    for role in ('pickup', 'delivery'):
        for k, name in enumerate(('lat', 'lon')):
            col = f'{role}_{name}'
            looked = f[role].map(lambda c, k=k: bundle['coords'][c][k])
            f[col] = (
                pd.to_numeric(f[col], errors='coerce').fillna(looked)
                if col in f.columns else looked
            )

    if 'market_index' not in f.columns:
        f['market_index'] = np.nan
    auto, _ = market_for(bundle, f['date'])
    f['market_index'] = f['market_index'].fillna(auto)
    return f


def predict_frame(bundle, df: pd.DataFrame) -> np.ndarray:
    return bundle['model'].predict(prepare_frame(bundle, df))


def _smoke_test(bundle) -> None:
    row = pd.DataFrame([{
        'pickup': 'Lexington', 'delivery': 'Fort Wayne', 'distance': 360.0,
        'equipment': 'Dry Van', 'weight': 32000.0, 'date': '2025-12-01',
    }])
    p = float(predict_frame(bundle, row)[0])
    if not np.isfinite(p) or not (100 < p < 5000):
        raise RuntimeError(f'smoke-test prediction out of range: {p}')


def _read_csvs(train_path, validation_path):
    train = pd.read_csv(train_path, parse_dates=['date'])
    validation = pd.read_csv(validation_path, parse_dates=['date'])
    return train, validation


def _build_bundle_from_model(model, train_path, validation_path):
    if not (train_path.exists() and validation_path.exists()):
        raise FileNotFoundError(
            f'CSVs missing at {train_path} / {validation_path}.'
        )
    train, validation = _read_csvs(train_path, validation_path)
    bundle = build_bundle(train, validation, model)
    _smoke_test(bundle)
    return bundle


def load_or_train(bundle_path, train_path, validation_path):
    """Returns (bundle, status)."""
    import warnings
    from pathlib import Path
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning

    bundle_path = Path(bundle_path)
    train_path = Path(train_path)
    validation_path = Path(validation_path)

    reason = 'no bundle found'
    if bundle_path.exists():
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                bundle = joblib.load(bundle_path)
            if any(issubclass(w.category, InconsistentVersionWarning) for w in caught):
                reason = 'bundle saved with a different scikit-learn version'
            else:
                _smoke_test(bundle)
                return bundle, 'loaded from bundle'
        except Exception as e:
            reason = f'bundle unusable ({type(e).__name__})'

    model_candidates = [
        bundle_path.parent.parent / 'freight_rate_model.joblib',
        bundle_path.parent / 'freight_rate_model.joblib',
        bundle_path.parent.parent / 'final_model.joblib',
        bundle_path.parent / 'final_model.joblib',
    ]
    for model_file in model_candidates:
        if not model_file.exists():
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = joblib.load(model_file)
            if any(issubclass(w.category, InconsistentVersionWarning) for w in caught):
                reason = f'{model_file.name} saved with a different scikit-learn version'
                break
            bundle = _build_bundle_from_model(model, train_path, validation_path)
            try:
                bundle_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(bundle, bundle_path, compress=3)
            except OSError:
                pass
            return bundle, f'loaded from {model_file.name}'
        except FileNotFoundError:
            reason = f'{model_file.name} found but the CSVs are missing'
            break
        except Exception as e:
            reason = f'{model_file.name} unusable ({type(e).__name__})'
            continue

    if not (train_path.exists() and validation_path.exists()):
        raise FileNotFoundError(
            f'No usable model ({reason}) and the training data was not found at '
            f'{train_path} / {validation_path}.'
        )
    train, validation = _read_csvs(train_path, validation_path)
    clean = drop_label_errors(train)
    model = TwoStageRegressor().fit(
        clean.drop(columns=['posted_rate', 'load_id', 'quote_signal'], errors='ignore'),
        clean['posted_rate'],
    )
    bundle = build_bundle(train, validation, model)
    _smoke_test(bundle)
    try:
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, bundle_path, compress=3)
    except OSError:
        pass
    return bundle, f'retrained: {reason}'
