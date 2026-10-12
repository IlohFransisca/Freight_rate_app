"""Fit the final model on all cleaned labeled data and save model/bundle.joblib.

    python train_model.py [--train data/train_test.csv] [--validation data/validation.csv]
"""
import argparse
import warnings
from pathlib import Path

import joblib
import pandas as pd

from freight_model import TwoStageRegressor, build_bundle, drop_label_errors

warnings.filterwarnings('ignore')
p = argparse.ArgumentParser()
p.add_argument('--train', default='data/train_test.csv')
p.add_argument('--validation', default='data/validation.csv')
p.add_argument('--out', default='model/bundle.joblib')
a = p.parse_args()

train = pd.read_csv(a.train, parse_dates=['date'])
validation = pd.read_csv(a.validation, parse_dates=['date'])
clean = drop_label_errors(train)
print(f'training on {len(clean):,} rows ({len(train) - len(clean)} label-error rows dropped)')
model = TwoStageRegressor().fit(clean.drop(columns=['posted_rate', 'load_id', 'quote_signal'], errors='ignore'), clean['posted_rate'])
print(f'slope on market_index: {model.slope_:.3f} | carried level: {model.offset_:.4f}')
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
joblib.dump(build_bundle(train, validation, model), a.out, compress=3)
print('saved', a.out)
