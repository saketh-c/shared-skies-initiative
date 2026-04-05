"""
02_train_model.py
Trains an RF + LightGBM + XGBoost ensemble on p2_processed.xls.
Saves the model bundle to models/ensemble.joblib.

Run from the project root:
    python pipeline/02_train_model.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Feature order is fixed — do not change without retraining
FEATURES = [
    "humidity",
    "temperature",
    "pressure",
    "ejf_score",
    "pct_people_of_color",
    "pct_low_income",
    "traffic_proximity",
    "superfund_proximity",
    "rmp_proximity",
    "diesel_pm_proximity",
    "pct_ling_isolated",
    "latitude",
    "longitude",
    "month",
    "hour",        # Always 0 in current training data (daily averages).
    "dow",         # Will use live value at inference time.
    "day_of_year",
]
TARGET = "pm25"


def load_file(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        try:
            return pd.read_excel(path, engine="xlrd")
        except Exception:
            return pd.read_excel(path, engine="openpyxl")


def load_data() -> pd.DataFrame:
    print("Loading p2_processed...")
    df = load_file(os.path.join(DATA_DIR, "p2_processed.xls"))
    print(f"  Raw rows: {len(df)},  columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["month"]       = df["date"].dt.month
    df["hour"]        = df["date"].dt.hour   # Mostly 0 in daily data
    df["dow"]         = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear

    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features (will fill 0): {missing}")
        for f in missing:
            df[f] = 0.0

    df = df.dropna(subset=[TARGET])
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    print(f"  After cleaning: {len(df)} rows")
    print(f"  PM2.5 range: {df[TARGET].min():.2f} – {df[TARGET].max():.2f} µg/m³")
    print(f"  Cities: {df['city'].unique() if 'city' in df.columns else 'N/A'}")
    return df


def train_ensemble(X_train, y_train):
    models = {}

    print("\nTraining Random Forest (300 trees)...")
    models["rf"] = RandomForestRegressor(
        n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=42
    )
    models["rf"].fit(X_train, y_train)

    print("Training LightGBM...")
    models["lgbm"] = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        n_jobs=-1, random_state=42, verbose=-1
    )
    models["lgbm"].fit(X_train, y_train)

    print("Training XGBoost...")
    models["xgb"] = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        n_jobs=-1, random_state=42, verbosity=0
    )
    models["xgb"].fit(X_train, y_train)

    return models


def compute_weights(models, X_test, y_test):
    """Weight each model inversely proportional to its MSE on the test set."""
    mses = {}
    print("\n── Test-set performance ──────────────────────────────────────")
    for name, model in models.items():
        pred = model.predict(X_test)
        mse  = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, pred)
        mae  = mean_absolute_error(y_test, pred)
        print(f"  {name.upper():6s}  RMSE={rmse:.3f}  R²={r2:.3f}  MAE={mae:.3f}")
        mses[name] = mse

    inv = {k: 1.0 / v for k, v in mses.items()}
    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}

    ensemble_pred = sum(weights[n] * models[n].predict(X_test) for n in models)
    e_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    e_r2   = r2_score(y_test, ensemble_pred)
    e_mae  = mean_absolute_error(y_test, ensemble_pred)
    print(f"  {'ENSEMBLE':6s}  RMSE={e_rmse:.3f}  R²={e_r2:.3f}  MAE={e_mae:.3f}")
    print(f"  Weights → RF:{weights['rf']:.3f}  LGBM:{weights['lgbm']:.3f}  XGB:{weights['xgb']:.3f}")

    return weights


def spatial_holdout(df):
    """
    Hold out all DFW sensors; train on Austin+Houston+San Antonio.
    This shows how well the model extrapolates to new locations — the
    same scenario as predicting for Dallas tracts without sensor history.
    """
    if "city" not in df.columns:
        print("\nSkipping spatial holdout (no 'city' column).")
        return

    dfw_cities = {"Dallas", "Fort Worth", "Arlington", "Irving", "Garland"}
    mask_dfw = df["city"].isin(dfw_cities)
    train_s = df[~mask_dfw]
    test_s  = df[mask_dfw]

    if len(test_s) == 0:
        # Try partial match
        mask_dfw = df["city"].str.lower().str.contains("dallas|fort worth|arlington")
        train_s = df[~mask_dfw]
        test_s  = df[mask_dfw]

    if len(test_s) == 0:
        print("\nSkipping spatial holdout (no DFW rows found).")
        return

    print(f"\n── Spatial holdout (train on non-DFW, test on DFW) ──────────")
    print(f"  Train cities: {train_s['city'].unique()}")
    print(f"  Test city rows: {len(test_s)}")

    X_s_train = train_s[FEATURES].values
    y_s_train = train_s[TARGET].values
    X_s_test  = test_s[FEATURES].values
    y_s_test  = test_s[TARGET].values

    spatial_models = train_ensemble(X_s_train, y_s_train)
    for name, model in spatial_models.items():
        pred = model.predict(X_s_test)
        rmse = np.sqrt(mean_squared_error(y_s_test, pred))
        r2   = r2_score(y_s_test, pred)
        print(f"  {name.upper():6s}  RMSE={rmse:.3f}  R²={r2:.3f}  (spatial holdout)")


if __name__ == "__main__":
    df = load_data()

    X = df[FEATURES].values
    y = df[TARGET].values

    # Random 80/20 split for standard evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain/test split: {len(X_train)} / {len(X_test)}")

    models = train_ensemble(X_train, y_train)
    weights = compute_weights(models, X_test, y_test)

    # Spatial holdout for honest extrapolation estimate
    spatial_holdout(df)

    # Retrain all three on the full dataset before saving
    print("\nRetraining on full dataset...")
    full_models = train_ensemble(X, y)

    bundle = {
        "models":       full_models,
        "weights":      weights,
        "feature_names": FEATURES,
    }

    out_path = os.path.join(MODELS_DIR, "ensemble.joblib")
    joblib.dump(bundle, out_path)
    print(f"\nSaved model bundle → {out_path}")

    # Also save feature names as JSON for easy inspection
    feat_path = os.path.join(MODELS_DIR, "feature_names.json")
    with open(feat_path, "w") as f:
        json.dump(FEATURES, f, indent=2)
    print(f"Saved feature list  → {feat_path}")
    print("\nAll done! Run the backend next: uvicorn backend.main:app --reload")
