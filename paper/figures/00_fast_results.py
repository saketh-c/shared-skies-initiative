"""
00_fast_results.py — Full experimental pipeline for figure rendering.

- Loads p2_processed.xls, adds engineered features.
- Trains MLR/SVR/RF/LightGBM/XGBoost under Random 80/20.
- Trains MLR/RF/LightGBM/XGBoost under full 240-fold LOSO-CV.
  Training data per fold is subsampled to 30k rows for throughput
  (prediction quality on 60k vs 30k train differs <1% R^2 in preliminary
  tests, worth the ~2x speedup). n_estimators reduced to 300 for the
  same reason. Both choices are documented in the paper.
- Computes LightGBM feature importance on the full dataset.
- Saves results.json.
"""
import gc
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# Force unbuffered output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "p2_processed.xls"
OUT_PATH = Path(__file__).parent / "results.json"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------- Load + feature-engineer ---------------------------------------
print("[1/6] Loading data...", flush=True)
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "pm25"]).sort_values(["sensor_id", "date"]).reset_index(drop=True)

df["month"] = df["date"].dt.month
df["dow"] = df["date"].dt.dayofweek
df["day_of_year"] = df["date"].dt.dayofyear

df["pm25_lag1"] = df.groupby("sensor_id")["pm25"].shift(1)
df["pm25_lag7"] = df.groupby("sensor_id")["pm25"].shift(7)
df["pm25_roll7"] = (
    df.groupby("sensor_id")["pm25"].shift(1)
      .rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
)
for col in ["pm25_lag1", "pm25_lag7", "pm25_roll7"]:
    df[col] = df.groupby("sensor_id")[col].transform(lambda s: s.fillna(s.median()))
df[["pm25_lag1", "pm25_lag7", "pm25_roll7"]] = (
    df[["pm25_lag1", "pm25_lag7", "pm25_roll7"]].fillna(df["pm25"].median())
)

FEATURES = [
    "humidity", "temperature", "pressure",
    "ejf_score", "pct_people_of_color", "pct_low_income",
    "traffic_proximity", "superfund_proximity", "rmp_proximity",
    "diesel_pm_proximity", "pct_ling_isolated",
    "latitude", "longitude",
    "month", "dow", "day_of_year",
    "pm25_lag1", "pm25_lag7", "pm25_roll7",
]
TARGET = "pm25"
df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
print(f"   Loaded: {len(df):,} rows, {df['sensor_id'].nunique()} sensors", flush=True)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb


def score(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# ---------- A. Random 80/20 split -----------------------------------------
print("\n[2/6] Random 80/20 split...", flush=True)
X_all = df[FEATURES].values
y_all = df[TARGET].values
Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, random_state=RANDOM_SEED)

random_split = {}
rs_models = {
    "MLR": LinearRegression(),
    "RF": RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=RANDOM_SEED),
    "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, num_leaves=31, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_SEED, verbose=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_SEED, verbosity=0),
}
ens_pred = np.zeros_like(yte, dtype=float)
for name, model in rs_models.items():
    t0 = time.time()
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    random_split[name] = score(yte, p)
    if name in ("RF", "LightGBM", "XGBoost"):
        ens_pred += p / 3
    print(f"   {name:10s} R^2={random_split[name]['r2']:.3f} RMSE={random_split[name]['rmse']:.3f} ({time.time()-t0:.1f}s)", flush=True)

# SVR on 10k subsample
rng = np.random.default_rng(RANDOM_SEED)
idx = rng.choice(len(Xtr), size=10000, replace=False)
svr = Pipeline([("scale", StandardScaler()), ("svr", SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1))])
t0 = time.time()
svr.fit(Xtr[idx], ytr[idx])
random_split["SVR"] = score(yte, svr.predict(Xte))
print(f"   {'SVR':10s} R^2={random_split['SVR']['r2']:.3f} RMSE={random_split['SVR']['rmse']:.3f} ({time.time()-t0:.1f}s)", flush=True)
random_split["Ensemble"] = score(yte, ens_pred)
print(f"   {'Ensemble':10s} R^2={random_split['Ensemble']['r2']:.3f} RMSE={random_split['Ensemble']['rmse']:.3f}", flush=True)

del rs_models, svr
gc.collect()

# ---------- B. 240-fold LOSO-CV with subsampled training ------------------
print("\n[3/6] 240-fold LOSO-CV (MLR/RF/LGBM/XGB, 30k train subsample per fold, reduced n_est)...", flush=True)
sensor_ids = df["sensor_id"].unique()

LOSO_MODELS = ["MLR", "RF", "LightGBM", "XGBoost"]
loso_per_model = {name: {"r2": [], "rmse": [], "mae": [], "n_test": []} for name in LOSO_MODELS + ["Ensemble"]}
held_out_records = []
loso_t0 = time.time()

SUBSAMPLE_N = 30000
rng_sub = np.random.default_rng(RANDOM_SEED)

for fi, sid in enumerate(sensor_ids):
    mask_test = df["sensor_id"].values == sid
    train_df = df.loc[~mask_test]
    test_df = df.loc[mask_test]
    if len(test_df) < 10:
        continue

    # Subsample training to SUBSAMPLE_N rows for speed
    if len(train_df) > SUBSAMPLE_N:
        sub_idx = rng_sub.choice(len(train_df), size=SUBSAMPLE_N, replace=False)
        train_df = train_df.iloc[sub_idx]

    Xtr_i = train_df[FEATURES].values
    ytr_i = train_df[TARGET].values
    Xte_i = test_df[FEATURES].values
    yte_i = test_df[TARGET].values

    ens_pred_i = np.zeros_like(yte_i, dtype=float)
    tree_count = 0
    for name in LOSO_MODELS:
        if name == "MLR":
            m = LinearRegression()
        elif name == "RF":
            m = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=RANDOM_SEED)
        elif name == "LightGBM":
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, num_leaves=31, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_SEED, verbose=-1)
        elif name == "XGBoost":
            m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_SEED, verbosity=0)
        m.fit(Xtr_i, ytr_i)
        pred_i = m.predict(Xte_i)
        s = score(yte_i, pred_i)
        loso_per_model[name]["r2"].append(s["r2"])
        loso_per_model[name]["rmse"].append(s["rmse"])
        loso_per_model[name]["mae"].append(s["mae"])
        loso_per_model[name]["n_test"].append(len(yte_i))
        if name in ("RF", "LightGBM", "XGBoost"):
            ens_pred_i += pred_i / 3
            tree_count += 1
        del m

    s_ens = score(yte_i, ens_pred_i)
    loso_per_model["Ensemble"]["r2"].append(s_ens["r2"])
    loso_per_model["Ensemble"]["rmse"].append(s_ens["rmse"])
    loso_per_model["Ensemble"]["mae"].append(s_ens["mae"])
    loso_per_model["Ensemble"]["n_test"].append(len(yte_i))

    rows_test = test_df[["sensor_id", "ejf_score", "city", "GEOID"]].to_dict("records")
    for y_t, y_p, rec in zip(yte_i, ens_pred_i, rows_test):
        held_out_records.append({
            "y_true": float(y_t),
            "y_pred": float(y_p),
            "sensor_id": int(rec["sensor_id"]),
            "ejf_score": float(rec["ejf_score"]),
            "city": rec["city"],
            "GEOID": int(rec["GEOID"]),
        })

    if (fi + 1) % 10 == 0 or fi == 0:
        dt = time.time() - loso_t0
        xgb_r2 = np.average(loso_per_model["XGBoost"]["r2"], weights=loso_per_model["XGBoost"]["n_test"])
        ens_r2 = np.average(loso_per_model["Ensemble"]["r2"], weights=loso_per_model["Ensemble"]["n_test"])
        print(f"   fold {fi + 1}/{len(sensor_ids)}  XGB={xgb_r2:.3f} Ens={ens_r2:.3f}  ({dt:.0f}s)", flush=True)

    gc.collect()

loso_total_time = time.time() - loso_t0
print(f"   LOSO total: {loso_total_time:.0f}s", flush=True)

# Aggregate
print("\n[4/6] Aggregating LOSO results...", flush=True)
loso_summary = {}
for name in loso_per_model:
    if not loso_per_model[name]["r2"]:
        continue
    r2_arr = np.array(loso_per_model[name]["r2"])
    rmse_arr = np.array(loso_per_model[name]["rmse"])
    mae_arr = np.array(loso_per_model[name]["mae"])
    n_arr = np.array(loso_per_model[name]["n_test"])
    loso_summary[name] = {
        "r2_mean": float(np.average(r2_arr, weights=n_arr)),
        "r2_std": float(np.std(r2_arr)),
        "rmse_mean": float(np.average(rmse_arr, weights=n_arr)),
        "rmse_std": float(np.std(rmse_arr)),
        "mae_mean": float(np.average(mae_arr, weights=n_arr)),
        "mae_std": float(np.std(mae_arr)),
        "n_folds": int(len(r2_arr)),
    }
    print(f"   {name:10s} R^2={loso_summary[name]['r2_mean']:.3f}±{loso_summary[name]['r2_std']:.3f} RMSE={loso_summary[name]['rmse_mean']:.3f}", flush=True)

y_true_all = np.array([r["y_true"] for r in held_out_records])
y_pred_all = np.array([r["y_pred"] for r in held_out_records])
pooled_ensemble = {
    "r2": float(r2_score(y_true_all, y_pred_all)),
    "rmse": float(np.sqrt(mean_squared_error(y_true_all, y_pred_all))),
    "mae": float(mean_absolute_error(y_true_all, y_pred_all)),
    "n": int(len(y_true_all)),
}
print(f"   Pooled Ensemble: R^2={pooled_ensemble['r2']:.3f} RMSE={pooled_ensemble['rmse']:.3f} MAE={pooled_ensemble['mae']:.3f}", flush=True)

# ---------- C. Feature importance -----------------------------------------
print("\n[5/6] Feature importance (LightGBM full-data fit)...", flush=True)
fi_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, num_leaves=31, n_jobs=-1, random_state=RANDOM_SEED, verbose=-1)
fi_model.fit(X_all, y_all)
fi = [{"feature": f, "importance": int(v)} for f, v in zip(FEATURES, fi_model.feature_importances_)]
fi = sorted(fi, key=lambda d: d["importance"], reverse=True)
for f in fi[:5]:
    print(f"   {f['feature']:22s} {f['importance']}", flush=True)

# ---------- D. Sensor metadata --------------------------------------------
sensor_meta = (df.groupby("sensor_id")
                 .agg(lat=("latitude", "mean"),
                      lon=("longitude", "mean"),
                      city=("city", "first"),
                      mean_pm25=("pm25", "mean"),
                      ejf_score=("ejf_score", "mean"),
                      n_days=("pm25", "size"))
                 .reset_index())

# ---------- E. Per-sensor LOSO errors -------------------------------------
by_sensor = defaultdict(lambda: {"y_true": [], "y_pred": []})
for r in held_out_records:
    by_sensor[r["sensor_id"]]["y_true"].append(r["y_true"])
    by_sensor[r["sensor_id"]]["y_pred"].append(r["y_pred"])

per_sensor_loso = []
sm_idx = sensor_meta.set_index("sensor_id")
for sid, dd in by_sensor.items():
    yt = np.array(dd["y_true"])
    yp = np.array(dd["y_pred"])
    per_sensor_loso.append({
        "sensor_id": int(sid),
        "ejf_score": float(sm_idx.loc[sid, "ejf_score"]),
        "city": str(sm_idx.loc[sid, "city"]),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "bias": float(np.mean(yp - yt)),
        "mean_pm25": float(yt.mean()),
        "n": int(len(yt)),
    })

# ---------- F. EJ analysis ------------------------------------------------
print("\n[6/6] EJ analysis...", flush=True)
sm_ok = sensor_meta.dropna(subset=["ejf_score"]).copy()
sm_ok["ej_quartile"] = pd.qcut(sm_ok["ejf_score"], q=4, labels=[1, 2, 3, 4], duplicates="drop")
ej_exceed = {}
ej_mean = {}
for q, g in sm_ok.groupby("ej_quartile", observed=True):
    ej_exceed[str(int(q))] = float((g["mean_pm25"] > 9.0).mean())
    ej_mean[str(int(q))] = float(g["mean_pm25"].mean())
r_val, p_val = pearsonr(sm_ok["ejf_score"], sm_ok["mean_pm25"])
ej_corr = {"r": float(r_val), "p": float(p_val), "n": int(len(sm_ok))}
print(f"   r={ej_corr['r']:.3f}, p={ej_corr['p']:.3g}, n={ej_corr['n']}", flush=True)
print(f"   exceed by Q: {ej_exceed}", flush=True)

# ---------- Output --------------------------------------------------------
summary_stats = {
    "n_rows": int(len(df)),
    "n_sensors": int(df["sensor_id"].nunique()),
    "n_tracts": int(df["GEOID"].nunique()),
    "pm25_mean": float(df["pm25"].mean()),
    "pm25_median": float(df["pm25"].median()),
    "pm25_std": float(df["pm25"].std()),
    "pm25_min": float(df["pm25"].min()),
    "pm25_max": float(df["pm25"].max()),
    "city_counts": df["city"].value_counts().to_dict(),
    "city_sensor_counts": df.groupby("city")["sensor_id"].nunique().to_dict(),
    "date_min": str(df["date"].min().date()),
    "date_max": str(df["date"].max().date()),
}
out = {
    "summary_stats": summary_stats,
    "features": FEATURES,
    "random_split": random_split,
    "loso_summary": loso_summary,
    "loso_pooled_ensemble": pooled_ensemble,
    "feature_importance": fi,
    "sensor_meta": sensor_meta.to_dict("records"),
    "per_sensor_loso": per_sensor_loso,
    "held_out_records": held_out_records,
    "ej_correlation": ej_corr,
    "ej_exceedance_by_quartile": ej_exceed,
    "ej_mean_pm25_by_quartile": ej_mean,
}
with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n✓ Saved → {OUT_PATH}", flush=True)
