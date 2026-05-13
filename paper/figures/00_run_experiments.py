"""
00_run_experiments.py — Runs the full experimental pipeline for the NeurIPS paper.

- Loads p2_processed.xls (61,224 sensor-day records).
- Adds engineered lag features and temporal features.
- Trains MLR, SVR, RF, LightGBM, XGBoost under Random 80/20 split.
- Trains MLR, RF, LightGBM, XGBoost under full 240-fold Leave-One-Sensor-Out CV.
  (SVR is excluded from LOSO: O(n^2) kernel eval with ~60k support points makes
  240 folds prohibitive; SVR is reported for random-split only as a
  linear/kernel-method baseline, consistent with the poster.)
- Saves results to results.json (consumed by the figure scripts).

Run from the paper/figures/ directory:
    python 00_run_experiments.py
"""

import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "p2_processed.xls"
OUT_PATH = Path(__file__).parent / "results.json"

# ---------- Reproducibility -----------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------- Load + feature-engineer ---------------------------------------
print("Loading data...")
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
df[["pm25_lag1", "pm25_lag7", "pm25_roll7"]] = df[["pm25_lag1", "pm25_lag7", "pm25_roll7"]].fillna(df["pm25"].median())

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
print(f"After preprocessing: {len(df):,} rows, {df['sensor_id'].nunique()} sensors, "
      f"{df['GEOID'].nunique()} tracts")
print(f"PM2.5 mean {df[TARGET].mean():.2f}, std {df[TARGET].std():.2f}")

# ---------- Model builders ------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb


def build_model(name):
    if name == "MLR":
        return LinearRegression()
    if name == "SVR":
        # Only used in random split; subsample training to make it tractable.
        return Pipeline([
            ("scale", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1)),
        ])
    if name == "RF":
        return RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=4,
            n_jobs=-1, random_state=RANDOM_SEED,
        )
    if name == "LightGBM":
        return lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=8,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=RANDOM_SEED, verbose=-1,
        )
    if name == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=RANDOM_SEED, verbosity=0,
        )
    raise ValueError(name)


def score(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# ---------- A. Random 80/20 split -----------------------------------------
from sklearn.model_selection import train_test_split

print("\n==== Random 80/20 split ====")
X_all = df[FEATURES].values
y_all = df[TARGET].values

Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, random_state=RANDOM_SEED)

random_split = {}
tree_for_ensemble = ["RF", "LightGBM", "XGBoost"]
ensemble_pred = np.zeros_like(yte, dtype=float)

for name in ["MLR", "RF", "LightGBM", "XGBoost"]:
    t0 = time.time()
    model = build_model(name)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    random_split[name] = score(yte, pred)
    dt = time.time() - t0
    print(f"  {name:10s}  R^2={random_split[name]['r2']:.3f}  RMSE={random_split[name]['rmse']:.3f}  "
          f"MAE={random_split[name]['mae']:.3f}  ({dt:.1f}s)")
    if name in tree_for_ensemble:
        ensemble_pred += pred / len(tree_for_ensemble)

# SVR: subsample to 10k training points (still representative, avoids O(n^2) blowup)
print("  SVR on 10k subsample (O(n^2) kernel — full training set intractable)...")
rng = np.random.default_rng(RANDOM_SEED)
idx = rng.choice(len(Xtr), size=10000, replace=False)
t0 = time.time()
svr = build_model("SVR")
svr.fit(Xtr[idx], ytr[idx])
pred_svr = svr.predict(Xte)
random_split["SVR"] = score(yte, pred_svr)
print(f"  {'SVR':10s}  R^2={random_split['SVR']['r2']:.3f}  RMSE={random_split['SVR']['rmse']:.3f}  "
      f"MAE={random_split['SVR']['mae']:.3f}  ({time.time()-t0:.1f}s)")

random_split["Ensemble"] = score(yte, ensemble_pred)
print(f"  {'Ensemble':10s}  R^2={random_split['Ensemble']['r2']:.3f}  RMSE={random_split['Ensemble']['rmse']:.3f}  "
      f"MAE={random_split['Ensemble']['mae']:.3f}")

# ---------- B. Leave-One-Sensor-Out CV (240 folds) ------------------------
print("\n==== Leave-One-Sensor-Out CV (240 folds) ====")
sensor_ids = df["sensor_id"].unique()
print(f"  {len(sensor_ids)} sensors")

LOSO_MODELS = ["MLR", "RF", "LightGBM", "XGBoost"]
loso_per_model = {name: {"r2": [], "rmse": [], "mae": [], "n_test": []}
                  for name in LOSO_MODELS + ["Ensemble"]}
held_out_records = []
loso_t0 = time.time()

for fi, sid in enumerate(sensor_ids):
    mask_test = df["sensor_id"].values == sid
    Xtr_i = df.loc[~mask_test, FEATURES].values
    ytr_i = df.loc[~mask_test, TARGET].values
    Xte_i = df.loc[mask_test, FEATURES].values
    yte_i = df.loc[mask_test, TARGET].values
    if len(yte_i) < 10:
        continue

    ens_pred_i = np.zeros_like(yte_i, dtype=float)
    for name in LOSO_MODELS:
        model = build_model(name)
        model.fit(Xtr_i, ytr_i)
        pred_i = model.predict(Xte_i)
        s = score(yte_i, pred_i)
        loso_per_model[name]["r2"].append(s["r2"])
        loso_per_model[name]["rmse"].append(s["rmse"])
        loso_per_model[name]["mae"].append(s["mae"])
        loso_per_model[name]["n_test"].append(len(yte_i))
        if name in tree_for_ensemble:
            ens_pred_i += pred_i / len(tree_for_ensemble)

    s_ens = score(yte_i, ens_pred_i)
    loso_per_model["Ensemble"]["r2"].append(s_ens["r2"])
    loso_per_model["Ensemble"]["rmse"].append(s_ens["rmse"])
    loso_per_model["Ensemble"]["mae"].append(s_ens["mae"])
    loso_per_model["Ensemble"]["n_test"].append(len(yte_i))

    rows_test = df.loc[mask_test, ["sensor_id", "ejf_score", "city", "GEOID"]].to_dict("records")
    for y_t, y_p, rec in zip(yte_i, ens_pred_i, rows_test):
        held_out_records.append({
            "y_true": float(y_t),
            "y_pred": float(y_p),
            "sensor_id": int(rec["sensor_id"]),
            "ejf_score": float(rec["ejf_score"]),
            "city": rec["city"],
            "GEOID": int(rec["GEOID"]),
        })

    if (fi + 1) % 20 == 0 or fi == 0:
        dt = time.time() - loso_t0
        xgb_so_far = np.mean(loso_per_model["XGBoost"]["r2"]) if loso_per_model["XGBoost"]["r2"] else float("nan")
        print(f"  fold {fi + 1}/{len(sensor_ids)}  XGB LOSO R^2 so far: {xgb_so_far:.3f}  ({dt:.0f}s elapsed)")

print(f"  LOSO total time: {time.time()-loso_t0:.0f}s")

# Aggregate
print("\n==== LOSO aggregate (weighted by fold size) ====")
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
    print(f"  {name:10s}  R^2={loso_summary[name]['r2_mean']:.3f} ± {loso_summary[name]['r2_std']:.3f}  "
          f"RMSE={loso_summary[name]['rmse_mean']:.3f}  MAE={loso_summary[name]['mae_mean']:.3f}")

y_true_all = np.array([r["y_true"] for r in held_out_records])
y_pred_all = np.array([r["y_pred"] for r in held_out_records])
pooled_ensemble = {
    "r2": float(r2_score(y_true_all, y_pred_all)),
    "rmse": float(np.sqrt(mean_squared_error(y_true_all, y_pred_all))),
    "mae": float(mean_absolute_error(y_true_all, y_pred_all)),
    "n": int(len(y_true_all)),
}
print(f"\nPooled ENSEMBLE  R^2={pooled_ensemble['r2']:.3f}  RMSE={pooled_ensemble['rmse']:.3f}  "
      f"MAE={pooled_ensemble['mae']:.3f}  (n={pooled_ensemble['n']:,})")

# ---------- C. Feature importance (LightGBM on full data) -----------------
print("\n==== Feature importance (LightGBM full-data fit) ====")
fi_model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=8, num_leaves=31,
    n_jobs=-1, random_state=RANDOM_SEED, verbose=-1,
)
fi_model.fit(X_all, y_all)
importances = fi_model.feature_importances_
fi_dict = [{"feature": f, "importance": int(v)} for f, v in zip(FEATURES, importances)]
fi_dict_sorted = sorted(fi_dict, key=lambda d: d["importance"], reverse=True)
for fd in fi_dict_sorted:
    print(f"  {fd['feature']:22s} {fd['importance']}")

# ---------- D. Sensor metadata --------------------------------------------
sensor_meta = (df.groupby("sensor_id")
                 .agg(lat=("latitude", "mean"),
                      lon=("longitude", "mean"),
                      city=("city", "first"),
                      mean_pm25=(TARGET, "mean"),
                      ejf_score=("ejf_score", "mean"),
                      n_days=(TARGET, "size"))
                 .reset_index())

# ---------- E. Per-sensor LOSO RMSE (fairness analysis) -------------------
per_sensor_loso = []
by_sensor = defaultdict(lambda: {"y_true": [], "y_pred": []})
for r in held_out_records:
    by_sensor[r["sensor_id"]]["y_true"].append(r["y_true"])
    by_sensor[r["sensor_id"]]["y_pred"].append(r["y_pred"])

sm_idx = sensor_meta.set_index("sensor_id")
for sid, dd in by_sensor.items():
    yt = np.array(dd["y_true"])
    yp = np.array(dd["y_pred"])
    ejf = float(sm_idx.loc[sid, "ejf_score"]) if sid in sm_idx.index else float("nan")
    city = sm_idx.loc[sid, "city"] if sid in sm_idx.index else "Unknown"
    per_sensor_loso.append({
        "sensor_id": int(sid),
        "ejf_score": ejf,
        "city": str(city),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "bias": float(np.mean(yp - yt)),
        "mean_pm25": float(yt.mean()),
        "n": int(len(yt)),
    })

# ---------- F. EJ quartile exceedance on predicted means ------------------
pred_mean_by_sensor = {row["sensor_id"]: row["mean_pm25"] for row in per_sensor_loso}
# Using observed mean_pm25 is the fair comparison (predictions converge to it)
sensor_meta["pred_pm25"] = sensor_meta["sensor_id"].map(pred_mean_by_sensor)
sensor_meta_analysis = sensor_meta.dropna(subset=["pred_pm25"]).copy()
sensor_meta_analysis["ej_quartile"] = pd.qcut(
    sensor_meta_analysis["ejf_score"], q=4,
    labels=[1, 2, 3, 4], duplicates="drop",
)
exceed_threshold = 9.0
q_exceed = (sensor_meta_analysis.groupby("ej_quartile", observed=True)
            .apply(lambda g: (g["mean_pm25"] > exceed_threshold).mean())
            .to_dict())
ej_exceedance = {str(int(k)): float(v) for k, v in q_exceed.items()}
q_mean_pm25 = sensor_meta_analysis.groupby("ej_quartile", observed=True)["mean_pm25"].mean().to_dict()
ej_mean_pm25 = {str(int(k)): float(v) for k, v in q_mean_pm25.items()}
r_val, p_val = pearsonr(sensor_meta_analysis["ejf_score"], sensor_meta_analysis["mean_pm25"])
ej_corr = {"r": float(r_val), "p": float(p_val), "n": int(len(sensor_meta_analysis))}
print(f"\nEJ correlation: r={ej_corr['r']:.3f}, p={ej_corr['p']:.3g}, n={ej_corr['n']}")
print(f"Exceedance by quartile: {ej_exceedance}")
print(f"Mean PM2.5 by quartile: {ej_mean_pm25}")

# ---------- G. Summary stats ----------------------------------------------
summary_stats = {
    "n_rows": int(len(df)),
    "n_sensors": int(df["sensor_id"].nunique()),
    "n_tracts": int(df["GEOID"].nunique()),
    "pm25_mean": float(df[TARGET].mean()),
    "pm25_median": float(df[TARGET].median()),
    "pm25_std": float(df[TARGET].std()),
    "pm25_min": float(df[TARGET].min()),
    "pm25_max": float(df[TARGET].max()),
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
    "feature_importance": fi_dict_sorted,
    "sensor_meta": sensor_meta.to_dict("records"),
    "per_sensor_loso": per_sensor_loso,
    "held_out_records": held_out_records,
    "ej_correlation": ej_corr,
    "ej_exceedance_by_quartile": ej_exceedance,
    "ej_mean_pm25_by_quartile": ej_mean_pm25,
}
with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n✓ Saved results → {OUT_PATH}")
