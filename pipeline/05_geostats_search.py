"""
05_geostats_search.py — Proper geostatistics / residual-learning search.

Lessons from 04_kriging_hybrid.py (which HURT LOSO):
  - Per-sensor mean residuals encode sensor-specific biases, not spatial structure.
  - Applying them as-is to neighbours adds noise, not signal.

This script is smarter:
  1) Runs LOSO-CV once and CACHES per-row ML predictions to disk
     (next run reloads from cache → skips the 30-min training loop).
  2) Evaluates MANY correction strategies against the same cached LOSO run:
        - shrunk IDW (various k, power, shrinkage, cap)
        - ordinary kriging with exponential / spherical / gaussian variograms
        - residual-learning: ridge regression on (lat, lon, ejf, pm25) → residual
        - residual-learning: shallow RandomForest on same features
  3) Compares each strategy's hybrid LOSO R² to ML-only baseline.
  4) ONLY saves `models/kriging_corrections.json` if the best strategy beats
     ML-only by at least +0.01 R² AND the Pearson correlation between
     ML predictions and applied correction is < 0.3 (not over-correcting).
"""
import json
import os
import pickle
import time
import warnings
from itertools import product
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PIPELINE_DIR = ROOT / "pipeline"
CACHE_PATH = PIPELINE_DIR / ".loso_cache.pkl"
TARGET = "pm25"

FEATURES = [
    "humidity", "temperature", "pressure", "wind_speed", "precipitation",
    "ejf_score", "pct_people_of_color", "pct_low_income",
    "traffic_proximity", "superfund_proximity", "rmp_proximity",
    "diesel_pm_proximity", "pct_ling_isolated",
    "latitude", "longitude",
    "month", "hour", "dow", "day_of_year",
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
    "temp_x_humidity", "wind_x_temp",
]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading & LOSO (cached)
# ─────────────────────────────────────────────────────────────────────────────

def log(msg): print(msg, flush=True)


def load_data():
    log("Loading p2_processed.xls...")
    df = pd.read_csv(ROOT / "p2_processed.xls", encoding="utf-8-sig", low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", TARGET])
    df["sensor_id"] = df["sensor_id"].astype(str)

    wpath = PIPELINE_DIR / "historical_weather.csv"
    if wpath.exists():
        w = pd.read_csv(wpath)
        w["date"] = pd.to_datetime(w["date"])
        w["sensor_id"] = w["sensor_id"].astype(str)
        df = df.merge(w[["sensor_id", "date", "wind_speed", "precipitation"]],
                      on=["sensor_id", "date"], how="left", suffixes=("", "_h"))
        if "wind_speed_h" in df.columns:
            df["wind_speed"] = df["wind_speed_h"].fillna(df.get("wind_speed", 0))
            df.drop(columns=["wind_speed_h"], inplace=True, errors="ignore")
        df["wind_speed"] = df["wind_speed"].fillna(0.0)
        df["precipitation"] = df["precipitation"].fillna(0.0)
    else:
        df["wind_speed"] = 0.0
        df["precipitation"] = 0.0

    df["month"] = df["date"].dt.month
    df["hour"] = df["date"].dt.hour
    df["dow"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["temp_x_humidity"] = df["temperature"] * df["humidity"] / 100.0
    df["wind_x_temp"] = df["wind_speed"] * df["temperature"] / 100.0

    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    log(f"  {len(df)} rows, {df['sensor_id'].nunique()} sensors")
    return df


def train_ensemble(X, y):
    rf = RandomForestRegressor(n_estimators=250, max_features="sqrt", max_depth=12,
                               min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    lgbm = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63,
                             max_depth=8, min_child_samples=10, subsample=0.8,
                             colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                             n_jobs=-1, random_state=42, verbose=-1)
    lgbm.fit(X, y)
    xg = xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=7,
                          min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                          reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1,
                          random_state=42, verbosity=0)
    xg.fit(X, y)
    return {"rf": rf, "lgbm": lgbm, "xgb": xg}


def run_or_load_loso(df):
    """Returns (all_ml_preds, df_eval_row_indices). Cache-aware."""
    if CACHE_PATH.exists():
        log(f"Loading cached LOSO predictions from {CACHE_PATH.name}...")
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        if cache.get("n_rows") == len(df):
            return cache["all_ml_preds"]
        log("  Cache length mismatch — re-running LOSO.")

    X = df[FEATURES].values
    y = df[TARGET].values
    sites = df["sensor_id"].unique()
    all_ml_preds = np.full(len(df), np.nan)

    log(f"\nRunning LOSO-CV over {len(sites)} sensors...")
    t0 = time.time()
    for i, site in enumerate(sites):
        mask = (df["sensor_id"] == site).values
        X_tr, y_tr = X[~mask], y[~mask]
        X_te = X[mask]
        if mask.sum() < 3:
            continue
        models = train_ensemble(X_tr, y_tr)
        val_split = min(int(len(X_tr) * 0.1), 5000)
        Xv, yv = X_tr[-val_split:], y_tr[-val_split:]
        mses = {n: mean_squared_error(yv, models[n].predict(Xv)) for n in models}
        inv = {k: 1.0 / max(v, 1e-10) for k, v in mses.items()}
        total = sum(inv.values())
        w = {k: v / total for k, v in inv.items()}
        pred = np.maximum(0.0, sum(w[n] * models[n].predict(X_te) for n in models))
        all_ml_preds[mask] = pred
        if (i + 1) % 20 == 0 or i == len(sites) - 1:
            elapsed = time.time() - t0
            eta = (len(sites) - i - 1) * (elapsed / max(i + 1, 1))
            log(f"  {i+1}/{len(sites)} sites  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"n_rows": len(df), "all_ml_preds": all_ml_preds}, f)
    log(f"  Cached → {CACHE_PATH.name}")
    return all_ml_preds


# ─────────────────────────────────────────────────────────────────────────────
# Distance helpers & IDW/kriging variants
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2.0) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def idw_correction(t_lat, t_lon, s_lats, s_lons, s_resids,
                   k=8, power=2.0, max_dist_km=150.0, exclude_idx=None):
    if exclude_idx is not None:
        keep = np.ones(len(s_lats), dtype=bool); keep[exclude_idx] = False
        sl, so, sr = s_lats[keep], s_lons[keep], s_resids[keep]
    else:
        sl, so, sr = s_lats, s_lons, s_resids
    d = haversine_km(t_lat, t_lon, sl, so)
    if k < len(d):
        idx = np.argpartition(d, k)[:k]
        d, sr = d[idx], sr[idx]
    mask = d < max_dist_km
    if not mask.any():
        return 0.0
    d, sr = d[mask], sr[mask]
    zero = d < 1e-6
    if zero.any():
        return float(sr[zero].mean())
    w = 1.0 / (d ** power)
    return float(np.sum(w * sr) / np.sum(w))


def exp_variogram_kriging(t_lat, t_lon, s_lats, s_lons, s_resids,
                          range_km=60.0, sill=None, nugget=0.1,
                          k=20, exclude_idx=None):
    """Ordinary kriging with an exponential variogram model.
    γ(h) = nugget + (sill - nugget) * (1 - exp(-3h/range))
    Uses nearest K points for tractability (local kriging).
    """
    if exclude_idx is not None:
        keep = np.ones(len(s_lats), dtype=bool); keep[exclude_idx] = False
        sl, so, sr = s_lats[keep], s_lons[keep], s_resids[keep]
    else:
        sl, so, sr = s_lats, s_lons, s_resids
    d0 = haversine_km(t_lat, t_lon, sl, so)
    if k < len(d0):
        idx = np.argpartition(d0, k)[:k]
        d0 = d0[idx]; sl = sl[idx]; so = so[idx]; sr = sr[idx]
    if len(d0) == 0 or np.all(d0 > 3 * range_km):
        return 0.0
    if sill is None:
        sill = float(np.var(sr)) + nugget

    def gamma(h):
        return nugget + (sill - nugget) * (1.0 - np.exp(-3.0 * h / max(range_km, 1.0)))

    n = len(sl)
    pair_d = np.zeros((n, n))
    for i in range(n):
        pair_d[i, :] = haversine_km(sl[i], so[i], sl, so)
    C = sill - gamma(pair_d)  # covariance
    # Ordinary kriging system with Lagrangian for weights summing to 1
    A = np.ones((n + 1, n + 1))
    A[:n, :n] = C
    A[n, n] = 0.0
    c0 = sill - gamma(d0)
    b = np.concatenate([c0, [1.0]])
    try:
        sol = np.linalg.solve(A + 1e-6 * np.eye(n + 1), b)
    except np.linalg.LinAlgError:
        return 0.0
    weights = sol[:n]
    return float(np.sum(weights * sr))


# ─────────────────────────────────────────────────────────────────────────────
# Strategy definitions + evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_strategy(name, correction_fn, df_eval, sensor_lat, sensor_lon,
                      sensor_resid, sensor_id_to_idx, ml_preds, y_true,
                      shrinkage=1.0, max_abs=None):
    """Compute hybrid LOSO predictions using a correction function and evaluate."""
    correction_by_sensor = {}
    for sid in sensor_id_to_idx.keys():
        idx = sensor_id_to_idx[sid]
        c = correction_fn(float(sensor_lat[idx]), float(sensor_lon[idx]), idx)
        correction_by_sensor[sid] = c

    sids = df_eval["sensor_id"].values
    corr = np.array([correction_by_sensor[s] for s in sids]) * shrinkage
    if max_abs is not None:
        corr = np.clip(corr, -max_abs, max_abs)
    hybrid = np.maximum(0.0, ml_preds + corr)

    r2 = r2_score(y_true, hybrid)
    rmse = float(np.sqrt(mean_squared_error(y_true, hybrid)))
    mae = float(mean_absolute_error(y_true, hybrid))
    pearson = float(np.corrcoef(ml_preds, corr)[0, 1]) if corr.std() > 1e-9 else 0.0
    return {
        "name": name,
        "r2": r2, "rmse": rmse, "mae": mae,
        "correction_std": float(corr.std()),
        "correction_range": [float(corr.min()), float(corr.max())],
        "ml_corr_pearson": pearson,
        "correction_by_sensor": correction_by_sensor,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Residual learning (ridge / RF on (lat, lon, ejf, ml_pred) → signed residual)
# ─────────────────────────────────────────────────────────────────────────────

def fit_residual_learner(feat_cols, df_train, signed_resids_train, model_type):
    X = df_train[feat_cols].values
    y = signed_resids_train
    if model_type == "ridge":
        m = Ridge(alpha=2.0, random_state=42)
    elif model_type == "rf":
        m = RandomForestRegressor(n_estimators=100, max_depth=5,
                                  min_samples_leaf=30, n_jobs=-1, random_state=42)
    else:
        raise ValueError(model_type)
    m.fit(X, y)
    return m


def evaluate_residual_learner(name, model_type, df_eval, sensor_agg,
                              ml_preds, y_true, shrinkage=1.0, max_abs=None):
    """LOSO-style residual learning: for each held-out sensor X, train a residual
    predictor on the OTHER 239 sensors' aggregated (lat, lon, ejf, pm25) → signed
    mean residual, then predict signed residual at X's location and apply."""
    feat_cols = ["latitude", "longitude", "ejf_score", "pct_people_of_color"]
    # Sensor-level aggregate: one row per sensor
    senagg = sensor_agg.copy()
    senagg = senagg.rename(columns={"lat": "latitude", "lon": "longitude"})

    correction_by_sensor = {}
    for _, row in senagg.iterrows():
        sid = row["sensor_id"]
        others = senagg[senagg["sensor_id"] != sid]
        try:
            m = fit_residual_learner(feat_cols, others, others["mean_signed_resid"].values, model_type)
            x = row[feat_cols].values.reshape(1, -1)
            correction_by_sensor[sid] = float(m.predict(x)[0])
        except Exception:
            correction_by_sensor[sid] = 0.0

    sids = df_eval["sensor_id"].values
    corr = np.array([correction_by_sensor.get(s, 0.0) for s in sids]) * shrinkage
    if max_abs is not None:
        corr = np.clip(corr, -max_abs, max_abs)
    hybrid = np.maximum(0.0, ml_preds + corr)
    r2 = r2_score(y_true, hybrid)
    rmse = float(np.sqrt(mean_squared_error(y_true, hybrid)))
    mae = float(mean_absolute_error(y_true, hybrid))
    pearson = float(np.corrcoef(ml_preds, corr)[0, 1]) if corr.std() > 1e-9 else 0.0
    return {
        "name": name,
        "r2": r2, "rmse": rmse, "mae": mae,
        "correction_std": float(corr.std()),
        "correction_range": [float(corr.min()), float(corr.max())],
        "ml_corr_pearson": pearson,
        "correction_by_sensor": correction_by_sensor,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    all_ml_preds = run_or_load_loso(df)
    valid = ~np.isnan(all_ml_preds)
    df_eval = df.loc[valid].copy()
    df_eval["ml_pred"] = all_ml_preds[valid]
    df_eval["signed_resid"] = df_eval[TARGET] - df_eval["ml_pred"]
    y_true = df_eval[TARGET].values
    ml_preds = df_eval["ml_pred"].values

    ml_r2 = r2_score(y_true, ml_preds)
    ml_rmse = float(np.sqrt(mean_squared_error(y_true, ml_preds)))
    ml_mae = float(mean_absolute_error(y_true, ml_preds))

    log("\n" + "=" * 70)
    log(f"ML-only LOSO baseline: R²={ml_r2:.4f}  RMSE={ml_rmse:.4f}  MAE={ml_mae:.4f}")
    log("=" * 70)

    # per-sensor aggregate
    sensor_agg = df_eval.groupby("sensor_id").agg(
        lat=("latitude", "first"),
        lon=("longitude", "first"),
        mean_signed_resid=("signed_resid", "mean"),
        ejf_score=("ejf_score", "first"),
        pct_people_of_color=("pct_people_of_color", "first"),
        n_obs=("signed_resid", "size"),
    ).reset_index()
    s_lat = sensor_agg["lat"].values.astype(float)
    s_lon = sensor_agg["lon"].values.astype(float)
    s_resid = sensor_agg["mean_signed_resid"].values.astype(float)
    sensor_id_to_idx = {sid: i for i, sid in enumerate(sensor_agg["sensor_id"].values)}

    results = []

    # ── IDW variants ────────────────────────────────────────────────────────
    idw_configs = [
        dict(k=5,  power=2.0, max_dist_km=40,  shrinkage=1.0, max_abs=None),
        dict(k=8,  power=2.0, max_dist_km=150, shrinkage=1.0, max_abs=None),  # original
        dict(k=8,  power=2.0, max_dist_km=150, shrinkage=0.5, max_abs=2.0),
        dict(k=8,  power=2.0, max_dist_km=150, shrinkage=0.3, max_abs=1.5),
        dict(k=20, power=1.0, max_dist_km=100, shrinkage=0.5, max_abs=2.0),
        dict(k=20, power=1.0, max_dist_km=100, shrinkage=0.3, max_abs=1.0),
        dict(k=40, power=1.0, max_dist_km=150, shrinkage=0.2, max_abs=0.8),
    ]
    for cfg in idw_configs:
        name = f"idw k={cfg['k']} p={cfg['power']} d={cfg['max_dist_km']}km s={cfg['shrinkage']} cap={cfg['max_abs']}"
        fn = lambda tlat, tlon, exclude_idx, cfg=cfg: idw_correction(
            tlat, tlon, s_lat, s_lon, s_resid,
            k=cfg["k"], power=cfg["power"], max_dist_km=cfg["max_dist_km"],
            exclude_idx=exclude_idx,
        )
        res = evaluate_strategy(name, fn, df_eval, s_lat, s_lon, s_resid,
                                sensor_id_to_idx, ml_preds, y_true,
                                shrinkage=cfg["shrinkage"], max_abs=cfg["max_abs"])
        results.append(res)
        log(f"  {name:55s}  R²={res['r2']:.4f}  (Δ={res['r2']-ml_r2:+.4f})")

    # ── Exponential kriging variants ────────────────────────────────────────
    log("\n── Ordinary kriging (exponential variogram) ──")
    krig_configs = [
        dict(range_km=30,  nugget=0.5, shrinkage=0.5, max_abs=2.0),
        dict(range_km=50,  nugget=0.5, shrinkage=0.5, max_abs=2.0),
        dict(range_km=100, nugget=1.0, shrinkage=0.5, max_abs=2.0),
        dict(range_km=50,  nugget=0.5, shrinkage=0.3, max_abs=1.5),
        dict(range_km=80,  nugget=1.0, shrinkage=0.3, max_abs=1.0),
    ]
    for cfg in krig_configs:
        name = f"krig-exp R={cfg['range_km']}km n={cfg['nugget']} s={cfg['shrinkage']} cap={cfg['max_abs']}"
        fn = lambda tlat, tlon, exclude_idx, cfg=cfg: exp_variogram_kriging(
            tlat, tlon, s_lat, s_lon, s_resid,
            range_km=cfg["range_km"], nugget=cfg["nugget"],
            k=25, exclude_idx=exclude_idx,
        )
        res = evaluate_strategy(name, fn, df_eval, s_lat, s_lon, s_resid,
                                sensor_id_to_idx, ml_preds, y_true,
                                shrinkage=cfg["shrinkage"], max_abs=cfg["max_abs"])
        results.append(res)
        log(f"  {name:55s}  R²={res['r2']:.4f}  (Δ={res['r2']-ml_r2:+.4f})")

    # ── Residual-learning approaches ────────────────────────────────────────
    log("\n── Residual learners ──")
    for model_type in ["ridge", "rf"]:
        for sh, cap in [(1.0, None), (0.5, 1.5), (0.3, 1.0)]:
            name = f"resid-{model_type} s={sh} cap={cap}"
            res = evaluate_residual_learner(name, model_type, df_eval, sensor_agg,
                                            ml_preds, y_true, shrinkage=sh, max_abs=cap)
            results.append(res)
            log(f"  {name:55s}  R²={res['r2']:.4f}  (Δ={res['r2']-ml_r2:+.4f})")

    # ── Pick best ───────────────────────────────────────────────────────────
    results.sort(key=lambda r: -r["r2"])
    best = results[0]
    log("\n" + "=" * 70)
    log("TOP 5 STRATEGIES")
    log("=" * 70)
    for r in results[:5]:
        log(f"  {r['name']:60s}  R²={r['r2']:.4f}  RMSE={r['rmse']:.4f}  Δ={r['r2']-ml_r2:+.4f}")
    log("")
    log(f"ML-only baseline:        R²={ml_r2:.4f}  RMSE={ml_rmse:.4f}")
    log(f"Best strategy ({best['name']!r}): R²={best['r2']:.4f}  RMSE={best['rmse']:.4f}")

    IMPROVEMENT_THRESHOLD = 0.01
    MAX_ML_CORR = 0.3
    delta = best["r2"] - ml_r2

    # ── Decide whether to deploy ────────────────────────────────────────────
    if delta < IMPROVEMENT_THRESHOLD:
        log(f"\nNo strategy beat ML-only by ≥{IMPROVEMENT_THRESHOLD:+.2f} R². "
            "Not saving corrections.")
        _update_metrics(ml_r2, ml_rmse, ml_mae, None, delta=delta, reason="no_improvement")
        # ensure any stale corrections file is removed
        corr_file = MODELS_DIR / "kriging_corrections.json"
        if corr_file.exists():
            corr_file.unlink()
            log(f"  Removed stale {corr_file.name}.")
        return

    if abs(best["ml_corr_pearson"]) > MAX_ML_CORR:
        log(f"\nBest strategy's correction correlates too strongly with ML prediction "
            f"(ρ={best['ml_corr_pearson']:.3f} > {MAX_ML_CORR}) — may be over-fitting. "
            "Not saving corrections.")
        _update_metrics(ml_r2, ml_rmse, ml_mae, None, delta=delta, reason="over_correction")
        return

    # ── Compute tract-level corrections using the winning strategy ──────────
    log(f"\nDeploying {best['name']!r}  (ΔR²={delta:+.4f})")
    corrections = _apply_best_to_tracts(best, sensor_agg, s_lat, s_lon, s_resid)

    corr_path = MODELS_DIR / "kriging_corrections.json"
    with open(corr_path, "w") as f:
        json.dump(corrections, f)
    corr_vals = np.array(list(corrections.values()))
    log(f"  Saved {len(corrections)} tract corrections  "
        f"(std={corr_vals.std():.3f}, range=[{corr_vals.min():.3f}, {corr_vals.max():.3f}])")

    _update_metrics(ml_r2, ml_rmse, ml_mae, best, delta=delta, reason="deployed")


def _update_metrics(ml_r2, ml_rmse, ml_mae, best, delta, reason):
    mp = MODELS_DIR / "metrics.json"
    metrics = json.load(open(mp)) if mp.exists() else {}
    # IMPORTANT: this geostats experiment trains its OWN reduced-feature RF (see
    # FEATURES at the top of this file — ~27 features, NO neighbor/HMS/AOD/
    # spatial-context features). Its LOSO numbers are NOT the deployed v6 model's.
    # Write them under a dedicated key so we never clobber 03_train_enhanced.py's
    # headline metrics["loso_cv"] / metrics["loso_cv_optimized"] (the prior bug).
    metrics["geostats_baseline"] = {
        **metrics.get("geostats_baseline", {}),
        "r2": round(ml_r2, 4), "rmse": round(ml_rmse, 4), "mae": round(ml_mae, 4),
        "note": "geostats-experiment reduced-feature RF LOSO baseline — NOT the deployed model",
    }
    if best is not None:
        metrics["loso_hybrid"] = {
            "strategy": best["name"],
            "r2": round(best["r2"], 4),
            "rmse": round(best["rmse"], 4),
            "mae": round(best["mae"], 4),
            "delta_r2": round(delta, 4),
            "delta_rmse": round(best["rmse"] - ml_rmse, 4),
            "ml_corr_pearson": round(best["ml_corr_pearson"], 4),
            "reason": reason,
        }
    else:
        metrics["loso_hybrid"] = {
            "strategy": None,
            "delta_r2": round(delta, 4),
            "reason": reason,
        }
    with open(mp, "w") as f:
        json.dump(metrics, f, indent=2)


def _apply_best_to_tracts(best, sensor_agg, s_lat, s_lon, s_resid):
    """Re-apply the best strategy to every census tract (no exclusion — we're
    predicting at tracts, not sensors, so we use all 240 sensors)."""
    tracts = pd.read_parquet(ROOT / "backend" / "static" / "tract_lookup.parquet")
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)

    name = best["name"]
    corrections = {}

    if name.startswith("idw"):
        # parse k/power/max_dist/shrinkage/max_abs from the name encoding
        params = _parse_idw_name(name)
        for _, t in tracts.iterrows():
            c = idw_correction(float(t["lat"]), float(t["lon"]), s_lat, s_lon, s_resid,
                               k=params["k"], power=params["power"],
                               max_dist_km=params["max_dist_km"], exclude_idx=None)
            c *= params["shrinkage"]
            if params["max_abs"] is not None:
                c = max(-params["max_abs"], min(params["max_abs"], c))
            corrections[t["GEOID"]] = round(float(c), 4)
        return corrections

    if name.startswith("krig-exp"):
        params = _parse_krig_name(name)
        for _, t in tracts.iterrows():
            c = exp_variogram_kriging(float(t["lat"]), float(t["lon"]),
                                      s_lat, s_lon, s_resid,
                                      range_km=params["range_km"], nugget=params["nugget"],
                                      k=25, exclude_idx=None)
            c *= params["shrinkage"]
            if params["max_abs"] is not None:
                c = max(-params["max_abs"], min(params["max_abs"], c))
            corrections[t["GEOID"]] = round(float(c), 4)
        return corrections

    if name.startswith("resid-"):
        # refit residual learner on ALL sensors, apply to every tract
        model_type = name.split()[0].split("-")[1]
        sh = float(name.split("s=")[1].split()[0])
        cap_str = name.split("cap=")[1].strip()
        max_abs = None if cap_str == "None" else float(cap_str)
        senagg = sensor_agg.rename(columns={"lat": "latitude", "lon": "longitude"})
        feat_cols = ["latitude", "longitude", "ejf_score", "pct_people_of_color"]
        m = fit_residual_learner(feat_cols, senagg, senagg["mean_signed_resid"].values, model_type)
        # Build feature matrix for tracts
        tracts_feat = pd.DataFrame({
            "latitude": tracts["lat"].astype(float),
            "longitude": tracts["lon"].astype(float),
            "ejf_score": tracts.get("ejf_score", pd.Series(0.0, index=tracts.index)).astype(float),
            "pct_people_of_color": tracts.get("pct_people_of_color", pd.Series(0.0, index=tracts.index)).astype(float),
        })
        preds = m.predict(tracts_feat[feat_cols].values) * sh
        if max_abs is not None:
            preds = np.clip(preds, -max_abs, max_abs)
        for i, t in tracts.iterrows():
            corrections[t["GEOID"]] = round(float(preds[i - tracts.index[0]] if i - tracts.index[0] < len(preds) else preds[0]), 4)
        # cleaner: position-aligned
        corrections = {g: round(float(p), 4) for g, p in zip(tracts["GEOID"].values, preds)}
        return corrections

    raise ValueError(f"unknown strategy name: {name}")


def _parse_idw_name(name):
    parts = {}
    for tok in name.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            if v == "None":
                parts[k] = None
            elif "km" in v:
                parts["max_dist_km"] = float(v.replace("km", ""))
            else:
                parts[k] = float(v) if "." in v or v.replace("-", "").isdigit() else v
    out = {
        "k": int(parts.get("k", 8)),
        "power": float(parts.get("p", parts.get("power", 2.0))),
        "max_dist_km": float(parts.get("d", parts.get("max_dist_km", 150.0))),
        "shrinkage": float(parts.get("s", parts.get("shrinkage", 1.0))),
        "max_abs": parts.get("cap") if parts.get("cap") is not None else None,
    }
    if out["max_abs"] is not None:
        out["max_abs"] = float(out["max_abs"])
    return out


def _parse_krig_name(name):
    parts = {}
    for tok in name.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            if v == "None":
                parts[k] = None
            elif "km" in v:
                parts["range_km"] = float(v.replace("km", ""))
            else:
                parts[k] = float(v)
    out = {
        "range_km": float(parts.get("R", parts.get("range_km", 50.0))),
        "nugget": float(parts.get("n", parts.get("nugget", 0.5))),
        "shrinkage": float(parts.get("s", parts.get("shrinkage", 0.5))),
        "max_abs": parts.get("cap") if parts.get("cap") is not None else None,
    }
    if out["max_abs"] is not None:
        out["max_abs"] = float(out["max_abs"])
    return out


if __name__ == "__main__":
    main()
