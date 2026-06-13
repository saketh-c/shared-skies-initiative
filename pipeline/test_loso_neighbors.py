"""Regression test for pipeline/neighbor_features.compute_neighbor_features_df.

Run:  python pipeline/test_loso_neighbors.py

Proves two things, across many random synthetic datasets:

  1. EQUIVALENCE — with pool == query, the shared function reproduces the
     ORIGINAL inline neighbor computation from 03_train_enhanced.py byte-for-byte
     (so swapping the inline block for the function does NOT change the deployed
     model's features).

  2. LOSO HONESTY — leaving sensor S out of the pool, the function's training-row
     features equal a brute-force recompute on the dataset with S's rows removed
     (so the per-fold call genuinely excludes the held-out sensor — the leakage
     fix is correct).

Only depends on numpy/pandas/scikit-learn (NOT the full training script), so it
runs without lightgbm/xgboost/catboost installed.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neighbor_features import compute_neighbor_features_df  # noqa: E402

TARGET = "pm25"


def inline_neighbors(df):
    """Faithful copy of the ORIGINAL inline algorithm in 03_train_enhanced.py
    load_data() (lines ~351-426 pre-refactor). The reference ground truth."""
    from sklearn.neighbors import BallTree
    EARTH_R_KM = 6371.0
    RADII_KM = [25.0, 50.0, 100.0]
    max_rad_rad = max(RADII_KM) / EARTH_R_KM

    nbr_mean = {r: np.full(len(df), np.nan) for r in RADII_KM}
    nbr_cnt = {r: np.zeros(len(df), dtype=np.int32) for r in RADII_KM}
    nbr_std50 = np.zeros(len(df), dtype=np.float64)

    df_idx = df.reset_index(drop=False).rename(columns={"index": "_row"})
    coords_rad = np.radians(df_idx[["latitude", "longitude"]].values)
    pm_arr = df_idx[TARGET].values
    row_arr = df_idx["_row"].values

    for _date_val, grp in df_idx.groupby("date"):
        g_idx = grp.index.values
        if len(g_idx) < 2:
            continue
        g_coords = coords_rad[g_idx]
        g_pm = pm_arr[g_idx]
        g_rows = row_arr[g_idx]
        tree = BallTree(g_coords, metric="haversine")
        ind, dist = tree.query_radius(g_coords, r=max_rad_rad, return_distance=True)
        for i in range(len(g_coords)):
            nbrs = ind[i]
            d_km = dist[i] * EARTH_R_KM
            keep = nbrs != i
            nbrs = nbrs[keep]
            d_km = d_km[keep]
            if len(nbrs) == 0:
                continue
            vals_all = g_pm[nbrs]
            for r in RADII_KM:
                m = d_km <= r
                if not m.any():
                    continue
                vr = vals_all[m]
                nbr_mean[r][g_rows[i]] = vr.mean()
                nbr_cnt[r][g_rows[i]] = len(vr)
                if r == 50.0 and len(vr) > 1:
                    nbr_std50[g_rows[i]] = vr.std()

    s = df.reset_index(drop=True)
    grp_sum = s.groupby("date")[TARGET].transform("sum")
    grp_cnt = s.groupby("date")[TARGET].transform("count")
    loo_day_mean = (grp_sum - s[TARGET]) / (grp_cnt - 1).clip(lower=1)

    m50 = pd.Series(nbr_mean[50.0]).fillna(loo_day_mean).fillna(s[TARGET].mean()).to_numpy()
    m25 = pd.Series(nbr_mean[25.0]).fillna(pd.Series(m50)).to_numpy()
    m100 = pd.Series(nbr_mean[100.0]).fillna(pd.Series(m50)).to_numpy()
    return {
        "nbr_pm25_25km": m25,
        "nbr_count_25km": nbr_cnt[25.0],
        "nbr_pm25_50km": m50,
        "nbr_count_50km": nbr_cnt[50.0],
        "nbr_std_50km": nbr_std50,
        "nbr_pm25_100km": m100,
        "nbr_count_100km": nbr_cnt[100.0],
    }


def make_synthetic(seed, n_sensors=9, n_days=40):
    """One row per (sensor, date), with random missing days so day-groups vary
    in size (exercises the zero/one-neighbor fallback). Sensors are spread so
    that 25/50/100 km radii all see different neighbor sets, plus one far-away
    sensor (isolated => fallback) flagged out-of-Texas."""
    rng = np.random.default_rng(seed)
    base_lat, base_lon = 30.0, -97.0
    sensors = []
    for k in range(n_sensors - 1):
        # clustered within ~0-160 km of the base point
        sensors.append((
            f"s{k}",
            base_lat + rng.uniform(-0.8, 0.8),   # ~+-90 km
            base_lon + rng.uniform(-0.8, 0.8),
            True,
        ))
    # one isolated, out-of-Texas sensor far to the NW (>200 km from the cluster)
    sensors.append((f"s{n_sensors-1}", 34.5, -103.0, False))

    base_dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for sid, lat, lon, in_tx in sensors:
        for d in base_dates:
            if rng.random() < 0.25:   # ~25% missing days per sensor
                continue
            rows.append({
                "sensor_id": sid,
                "latitude": lat,
                "longitude": lon,
                "date": pd.Timestamp(d),
                TARGET: float(rng.uniform(0.5, 40.0)),
                "in_tx": in_tx,
            })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def _assert_equal(a, b, label):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True):
        bad = np.where(~np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True))[0]
        raise AssertionError(
            f"{label}: mismatch at {len(bad)} row(s); first few: "
            f"idx={bad[:5].tolist()} fn={a[bad[:5]].tolist()} ref={b[bad[:5]].tolist()}"
        )


COLS = ["nbr_pm25_25km", "nbr_count_25km", "nbr_pm25_50km", "nbr_count_50km",
        "nbr_std_50km", "nbr_pm25_100km", "nbr_count_100km"]


def test_equivalence(seeds):
    for seed in seeds:
        df = make_synthetic(seed)
        ref = inline_neighbors(df)
        got = compute_neighbor_features_df(df, df, target_col=TARGET)
        for c in COLS:
            _assert_equal(got[c], ref[c], f"[equiv seed={seed}] {c}")
    print(f"PASS  equivalence (pool==query reproduces inline) over {len(seeds)} datasets")


def test_loso_honest(seeds):
    n_checks = 0
    for seed in seeds:
        df = make_synthetic(seed)
        tx_sensors = sorted(df.loc[df["in_tx"], "sensor_id"].unique())
        for S in tx_sensors:
            pool_df = df[df["sensor_id"] != S]
            train_df = df[(df["sensor_id"] != S) & df["in_tx"]]
            if len(train_df) == 0:
                continue
            # brute force: recompute on the dataset with S removed, then pull the
            # training rows back out (matched by a stable row id).
            sub = df[df["sensor_id"] != S].reset_index(drop=False).rename(columns={"index": "_gid"})
            ref_all = inline_neighbors(sub)
            train_gids = train_df.index.to_numpy()
            sub_pos_of_gid = {int(g): i for i, g in enumerate(sub["_gid"].to_numpy())}
            sel = np.array([sub_pos_of_gid[int(g)] for g in train_gids])

            got = compute_neighbor_features_df(train_df, pool_df, target_col=TARGET)
            for c in COLS:
                _assert_equal(got[c], np.asarray(ref_all[c])[sel],
                              f"[loso seed={seed} holdout={S}] {c}")
            n_checks += 1
    print(f"PASS  LOSO honesty (leave-S-out == brute force) over {n_checks} folds")


if __name__ == "__main__":
    seeds = list(range(1, 13))
    test_equivalence(seeds)
    test_loso_honest(seeds)
    print("\nALL TESTS PASSED")
