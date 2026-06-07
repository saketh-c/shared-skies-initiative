"""Build the hms_smoke training feature from the NOAA HMS historical archive.

For every (sensor_id, date) in the training set, computes hms_smoke =
max smoke-density ordinal (0=none, 1=Light, 2=Medium, 3=Heavy) of the NOAA HMS
smoke polygons containing that sensor on that date.

Uses the SAME point-in-polygon helpers as the live inference path
(backend/hms.py: parse_hms_zip, build_density_polygons, smoke_density_at) so the
feature is computed identically at train and serve time — no train/inference
skew. Downloads are cached to pipeline/data_pull_cache/hms/ so re-runs are free.

Output: pipeline/hms_smoke_by_sensor.parquet  [sensor_id, date, hms_smoke]
and (if --merge) writes the augmented training parquet with the hms_smoke column.

Run:
    python pipeline/10_build_hms_history.py
"""
import os
import sys
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import httpx

# Import the SHARED HMS helpers (single source of truth for train+serve).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from backend.hms import (
    HMS_BASE_URL, parse_hms_zip, build_density_polygons, smoke_density_at,
)

DATASET = os.path.join(ROOT, "pipeline", "purpleair_full_dataset.parquet")
CACHE_DIR = os.path.join(ROOT, "pipeline", "data_pull_cache", "hms")
OUT = os.path.join(ROOT, "pipeline", "hms_smoke_by_sensor.parquet")
N_WORKERS = 12


def cache_path(yyyymmdd: str) -> str:
    year = yyyymmdd[:4]
    d = os.path.join(CACHE_DIR, year)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"hms_smoke{yyyymmdd}.zip")


def download_one(yyyymmdd: str) -> bytes | None:
    """Download (or read cached) HMS zip for a date. Returns bytes or None.
    A missing/404 day is a legitimate 'no smoke analysis' -> caller treats as 0."""
    cp = cache_path(yyyymmdd)
    if os.path.exists(cp):
        try:
            with open(cp, "rb") as f:
                data = f.read()
            return data if data else None
        except Exception:
            pass
    year, month = yyyymmdd[:4], yyyymmdd[4:6]
    url = f"{HMS_BASE_URL}/{year}/{month}/hms_smoke{yyyymmdd}.zip"
    try:
        r = httpx.get(url, timeout=60.0)
        if r.status_code != 200 or not r.content:
            # cache an empty marker so re-runs don't re-fetch a known-missing day
            with open(cp, "wb") as f:
                f.write(b"")
            return None
        with open(cp, "wb") as f:
            f.write(r.content)
        return r.content
    except Exception as e:
        print(f"[hms] {yyyymmdd} download error: {e}")
        return None


def main():
    print(f"Loading {DATASET} ...")
    df = pd.read_parquet(DATASET, columns=["sensor_id", "date", "latitude", "longitude"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["sensor_id"] = df["sensor_id"].astype(str)

    # Unique sensor coords (constant per sensor) and unique dates.
    sensors = df[["sensor_id", "latitude", "longitude"]].drop_duplicates("sensor_id").reset_index(drop=True)
    s_ids = sensors["sensor_id"].values
    s_lat = sensors["latitude"].values
    s_lon = sensors["longitude"].values
    dates = sorted(df["date"].dropna().unique())
    print(f"  {len(df):,} rows, {len(s_ids)} sensors, {len(dates)} unique dates")

    # 1. Parallel-download all date zips (cached).
    date_labels = [pd.Timestamp(d).strftime("%Y%m%d") for d in dates]
    print(f"Downloading/parsing {len(date_labels)} HMS days with {N_WORKERS} workers...")
    zip_by_date: dict[str, bytes | None] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(download_one, lbl): lbl for lbl in date_labels}
        for fut in as_completed(futs):
            lbl = futs[fut]
            zip_by_date[lbl] = fut.result()
            done += 1
            if done % 200 == 0:
                print(f"  downloaded {done}/{len(date_labels)}")

    # 2. For each date, build polygons once and test all sensors.
    print("Computing point-in-polygon smoke density per (sensor, date)...")
    rows = []
    smoke_days = 0
    for i, (d, lbl) in enumerate(zip(dates, date_labels)):
        zb = zip_by_date.get(lbl)
        if not zb:
            continue  # no smoke that day -> all sensors 0 (omit; fillna(0) later)
        try:
            feats = parse_hms_zip(zb, lbl)
            polys = build_density_polygons(feats)
        except (zipfile.BadZipFile, Exception):
            polys = []
        if not polys:
            continue
        any_smoke = False
        for sid, lat, lon in zip(s_ids, s_lat, s_lon):
            v = smoke_density_at(polys, float(lon), float(lat))
            if v > 0:
                rows.append((sid, d, v))
                any_smoke = True
        if any_smoke:
            smoke_days += 1
        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{len(dates)} dates, {len(rows):,} nonzero so far")

    hms = pd.DataFrame(rows, columns=["sensor_id", "date", "hms_smoke"])
    hms["hms_smoke"] = hms["hms_smoke"].astype("int8")
    hms.to_parquet(OUT, index=False)

    print()
    print(f"Saved {OUT}: {len(hms):,} nonzero (sensor,date) rows across {smoke_days} smoke days")
    if len(hms):
        dist = hms["hms_smoke"].value_counts().sort_index()
        print(f"  nonzero tier distribution: {dict(dist)}")
        # full-coverage tier fractions (0 = the implicit rest)
        total = len(df)
        nz = len(hms)
        print(f"  rows with smoke: {nz:,}/{total:,} ({100*nz/total:.1f}%); "
              f"rest are tier 0 (no smoke)")


if __name__ == "__main__":
    main()
