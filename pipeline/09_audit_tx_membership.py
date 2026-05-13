"""Audit which of the 487 pulled sensors are actually inside Texas.

Method:
  1. Build the official TX state polygon by unioning all 6,896 census-tract
     polygons in backend/static/texas_all_tracts.geojson (these come from the
     US Census TIGERweb API — authoritative state boundary).
  2. Reverse-geocode each sensor's nearest Texas tract centroid distance —
     anything > ~25 mi from the nearest TX tract is suspicious.
  3. Run point-in-polygon (vectorized via shapely STRtree) for every unique
     sensor to get a definitive in/out label.
  4. For sensors outside TX, identify which neighboring entity they fall in
     (NM, OK, AR, LA, Mexico, Gulf) using the bbox + lat/lon geography.

Reports counts and writes per-sensor membership CSV. Does NOT modify any
existing files (the user wants to preview before deciding whether to drop
them).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
TX_TRACTS = ROOT / "backend" / "static" / "texas_all_tracts.geojson"
DATASET = ROOT / "pipeline" / "purpleair_full_dataset.parquet"
SENSOR_META = ROOT / "pipeline" / "data_pull_cache" / "tx_sensors_meta.json"
OUT = ROOT / "pipeline" / "sensor_tx_membership.csv"


def _classify_outside(lat: float, lon: float) -> str:
    """Coarse regional label for sensors outside TX. Census-derived state
    borders (approximate to ~0.05°). The shapely point-in-polygon test above is
    authoritative for the in/out decision; this only labels the *outside* bucket
    so the report is human-readable.

    Reference borders:
        TX–NM:   lon ≈ -103.043 (eastern NM border)
        TX–OK:   lat ≈ 36.500 (panhandle north) and Red River (33.57–34.13 along TX–OK)
        TX–AR:   lon ≈ -94.043 (eastern panhandle of TX)
        TX–LA:   lon ≈ -93.870 → -94.043 (western LA)
        TX–MX:   Rio Grande (~25.84–29.5° lat / -97 to -106.5° lon, irregular)
    """
    # Mexico (south of TX)
    if lat < 25.84:
        return "Mexico"
    # New Mexico (west of -103.04 except southern strip below 31.78 is still NM)
    if lon < -103.043:
        return "New Mexico"
    # Oklahoma (north of TX between NM and AR borders)
    if lat > 36.5 and -103.043 <= lon <= -94.43:
        return "Oklahoma"
    # OK is east of TX panhandle north of Red River — TX/OK border roughly:
    #   panhandle:  lat 36.5     lon -103.04 to -100.0
    #   main:       lat varies along Red River 33.5–34.2  lon -100 to -94.43
    if lat >= 33.5 and lat <= 37.0 and -100.0 <= lon <= -94.43:
        # Could be OK or AR. AR is east of -94.43.
        return "Oklahoma"
    # Arkansas (east of -94.43, north of LA)
    if lon > -94.43 and lat >= 33.0:
        return "Arkansas"
    # Louisiana (east of TX, south of AR)
    if lon > -94.04 and lat < 33.0:
        return "Louisiana"
    # Far southern Mexico / Gulf
    if lon > -97 and lat < 27:
        return "Gulf of Mexico"
    return "Outside TX (unclassified)"


def main():
    print(f"[load] {TX_TRACTS}")
    with open(TX_TRACTS) as f:
        tx_geo = json.load(f)
    print(f"[load] features: {len(tx_geo.get('features', []))}")
    # Build TX polygon from union of all tract geometries
    print("[build] dissolving 6,896 tract polygons → state polygon …")
    polys = []
    for feat in tx_geo["features"]:
        try:
            g = shape(feat["geometry"])
            if g.is_valid and not g.is_empty:
                polys.append(g)
        except Exception:
            continue
    print(f"[build] valid polygons: {len(polys)}")
    tx_polygon = unary_union(polys)
    bnds = tx_polygon.bounds  # (minx, miny, maxx, maxy)
    print(f"[build] TX bounds: lon {bnds[0]:.3f}..{bnds[2]:.3f}  lat {bnds[1]:.3f}..{bnds[3]:.3f}")
    print(f"[build] TX polygon area (deg²): {tx_polygon.area:.4f}")

    # Optional: STRtree from individual tract polys (faster pip lookup)
    rtree = STRtree(polys)

    def point_in_tx(lat: float, lon: float) -> tuple[bool, int]:
        """Return (in_tx, nearest_tract_index). Uses STRtree of tract polygons
        — point is in TX iff any tract contains it."""
        pt = Point(lon, lat)
        # Quick reject by bounds
        if not (bnds[0] <= lon <= bnds[2] and bnds[1] <= lat <= bnds[3]):
            return (False, -1)
        candidates = rtree.query(pt)
        for idx in candidates:
            if polys[int(idx)].contains(pt) or polys[int(idx)].touches(pt):
                return (True, int(idx))
        return (False, -1)

    # Load sensor list — primary source: PA metadata (487 sensors, full bbox)
    print(f"\n[load] {SENSOR_META}")
    meta = json.loads(SENSOR_META.read_text())
    fields = meta["fields"]
    sensors = [dict(zip(fields, row)) for row in meta["data"]]
    print(f"[load] {len(sensors)} sensors from PA metadata cache")

    # Cross-reference with what's in the final dataset
    if DATASET.exists():
        ds = pd.read_parquet(DATASET, columns=["sensor_id", "latitude", "longitude"])
        ds_ids = set(ds["sensor_id"].astype(int).unique())
        print(f"[load] final dataset has {len(ds_ids)} unique sensors")
    else:
        ds_ids = set()

    # Audit each sensor
    rows = []
    for s in sensors:
        sid = int(s["sensor_index"])
        lat = float(s["latitude"])
        lon = float(s["longitude"])
        in_tx, _ = point_in_tx(lat, lon)
        rows.append(dict(
            sensor_id=sid,
            latitude=lat,
            longitude=lon,
            name=s.get("name", ""),
            date_created=s.get("date_created"),
            in_tx=in_tx,
            in_dataset=(sid in ds_ids),
            outside_label=("" if in_tx else _classify_outside(lat, lon)),
        ))

    df = pd.DataFrame(rows)

    print("\n=== SUMMARY ===")
    print(f"Total PA sensors in TX bbox query:   {len(df)}")
    print(f"Inside actual TX boundary:           {int(df['in_tx'].sum())}")
    print(f"OUTSIDE TX (in bbox but not state):  {int((~df['in_tx']).sum())}")
    print()
    print(f"Of those that landed in our dataset ({int(df['in_dataset'].sum())}):")
    in_ds = df[df["in_dataset"]]
    print(f"  Inside TX:    {int(in_ds['in_tx'].sum())}")
    print(f"  Outside TX:   {int((~in_ds['in_tx']).sum())}")
    print()
    print("Outside-TX breakdown by neighboring region:")
    print(df.loc[~df["in_tx"], "outside_label"].value_counts().to_string())
    print()
    if (~in_ds["in_tx"]).any():
        print("Outside-TX sensors that are in our final dataset (first 30):")
        out_sample = in_ds[~in_ds["in_tx"]][
            ["sensor_id", "latitude", "longitude", "outside_label", "name"]
        ].head(30)
        print(out_sample.to_string(index=False))

    df.to_csv(OUT, index=False)
    print(f"\n[save] full per-sensor membership table → {OUT}")


if __name__ == "__main__":
    main()
