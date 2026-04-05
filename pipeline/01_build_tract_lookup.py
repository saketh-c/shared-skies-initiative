"""
01_build_tract_lookup.py
Merges EJScreen + TX tract centroids + population into a single lookup table.
Downloads census tract GeoJSON from TIGERweb for multiple Texas counties (Dallas, Austin, Houston, San Antonio).

Run from the project root:
    python pipeline/01_build_tract_lookup.py
"""

import os
import json
import pandas as pd
import numpy as np

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    import urllib.request
    HAS_HTTPX = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT
STATIC_DIR = os.path.join(ROOT, "backend", "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# City configuration for multi-city support
CITIES = {
    "dallas": {"state": "48", "county": "113"},
    "austin": {"state": "48", "county": "453"},
    "houston": {"state": "48", "county": "201"},
    "san_antonio": {"state": "48", "county": "029"},
}

# Major Texas counties for statewide coverage
TEXAS_COUNTIES = {
    "48113": "dallas",           # Dallas
    "48201": "houston",          # Harris (Houston)
    "48453": "austin",           # Travis (Austin)
    "48029": "san_antonio",      # Bexar (San Antonio)
    "48439": "tarrant",          # Tarrant (Fort Worth)
    "48085": "collin",           # Collin (Plano/McKinney)
    "48121": "denton",           # Denton (Denton)
    "48491": "williamson",       # Williamson (Austin area)
    "48157": "fort_bend",        # Fort Bend (Houston area)
    "48015": "brazo_county",     # Brazos (College Station)
    "48355": "nueces",           # Nueces (Corpus Christi)
    "48393": "smith",            # Smith (Tyler)
    "48141": "el_paso",          # El Paso
    "48251": "jefferson",        # Jefferson (Beaumont)
    "48167": "galveston",        # Galveston
    "48369": "polk",             # Polk
    "48309": "mclennan",         # McLennan (Waco)
    "48027": "bell",             # Bell (Killeen)
    "48303": "lubbock",          # Lubbock
}

# EJScreen column → model feature name
EJ_COLUMN_MAP = {
    "T_DEMOGIDX_2":  "ejf_score",
    "T_PEOPCOLORPCT":"pct_people_of_color",
    "T_LOWINCPCT":   "pct_low_income",
    "T_PTRAF":       "traffic_proximity",
    "T_PNPL":        "superfund_proximity",
    "T_PRMP":        "rmp_proximity",
    "T_DSLPM":       "diesel_pm_proximity",
    "T_LINGISOPCT":  "pct_ling_isolated",
}


def load_file(path: str) -> pd.DataFrame:
    """Load a .xls file that is actually a CSV (UTF-8 BOM)."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        try:
            return pd.read_excel(path, engine="xlrd")
        except Exception:
            return pd.read_excel(path, engine="openpyxl")


def normalize_geoid(val) -> str | None:
    """Convert any GEOID representation to a zero-padded 11-char string."""
    if pd.isna(val):
        return None
    try:
        return str(int(float(val))).zfill(11)
    except (ValueError, TypeError):
        return None


def build_lookup():
    # ── EJScreen ──────────────────────────────────────────────────────────────
    print("Loading ejscreendata...")
    ej = load_file(os.path.join(DATA_DIR, "ejscreendata.xls"))
    print(f"  Raw rows: {len(ej)}")

    # Filter out the ~78k empty scaffold rows
    ej = ej[ej["T_DEMOGIDX_2"].notna()].copy()
    print(f"  After filtering empty rows: {len(ej)}")

    ej["GEOID"] = ej["ID"].apply(normalize_geoid)
    ej = ej.dropna(subset=["GEOID"])

    # Keep only columns we need
    keep_cols = ["GEOID", "STATE_NAME", "CNTY_NAME"] + list(EJ_COLUMN_MAP.keys())
    keep_cols = [c for c in keep_cols if c in ej.columns]
    ej = ej[keep_cols].rename(columns=EJ_COLUMN_MAP)
    print(f"  EJScreen columns kept: {list(ej.columns)}")

    # ── Centroids ─────────────────────────────────────────────────────────────
    print("Loading tx_tract_centroids...")
    centroids = load_file(os.path.join(DATA_DIR, "tx_tract_centroids.xls"))
    print(f"  Rows: {len(centroids)}")

    centroids["GEOID"] = (
        centroids["STATEFP"].astype(str).str.zfill(2)
        + centroids["COUNTYFP"].astype(str).str.zfill(3)
        + centroids["TRACTCE"].astype(str).str.zfill(6)
    )
    centroids = centroids.rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon"})

    # ── Population ────────────────────────────────────────────────────────────
    print("Loading popdata...")
    pop = load_file(os.path.join(DATA_DIR, "popdata.xls"))
    pop = pop.dropna(subset=["ID"])
    pop["GEOID"] = pop["ID"].apply(normalize_geoid)
    pop = pop.dropna(subset=["GEOID"])[["GEOID", "population"]]
    print(f"  Population rows after cleaning: {len(pop)}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    print("Merging datasets...")
    lookup = centroids[["GEOID", "lat", "lon", "POPULATION"]].merge(
        ej, on="GEOID", how="left"
    ).merge(
        pop, on="GEOID", how="left"
    )

    # Fill nulls in EJ columns with column medians (28 tracts statewide)
    ej_feature_cols = list(EJ_COLUMN_MAP.values())
    for col in ej_feature_cols:
        if col in lookup.columns:
            median_val = lookup[col].median()
            null_count = lookup[col].isna().sum()
            if null_count > 0:
                print(f"  Filling {null_count} nulls in '{col}' with median {median_val:.1f}")
            lookup[col] = lookup[col].fillna(median_val)

    print(f"Final lookup: {len(lookup)} tracts, {len(lookup.columns)} columns")
    print(f"Dallas tracts (48113...): {lookup['GEOID'].str.startswith('48113').sum()}")
    print(f"Austin tracts (48453...): {lookup['GEOID'].str.startswith('48453').sum()}")
    print(f"Houston tracts (48201...): {lookup['GEOID'].str.startswith('48201').sum()}")
    print(f"San Antonio tracts (48029...): {lookup['GEOID'].str.startswith('48029').sum()}")

    out_path = os.path.join(STATIC_DIR, "tract_lookup.parquet")
    lookup.to_parquet(out_path, index=False)
    print(f"Saved → {out_path}")

    return lookup


def download_all_geojson():
    """Download census tract boundaries for all target counties and merge."""
    print("Downloading census tract GeoJSON from TIGERweb for Texas counties...")

    all_features = []

    for fips_code, county_name in TEXAS_COUNTIES.items():
        state_code = fips_code[:2]
        county_code = fips_code[2:]

        geojson_path = os.path.join(STATIC_DIR, f"{county_name}_tracts.geojson")
        if os.path.exists(geojson_path):
            print(f"  {county_name}: GeoJSON already exists, loading...")
            with open(geojson_path) as f:
                data = json.load(f)
                all_features.extend(data.get("features", []))
        else:
            print(f"  Downloading {county_name.replace('_', ' ').title()} County...")
            url = (
                "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
                "tigerWMS_Current/MapServer/8/query"
                f"?where=STATE%3D%27{state_code}%27+AND+COUNTY%3D%27{county_code}%27"
                "&outFields=GEOID%2CNAME%2CAREALAND"
                "&returnGeometry=true"
                "&f=geojson"
                "&outSR=4326"
            )

            try:
                if HAS_HTTPX:
                    resp = httpx.get(url, timeout=60.0)
                    resp.raise_for_status()
                    data = resp.json()
                else:
                    import urllib.request
                    with urllib.request.urlopen(url, timeout=60) as r:
                        data = json.loads(r.read().decode())

                feature_count = len(data.get("features", []))
                print(f"    Downloaded {feature_count} tract features")

                # Normalize GEOIDs and add to collection
                for feat in data.get("features", []):
                    props = feat.get("properties", {})
                    if "GEOID" in props and props["GEOID"]:
                        props["GEOID"] = str(props["GEOID"]).zfill(11)
                    all_features.append(feat)

                # Also save individual county file
                with open(geojson_path, "w") as f:
                    json.dump(data, f)
                print(f"    Saved → {geojson_path}")

            except Exception as e:
                print(f"    WARNING: Could not download {county_name}: {e}")

    # Create merged statewide GeoJSON
    statewide_path = os.path.join(STATIC_DIR, "texas_all_tracts.geojson")
    statewide_data = {
        "type": "FeatureCollection",
        "features": all_features
    }
    with open(statewide_path, "w") as f:
        json.dump(statewide_data, f)
    print(f"\n✓ Merged statewide GeoJSON: {len(all_features)} features")
    print(f"  Saved → {statewide_path}")


if __name__ == "__main__":
    lookup = build_lookup()
    download_all_geojson()
    print("\nAll done! Run pipeline/02_train_model.py next.")
