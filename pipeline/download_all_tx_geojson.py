"""
Download ALL Texas census tract GeoJSON polygons from TIGERweb.
Covers all 254 counties and ~6,896 census tracts.
Saves merged result to backend/static/texas_all_tracts.geojson
"""
import os
import json
import time
import pandas as pd

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    import urllib.request
    HAS_HTTPX = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT, "backend", "static")
LOOKUP_PATH = os.path.join(STATIC_DIR, "tract_lookup.parquet")
OUTPUT_PATH = os.path.join(STATIC_DIR, "texas_all_tracts.geojson")

BASE_URL = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
    "tigerWMS_Current/MapServer/8/query"
)


def fetch_county_geojson(state_code, county_code):
    """Download GeoJSON for a single county."""
    url = (
        f"{BASE_URL}"
        f"?where=STATE%3D%27{state_code}%27+AND+COUNTY%3D%27{county_code}%27"
        f"&outFields=GEOID%2CNAME%2CAREALAND"
        f"&returnGeometry=true"
        f"&f=geojson"
        f"&outSR=4326"
    )

    if HAS_HTTPX:
        resp = httpx.get(url, timeout=60.0)
        resp.raise_for_status()
        return resp.json()
    else:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read().decode())


def main():
    # Get all unique county FIPS from the lookup table
    lookup = pd.read_parquet(LOOKUP_PATH)
    county_fips = sorted(lookup["GEOID"].str[:5].unique())
    print(f"Found {len(county_fips)} Texas counties to download")
    print(f"Expected ~{len(lookup)} census tracts total\n")

    all_features = []
    failed_counties = []
    downloaded_geoids = set()

    for i, fips in enumerate(county_fips):
        state_code = fips[:2]
        county_code = fips[2:]
        pct = (i + 1) / len(county_fips) * 100

        try:
            data = fetch_county_geojson(state_code, county_code)
            features = data.get("features", [])

            # Normalize GEOIDs
            for feat in features:
                props = feat.get("properties", {})
                if "GEOID" in props and props["GEOID"]:
                    props["GEOID"] = str(props["GEOID"]).zfill(11)
                    downloaded_geoids.add(props["GEOID"])
                all_features.append(feat)

            print(f"  [{i+1:3d}/{len(county_fips)}] {pct:5.1f}%  County {fips}: {len(features)} tracts  (total: {len(all_features)})")

            # Be polite to the API - small delay between requests
            if i < len(county_fips) - 1:
                time.sleep(0.15)

        except Exception as e:
            print(f"  [{i+1:3d}/{len(county_fips)}] {pct:5.1f}%  County {fips}: FAILED - {e}")
            failed_counties.append(fips)

    # Save merged GeoJSON
    statewide = {
        "type": "FeatureCollection",
        "features": all_features
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(statewide, f)

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Total features: {len(all_features)}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Saved to: {OUTPUT_PATH}")

    if failed_counties:
        print(f"\n  Failed counties ({len(failed_counties)}):")
        for fc in failed_counties:
            print(f"    - {fc}")

    # Check coverage
    all_geoids = set(lookup["GEOID"].values)
    missing = all_geoids - downloaded_geoids
    if missing:
        print(f"\n  Missing tracts (no polygon): {len(missing)}")
    else:
        print(f"\n  Full coverage: all {len(all_geoids)} tracts have polygons!")


if __name__ == "__main__":
    main()
