"""
Simplify texas_all_tracts.geojson to reduce file size for browser rendering.
Reduces coordinate precision and removes unnecessary vertices.
"""
import os
import json
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT, "backend", "static", "texas_all_tracts.geojson")
OUTPUT_PATH = os.path.join(ROOT, "backend", "static", "texas_all_tracts.geojson")

TOLERANCE = 0.0008  # ~80m — good balance of detail vs. size


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def simplify_ring(ring, tol):
    """Simple vertex-removal simplification."""
    if len(ring) <= 4:
        return ring
    result = [ring[0]]
    for i in range(1, len(ring) - 1):
        if distance(ring[i], result[-1]) > tol:
            result.append(ring[i])
    result.append(ring[-1])
    # Ensure ring is still valid (min 4 points for polygon)
    if len(result) < 4:
        return ring
    return result


def round_ring(ring, decimals=5):
    """Round coordinates to N decimal places."""
    return [[round(c[0], decimals), round(c[1], decimals)] for c in ring]


def process_geometry(geom):
    """Simplify and round a geometry."""
    if not geom:
        return geom

    if geom["type"] == "Polygon":
        geom["coordinates"] = [
            round_ring(simplify_ring(ring, TOLERANCE))
            for ring in geom["coordinates"]
        ]
    elif geom["type"] == "MultiPolygon":
        geom["coordinates"] = [
            [round_ring(simplify_ring(ring, TOLERANCE)) for ring in polygon]
            for polygon in geom["coordinates"]
        ]
    return geom


def count_points(data):
    total = 0
    for feat in data["features"]:
        g = feat.get("geometry")
        if not g:
            continue
        if g["type"] == "Polygon":
            for ring in g["coordinates"]:
                total += len(ring)
        elif g["type"] == "MultiPolygon":
            for poly in g["coordinates"]:
                for ring in poly:
                    total += len(ring)
    return total


def main():
    print("Loading GeoJSON...")
    with open(INPUT_PATH) as f:
        data = json.load(f)

    orig_count = count_points(data)
    print(f"Original: {len(data['features'])} features, {orig_count:,} points")

    print("Simplifying geometries...")
    for feat in data["features"]:
        process_geometry(feat.get("geometry"))
        # Strip unnecessary properties — keep only GEOID and NAME
        props = feat.get("properties", {})
        feat["properties"] = {
            "GEOID": props.get("GEOID", ""),
            "NAME": props.get("NAME", ""),
        }

    new_count = count_points(data)
    print(f"Simplified: {len(data['features'])} features, {new_count:,} points")
    print(f"Removed: {orig_count - new_count:,} points ({(1 - new_count/orig_count)*100:.1f}%)")

    print("Writing output...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"File size: {file_size:.1f} MB")
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
