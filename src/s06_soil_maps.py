"""
SoilGrids 250m v2.0 — Soil Texture & Properties Extraction
============================================================
Extract sand/silt/clay fractions and other soil properties from
ISRIC SoilGrids hosted on Google Earth Engine.

SoilGrids v2.0 provides global predictions at 250m resolution for
standard depths (0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm).

We extract the primary topsoil depth (0-5cm) plus a weighted average
over 0-30cm to approximate typical field sampling depth (0-20 or 0-30cm).

Reference:
    Poggio, L., et al. (2021). SoilGrids 2.0: producing soil information
    for the globe with quantified spatial uncertainty. SOIL, 7, 217-240.
    doi:10.5194/soil-7-217-2021

Output: data/soil_maps/soilgrids_features.csv

Units in output (already converted from raw):
    sand_pct, silt_pct, clay_pct   — % (0-100)
    soc_gkg                        — g/kg
    ph_h2o                         — pH units
    cec                            — mmol(c)/kg (no conversion needed)
    bdod_gcm3                      — g/cm3
    nitrogen_gkg                   — g/kg
"""
from __future__ import annotations

import json
import time

import ee
import geopandas as gpd
import pandas as pd

from . import config
from .db_utils import get_field_polygons
from .file_utils import should_skip_file
from .gee_auth import authenticate_and_initialize, with_retry

# ── Unit conversion factors ──────────────────────────────────────
# SoilGrids stores data in integer-friendly units to save space.
# We convert to standard scientific units on extraction.
UNIT_CONVERSIONS = {
    "sand": 0.1,       # g/kg  -> %
    "silt": 0.1,       # g/kg  -> %
    "clay": 0.1,       # g/kg  -> %
    "soc": 0.1,        # dg/kg -> g/kg
    "ph": 0.1,         # pH*10 -> pH
    "cec": 1.0,        # mmol(c)/kg (no conversion)
    "bdod": 0.01,      # cg/cm3 -> g/cm3
    "nitrogen": 0.01,  # cg/kg -> g/kg
}

# Column names after conversion (more descriptive for the final dataset)
OUTPUT_NAMES = {
    "sand": "sand_pct",
    "silt": "silt_pct",
    "clay": "clay_pct",
    "soc": "soc_gkg",
    "ph": "ph_h2o",
    "cec": "cec_mmol",
    "bdod": "bdod_gcm3",
    "nitrogen": "nitrogen_gkg",
}

# Depth layer weights for 0-30cm weighted average
# Standard depths: 0-5 (5cm), 5-15 (10cm), 15-30 (15cm) = total 30cm
DEPTH_WEIGHTS_0_30CM = {
    "0-5cm_mean": 5 / 30,     # 5cm out of 30cm
    "5-15cm_mean": 10 / 30,   # 10cm out of 30cm
    "15-30cm_mean": 15 / 30,  # 15cm out of 30cm
}


@with_retry()
def _reduce_soilgrids_band(
    image: ee.Image,
    polygon: ee.Geometry,
    band_name: str,
) -> float | None:
    """
    Extract a single SoilGrids band value for a polygon.

    Falls back to centroid if polygon reduction returns None
    (e.g., when the polygon is smaller than a 250m pixel).
    """
    stats = image.select(band_name).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon,
        scale=config.SOILGRIDS_SCALE,
        maxPixels=config.GEE_MAX_PIXELS,
    ).getInfo()

    val = stats.get(band_name)
    if val is not None:
        return val

    # Fallback: centroid with larger scale
    centroid = polygon.centroid(maxError=100)
    stats = image.select(band_name).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=centroid,
        scale=config.SOILGRIDS_SCALE * 2,
        maxPixels=config.GEE_MAX_PIXELS,
    ).getInfo()

    return stats.get(band_name)


def extract_soilgrids_features(
    polygon: ee.Geometry,
    properties: list[str] | None = None,
) -> dict[str, float | None]:
    """
    Extract SoilGrids features for a single polygon.

    Extracts:
        1. Topsoil (0-5cm) values for each property
        2. Weighted 0-30cm average for each property

    Args:
        polygon: Field polygon geometry (ee.Geometry)
        properties: List of property keys from config.SOILGRIDS.
                    Defaults to ["sand", "silt", "clay", "soc", "ph",
                    "cec", "bdod", "nitrogen"]

    Returns:
        Dictionary with converted feature values:
            {
                "sand_pct_0_5cm": ..., "sand_pct_0_30cm": ...,
                "silt_pct_0_5cm": ..., "silt_pct_0_30cm": ...,
                ...
            }
    """
    if properties is None:
        properties = list(config.SOILGRIDS.keys())

    result: dict[str, float | None] = {}

    for prop in properties:
        asset_id = config.SOILGRIDS.get(prop)
        if asset_id is None:
            continue

        conversion = UNIT_CONVERSIONS.get(prop, 1.0)
        out_name = OUTPUT_NAMES.get(prop, prop)

        try:
            image = ee.Image(asset_id)
        except Exception as e:
            # Asset may not be accessible — skip gracefully
            print(f"      WARNING: Cannot load {asset_id}: {e}")
            result[f"{out_name}_0_5cm"] = None
            result[f"{out_name}_0_30cm"] = None
            continue

        # ── 1. Primary depth: 0-5cm ──
        primary_band = config.SOILGRIDS_PRIMARY_DEPTH
        raw_val = _reduce_soilgrids_band(image, polygon, primary_band)
        result[f"{out_name}_0_5cm"] = (
            round(raw_val * conversion, 2) if raw_val is not None else None
        )

        # ── 2. Weighted average 0-30cm ──
        weighted_sum = 0.0
        total_weight = 0.0
        for depth_band, weight in DEPTH_WEIGHTS_0_30CM.items():
            depth_val = _reduce_soilgrids_band(image, polygon, depth_band)
            if depth_val is not None:
                weighted_sum += depth_val * weight
                total_weight += weight

        if total_weight >= 0.5:
            # At least 50% of depths are available — safe to label as 0-30cm
            avg_val = weighted_sum / total_weight
            result[f"{out_name}_0_30cm"] = round(avg_val * conversion, 2)
        else:
            # Not enough depth layers to call this a 0-30cm average
            result[f"{out_name}_0_30cm"] = None

    return result


def main() -> None:
    """Main execution: extract SoilGrids features for all unique fields."""
    print("=" * 60)
    print("SoilGrids 250m v2.0 — Soil Texture & Properties Extraction")
    print("=" * 60)

    authenticate_and_initialize()

    print("\nLoading field polygons from database...")
    fields_gdf = get_field_polygons()
    print(f"  Found {len(fields_gdf)} unique field polygons")

    # SoilGrids data is static (not temporal), so we extract per
    # unique (farm, field_name) geometry, then replicate across years.
    unique_fields = (
        fields_gdf
        .drop_duplicates(subset=["farm", "field_name"])
        .reset_index(drop=True)
    )
    print(f"  Unique fields (ignoring year): {len(unique_fields)}")

    config.SOIL_DIR.mkdir(parents=True, exist_ok=True)

    out_path = config.SOIL_DIR / "soilgrids_features.csv"
    if should_skip_file(out_path):
        print(f"\n  Already exists: {out_path.name}")
        print("  SoilGrids data is static. Delete the file to re-extract.")
        print("\n" + "=" * 60)
        print("SoilGrids extraction complete!")
        print("=" * 60)
        return

    # ── Properties to extract ──
    # Focus on texture (sand/silt/clay) as primary target,
    # plus soc, ph, cec, bdod, nitrogen as auxiliary
    target_properties = ["sand", "silt", "clay", "soc", "ph", "cec", "bdod", "nitrogen"]
    available = [p for p in target_properties if p in config.SOILGRIDS]
    print(f"\n  Properties to extract: {available}")
    print(f"  Depths: 0-5cm (primary) + 0-30cm weighted average")

    results = []
    total = len(unique_fields)
    errors = 0

    for idx, row in unique_fields.iterrows():
        geom_json = json.loads(gpd.GeoSeries([row["geometry"]]).to_json())
        ee_geom = ee.Geometry(geom_json["features"][0]["geometry"])

        try:
            features = extract_soilgrids_features(ee_geom, available)

            result = {
                "farm": row["farm"],
                "field_name": row["field_name"],
                "centroid_lon": row["centroid_lon"],
                "centroid_lat": row["centroid_lat"],
                **features,
            }
            results.append(result)

            # Progress
            progress = len(results)
            non_null = sum(1 for v in features.values() if v is not None)
            total_feats = len(features)
            print(
                f"    [{progress}/{total}] {row['farm']}/{row['field_name']} "
                f"  {non_null}/{total_feats} features extracted"
            )

        except Exception as e:
            errors += 1
            print(f"    [{len(results) + errors}/{total}] "
                  f"{row['farm']}/{row['field_name']} ERROR: {e}")

            # Still record the row with nulls
            result = {
                "farm": row["farm"],
                "field_name": row["field_name"],
                "centroid_lon": row["centroid_lon"],
                "centroid_lat": row["centroid_lat"],
            }
            results.append(result)

        # Rate limiting: GEE can throttle rapid requests
        if (len(results) + errors) % 20 == 0:
            time.sleep(1)

    if results:
        df = pd.DataFrame(results)

        # ── Validation: sand + silt + clay should sum to ~100% ──
        texture_cols = [c for c in df.columns if c.endswith("_0_5cm") and
                        any(c.startswith(t) for t in ["sand_pct", "silt_pct", "clay_pct"])]
        if len(texture_cols) == 3:
            texture_sum = df[texture_cols].sum(axis=1)
            valid = texture_sum.between(95, 105)
            print(f"\n  Texture validation (0-5cm): "
                  f"{valid.sum()}/{len(df)} fields sum to 95-105%")

        df.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path.name} ({len(df)} fields, "
              f"{len(df.columns)} columns)")
    else:
        print("\n  WARNING: No data extracted")

    if errors > 0:
        print(f"\n  Errors: {errors} fields had extraction failures")

    print("\n" + "=" * 60)
    print("SoilGrids extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
