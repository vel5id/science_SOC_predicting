"""
Sentinel-1 SAR Feature Extraction
==================================
Extract Sentinel-1 SAR features (VV, VH polarizations) for each field polygon.

Output: data/sentinel1/s1_features_{year}.csv
"""
from __future__ import annotations

import json

import ee
import geopandas as gpd
import pandas as pd

from . import config
from .db_utils import get_field_polygons
from .file_utils import should_skip_file
from .gee_auth import authenticate_and_initialize, with_retry
from .s02_sentinel2 import load_seasonal_windows


@with_retry()
def extract_s1_features(
    polygon: ee.Geometry,
    start_date: str,
    end_date: str,
) -> dict[str, float] | None:
    """Extract S1 SAR features for a single polygon and date range."""
    # Load S1 collection
    s1 = (
        ee.ImageCollection(config.S1_COLLECTION)
        .filterBounds(polygon)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    
    count = s1.size().getInfo()
    if count == 0:
        return None
    
    # Create median composite
    composite = s1.median()
    
    # Select polarizations
    vv = composite.select("VV")
    vh = composite.select("VH")
    
    # Compute VV/VH ratio in dB domain (S1 GRD values are already in dB).
    # dB subtraction = logarithm of linear ratio: VV_dB - VH_dB = 10*log10(VV_lin/VH_lin)
    ratio = vv.subtract(vh).rename("VV_VH_ratio")
    
    composite = composite.addBands(ratio)
    
    # Extract features
    stats = composite.select(["VV", "VH", "VV_VH_ratio"]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon,
        scale=10,
        maxPixels=config.GEE_MAX_PIXELS,
    ).getInfo()
    
    stats["image_count"] = count
    
    return stats


def main() -> None:
    """Main execution: extract S1 features for all years.

    If no S1 data is found for the growing season, tries a full-year
    fallback (Jan 1 - Dec 31) since S1 coverage can be intermittent
    in Central Asia / Kazakhstan.
    """
    print("="*60)
    print("Sentinel-1 SAR Feature Extraction")
    print("="*60)

    authenticate_and_initialize()

    print("\nLoading seasonal windows...")
    seasonal_windows = load_seasonal_windows()

    print("Loading field polygons from database...")
    fields_gdf = get_field_polygons()
    print(f"  Found {len(fields_gdf)} unique field polygons")

    config.S1_DIR.mkdir(parents=True, exist_ok=True)

    for year in config.YEARS:
        print(f"\n── {year} ──")

        # Check if data already exists
        out_path = config.S1_DIR / f"s1_features_{year}.csv"
        if should_skip_file(out_path):
            print(f"  ✓ Already exists: {out_path.name}")
            continue

        # Determine date range: growing season first, full year as fallback
        if year in seasonal_windows:
            all_dates = [
                (start, end) for start, end in seasonal_windows[year].values()
            ]
            start_date = min(start for start, _ in all_dates)
            end_date = max(end for _, end in all_dates)
        else:
            # Fallback: use full year if no seasonal windows
            start_date = f"{year}-04-01"
            end_date = f"{year}-10-31"
            print(f"  ⚠ No seasonal windows for {year}, using default Apr-Oct")

        print(f"  Processing ({start_date} to {end_date})...")

        year_fields = fields_gdf[fields_gdf["year"] == year].copy()
        results = []
        fallback_results = []
        total = len(year_fields)
        no_data_count = 0

        for idx, row in year_fields.iterrows():
            geom_json = json.loads(gpd.GeoSeries([row["geometry"]]).to_json())
            ee_geom = ee.Geometry(geom_json["features"][0]["geometry"])

            features = extract_s1_features(ee_geom, start_date, end_date)

            # Fallback: try full year if growing season has no data
            if features is None:
                full_start = f"{year}-01-01"
                full_end = f"{year}-12-31"
                features = extract_s1_features(ee_geom, full_start, full_end)
                if features:
                    fallback_results.append(True)

            if features:
                result = {
                    "year": year,
                    "farm": row["farm"],
                    "field_name": row["field_name"],
                    "centroid_lon": row["centroid_lon"],
                    "centroid_lat": row["centroid_lat"],
                    **features,
                }
                results.append(result)
                print(f"    [{len(results)}/{total}] {row['farm']}/{row['field_name']} ✓")
            else:
                no_data_count += 1
                print(f"    [{len(results) + no_data_count}/{total}] {row['farm']}/{row['field_name']} ✗ no S1 data")

        if fallback_results:
            print(f"  Note: {len(fallback_results)} fields used full-year fallback")

        if results:
            df = pd.DataFrame(results)
            df.to_csv(out_path, index=False)
            print(f"  ✓ Saved: {out_path.name} ({len(df)} fields)")
        else:
            # Save empty CSV with header to indicate extraction was attempted
            empty_df = pd.DataFrame(columns=[
                "year", "farm", "field_name", "centroid_lon", "centroid_lat",
                "VH", "VV", "VV_VH_ratio", "image_count",
            ])
            empty_df.to_csv(out_path, index=False)
            print(f"  ⚠ No S1 data for {year} — saved empty file: {out_path.name}")

    print("\n" + "="*60)
    print("Sentinel-1 extraction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
