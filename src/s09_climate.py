"""
ERA5-Land Climate Covariates Extraction
========================================
Extract climate covariates (MAT, MAP, growing-season averages) for each field.

Output: data/climate/climate_features_{year}.csv
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


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert Kelvin to Celsius."""
    return kelvin - 273.15


@with_retry()
def _reduce_with_fallback(
    image: ee.Image,
    polygon: ee.Geometry,
    band_name: str,
    scale: int = 11132,
) -> float | None:
    """
    Extract a value from an image, falling back to centroid point
    if polygon-level reduction returns None.

    ERA5-Land has ~11km resolution. Small field polygons can fall between
    grid cell centers, causing reduceRegion to return None. Using the
    centroid as a point geometry with a larger scale fixes this.
    """
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon,
        scale=scale,
        maxPixels=config.GEE_MAX_PIXELS,
    ).getInfo()

    val = stats.get(band_name)
    if val is not None:
        return val

    # Fallback: use centroid point with larger scale
    centroid = polygon.centroid(maxError=100)
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=centroid,
        scale=scale * 2,
        maxPixels=config.GEE_MAX_PIXELS,
    ).getInfo()

    return stats.get(band_name)


def extract_climate_features(
    polygon: ee.Geometry,
    year: int,
    growing_season_start: str,
    growing_season_end: str,
) -> dict[str, float]:
    """
    Extract climate features for a single polygon and year.

    Uses centroid fallback if polygon reduction returns None (common with
    ERA5's coarse ~11km resolution and small field polygons).

    Args:
        polygon: Field polygon geometry
        year: Year (e.g., 2020)
        growing_season_start: Start date of growing season (YYYY-MM-DD)
        growing_season_end: End date of growing season (YYYY-MM-DD)

    Returns:
        Dictionary with MAT, MAP, GS_temp, GS_precip
    """
    # Load ERA5-Land monthly aggregated data
    era5 = ee.ImageCollection(config.ERA5_COLLECTION).filterBounds(polygon)

    # Annual data (full year)
    annual = era5.filterDate(f"{year}-01-01", f"{year + 1}-01-01")

    # Growing season data
    growing = era5.filterDate(growing_season_start, growing_season_end)

    # Mean Annual Temperature (MAT)
    mat_img = annual.select("temperature_2m").mean()
    mat_k = _reduce_with_fallback(mat_img, polygon, "temperature_2m")
    mat_c = kelvin_to_celsius(mat_k) if mat_k is not None else None

    # Mean Annual Precipitation (MAP)
    map_img = annual.select("total_precipitation_sum").sum()
    map_m = _reduce_with_fallback(map_img, polygon, "total_precipitation_sum")
    # ERA5 precipitation is in meters, convert to mm
    map_mm = map_m * 1000 if map_m is not None else None

    # Growing Season Temperature (mean)
    gs_temp_img = growing.select("temperature_2m").mean()
    gs_temp_k = _reduce_with_fallback(gs_temp_img, polygon, "temperature_2m")
    gs_temp_c = kelvin_to_celsius(gs_temp_k) if gs_temp_k is not None else None

    # Growing Season Precipitation (total)
    gs_precip_img = growing.select("total_precipitation_sum").sum()
    gs_precip_m = _reduce_with_fallback(gs_precip_img, polygon, "total_precipitation_sum")
    gs_precip_mm = gs_precip_m * 1000 if gs_precip_m is not None else None

    return {
        "MAT": round(mat_c, 2) if mat_c is not None else None,
        "MAP": round(map_mm, 1) if map_mm is not None else None,
        "GS_temp": round(gs_temp_c, 2) if gs_temp_c is not None else None,
        "GS_precip": round(gs_precip_mm, 1) if gs_precip_mm is not None else None,
    }


def main() -> None:
    """Main execution: extract climate features for all years."""
    print("="*60)
    print("ERA5-Land Climate Covariates Extraction")
    print("="*60)
    
    authenticate_and_initialize()
    
    print("\nLoading seasonal windows...")
    seasonal_windows = load_seasonal_windows()
    
    print("Loading field polygons from database...")
    fields_gdf = get_field_polygons()
    print(f"  Found {len(fields_gdf)} unique field polygons")
    
    config.CLIMATE_DIR.mkdir(parents=True, exist_ok=True)
    
    for year in config.YEARS:
        print(f"\n── {year} ──")
        
        # Check if data already exists
        out_path = config.CLIMATE_DIR / f"climate_features_{year}.csv"
        if should_skip_file(out_path):
            print(f"  ✓ Already exists: {out_path.name}")
            continue
        
        if year not in seasonal_windows:
            print(f"  ⚠ No seasonal windows for {year}, skipping")
            continue
        
        # Use full growing season (all seasons combined)
        all_dates = [
            (start, end) for start, end in seasonal_windows[year].values()
        ]
        start_date = min(start for start, _ in all_dates)
        end_date = max(end for _, end in all_dates)
        
        print(f"  Processing full growing season ({start_date} to {end_date})...")
        
        year_fields = fields_gdf[fields_gdf["year"] == year].copy()
        results = []
        total = len(year_fields)
        
        for idx, row in year_fields.iterrows():
            geom_json = json.loads(gpd.GeoSeries([row["geometry"]]).to_json())
            ee_geom = ee.Geometry(geom_json["features"][0]["geometry"])
            
            features = extract_climate_features(ee_geom, year, start_date, end_date)
            
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
        
        if results:
            df = pd.DataFrame(results)
            df.to_csv(out_path, index=False)
            print(f"  ✓ Saved: {out_path.name} ({len(df)} fields)")
        else:
            print(f"  ⚠ No data extracted for {year}")
    
    print("\n" + "="*60)
    print("Climate extraction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
