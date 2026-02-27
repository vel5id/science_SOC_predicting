"""
Sentinel-2 Feature Extraction
==============================
Extract Sentinel-2 bands and spectral indices for each field polygon
across 4 seasonal composites (spring, summer, late_summer, autumn).

Output: data/sentinel2/s2_features_{year}_{season}.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd

from . import config
from .db_utils import get_field_polygons
from .file_utils import should_skip_file
from .gee_auth import authenticate_and_initialize, with_retry


def load_seasonal_windows() -> dict[int, dict[str, tuple[str, str]]]:
    """
    Load seasonal windows from temperature extraction output.
    
    Returns:
        {year: {season: (start_date, end_date)}}
    """
    windows_path = config.TEMP_DIR / "seasonal_windows.txt"
    if not windows_path.exists():
        raise FileNotFoundError(
            f"Seasonal windows not found: {windows_path}\n"
            "Run s01_temperature.py first!"
        )
    
    # Parse the text file
    windows = {}
    current_year = None
    
    with open(windows_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.endswith(":"):
                current_year = int(line[:-1])
                windows[current_year] = {}
            elif ":" in line and current_year is not None:
                # Parse "  spring      : 2020-04-01 to 2020-06-01"
                parts = line.split(":")
                season = parts[0].strip()
                date_range = parts[1].strip()
                start, end = date_range.split(" to ")
                windows[current_year][season] = (start.strip(), end.strip())
    
    return windows


def cloud_mask_s2(image: ee.Image) -> ee.Image:
    """Apply cloud mask to Sentinel-2 image using SCL band."""
    scl = image.select("SCL")
    
    # SCL values: 3=cloud shadow, 8=cloud medium prob, 9=cloud high prob, 10=thin cirrus
    cloud_free = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    
    return image.updateMask(cloud_free)


def compute_s2_indices(image: ee.Image) -> ee.Image:
    """Compute spectral indices from Sentinel-2 bands."""
    indices = {}
    
    for idx_name, formula in config.S2_INDICES.items():
        # Replace band names with actual band selections
        expr = formula
        for band_name in config.S2_BANDS.keys():
            expr = expr.replace(band_name, f"image.select('{band_name}')")
        
        # Evaluate expression
        if idx_name == "NDVI":
            indices[idx_name] = image.normalizedDifference(["B8", "B4"]).rename(idx_name)
        elif idx_name == "NDRE":
            indices[idx_name] = image.normalizedDifference(["B8", "B5"]).rename(idx_name)
        elif idx_name == "GNDVI":
            indices[idx_name] = image.normalizedDifference(["B8", "B3"]).rename(idx_name)
        elif idx_name == "Cl_Red_Edge":
            indices[idx_name] = (
                image.select("B7").divide(image.select("B5")).subtract(1).rename(idx_name)
            )
        elif idx_name == "SAVI":
            # SAVI = ((NIR − RED) / (NIR + RED + L)) × (1 + L) [Huete, 1988]
            # S2 DN range is 0–10000, so L must be scaled: L_scaled = L * 10000
            L = config.SAVI_L_FACTOR
            L_scaled = L * 10000  # scale L from reflectance [0,1] to DN [0,10000]
            nir = image.select("B8")
            red = image.select("B4")
            indices[idx_name] = (
                nir.subtract(red)
                .divide(nir.add(red).add(L_scaled))
                .multiply(1 + L)
                .rename(idx_name)
            )
        elif idx_name == "EVI":
            # EVI constants (C1=6, C2=7.5, L=1) calibrated for reflectance 0–1.
            # S2 DN values are 0–10000; L must be scaled to L_evi=10000 for correct
            # denominator weighting (without scaling, L=1 is negligible vs ~3000 DN).
            nir = image.select("B8")
            red = image.select("B4")
            blue = image.select("B2")
            L_evi = 10000  # L=1 scaled to DN range
            indices[idx_name] = (
                nir.subtract(red)
                .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(L_evi))
                .multiply(2.5)
                .rename(idx_name)
            )
        elif idx_name == "BSI":
            swir = image.select("B11")
            red = image.select("B4")
            nir = image.select("B8")
            blue = image.select("B2")
            indices[idx_name] = (
                swir.add(red).subtract(nir.add(blue))
                .divide(swir.add(red).add(nir.add(blue)))
                .rename(idx_name)
            )
    
    # Add all indices to image
    for idx_img in indices.values():
        image = image.addBands(idx_img)
    
    return image


@with_retry()
def extract_s2_features(
    polygon: ee.Geometry,
    start_date: str,
    end_date: str,
) -> dict[str, float] | None:
    """
    Extract S2 features for a single polygon and date range.
    
    Args:
        polygon: Field polygon geometry
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Dictionary of band/index values, or None if no data
    """
    # Load S2 collection
    s2 = (
        ee.ImageCollection(config.S2_COLLECTION)
        .filterBounds(polygon)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config.S2_CLOUD_THRESHOLD))
        .map(cloud_mask_s2)
    )
    
    # Check if any images available
    count = s2.size().getInfo()
    if count == 0:
        return None
    
    # Create median composite
    composite = s2.median()
    
    # Select bands
    bands_to_extract = list(config.S2_BANDS.keys())
    composite = composite.select(bands_to_extract)
    
    # Compute indices
    composite = compute_s2_indices(composite)
    
    # Extract all bands + indices
    all_bands = bands_to_extract + list(config.S2_INDICES.keys())
    
    stats = composite.select(all_bands).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon,
        scale=config.GEE_SCALE,
        maxPixels=config.GEE_MAX_PIXELS,
    ).getInfo()
    
    # Add image count
    stats["image_count"] = count
    
    return stats


def process_fields_for_season(
    fields_gdf: gpd.GeoDataFrame,
    year: int,
    season: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Process all fields for a given year and season.
    
    Args:
        fields_gdf: GeoDataFrame of field polygons
        year: Year
        season: Season name
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with extracted features
    """
    print(f"  Processing {season} ({start_date} to {end_date})...")
    
    # Filter fields for this year
    year_fields = fields_gdf[fields_gdf["year"] == year].copy()
    
    results = []
    total = len(year_fields)
    
    for idx, row in year_fields.iterrows():
        # Convert shapely geometry to ee.Geometry
        geom_json = json.loads(gpd.GeoSeries([row["geometry"]]).to_json())
        ee_geom = ee.Geometry(geom_json["features"][0]["geometry"])
        
        # Extract features
        features = extract_s2_features(ee_geom, start_date, end_date)
        
        if features:
            result = {
                "year": year,
                "season": season,
                "farm": row["farm"],
                "field_name": row["field_name"],
                "centroid_lon": row["centroid_lon"],
                "centroid_lat": row["centroid_lat"],
                **features,
            }
            results.append(result)
            print(f"    [{len(results)}/{total}] {row['farm']}/{row['field_name']} ✓")
        else:
            print(f"    [{len(results)}/{total}] {row['farm']}/{row['field_name']} ✗ (no data)")
    
    return pd.DataFrame(results)


def main() -> None:
    """Main execution: extract S2 features for all years and seasons."""
    print("="*60)
    print("Sentinel-2 Feature Extraction (4 Seasonal Composites)")
    print("="*60)
    
    # Initialize GEE
    authenticate_and_initialize()
    
    # Load seasonal windows
    print("\nLoading seasonal windows...")
    seasonal_windows = load_seasonal_windows()
    
    # Get field polygons
    print("Loading field polygons from database...")
    fields_gdf = get_field_polygons()
    print(f"  Found {len(fields_gdf)} unique field polygons")
    
    # Create output directory
    config.S2_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each year and season
    for year in config.YEARS:
        print(f"\n── {year} ──")
        
        if year not in seasonal_windows:
            print(f"  ⚠ No seasonal windows for {year}, skipping")
            continue
        
        for season, (start_date, end_date) in seasonal_windows[year].items():
            # Check if data already exists
            out_path = config.S2_DIR / f"s2_features_{year}_{season}.csv"
            if should_skip_file(out_path):
                print(f"  ✓ Already exists: {out_path.name}")
                continue
            
            # Extract features
            df = process_fields_for_season(
                fields_gdf, year, season, start_date, end_date
            )
            
            if not df.empty:
                # Save to CSV
                df.to_csv(out_path, index=False)
                print(f"  ✓ Saved: {out_path.name} ({len(df)} fields)")
            else:
                print(f"  ⚠ No data extracted for {season}")
    
    print("\n" + "="*60)
    print("Sentinel-2 extraction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
