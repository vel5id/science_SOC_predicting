"""
Landsat 8 Feature Extraction
=============================
Extract Landsat 8 bands and spectral indices for each field polygon
across 4 seasonal composites.

Output: data/landsat8/l8_features_{year}_{season}.csv
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


def cloud_mask_l8(image: ee.Image) -> ee.Image:
    """Apply cloud mask to Landsat 8 using QA_PIXEL band."""
    qa = image.select("QA_PIXEL")
    
    # Bit 3: cloud, Bit 4: cloud shadow
    cloud_bit = 1 << 3
    shadow_bit = 1 << 4
    
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(shadow_bit).eq(0))
    
    return image.updateMask(mask)


def apply_l8_scale(image: ee.Image) -> ee.Image:
    """
    Apply Collection 2 Level 2 scale factors to Landsat 8 SR bands.

    Converts raw DN to surface reflectance:
        ρ = DN × 0.0000275 − 0.2

    Reference:
        USGS Landsat Collection 2 Level-2 Science Products
        https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products
    """
    sr_bands = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
    scaled = image.select(sr_bands).multiply(config.L8_SCALE_FACTOR).add(config.L8_OFFSET)
    # Clamp to valid reflectance range [0, 1]
    scaled = scaled.max(0).min(1)
    return image.addBands(scaled, overwrite=True)


def compute_l8_indices(image: ee.Image) -> ee.Image:
    """
    Compute spectral indices from Landsat 8 bands.

    Assumes bands are already in reflectance (0–1) via apply_l8_scale().
    """
    # NDVI = (NIR − RED) / (NIR + RED)
    ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")

    # GNDVI = (NIR − GREEN) / (NIR + GREEN)
    gndvi = image.normalizedDifference(["SR_B5", "SR_B3"]).rename("GNDVI")

    # SAVI = ((NIR − RED) / (NIR + RED + L)) × (1 + L)  [Huete 1988]
    L = config.SAVI_L_FACTOR
    nir = image.select("SR_B5")
    red = image.select("SR_B4")
    savi = (
        nir.subtract(red)
        .divide(nir.add(red).add(L))
        .multiply(1 + L)
        .rename("SAVI")
    )

    return image.addBands([ndvi, gndvi, savi])


@with_retry()
def extract_l8_features(
    polygon: ee.Geometry,
    start_date: str,
    end_date: str,
) -> dict[str, float] | None:
    """Extract L8 features for a single polygon and date range."""
    # Load L8 collection
    l8 = (
        ee.ImageCollection(config.L8_COLLECTION)
        .filterBounds(polygon)
        .filterDate(start_date, end_date)
        .map(cloud_mask_l8)
        .map(apply_l8_scale)
    )
    
    count = l8.size().getInfo()
    if count == 0:
        return None
    
    # Create median composite
    composite = l8.median()
    
    # Select bands
    bands_to_extract = list(config.L8_BANDS.keys())
    composite = composite.select(bands_to_extract)
    
    # Compute indices
    composite = compute_l8_indices(composite)
    
    # Extract all bands + indices
    all_bands = bands_to_extract + list(config.L8_INDICES.keys())
    
    stats = composite.select(all_bands).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon,
        scale=30,  # Landsat 8 resolution
        maxPixels=config.GEE_MAX_PIXELS,
    ).getInfo()
    
    stats["image_count"] = count
    
    return stats


def process_fields_for_season(
    fields_gdf: gpd.GeoDataFrame,
    year: int,
    season: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Process all fields for a given year and season."""
    print(f"  Processing {season} ({start_date} to {end_date})...")
    
    year_fields = fields_gdf[fields_gdf["year"] == year].copy()
    results = []
    total = len(year_fields)
    
    for idx, row in year_fields.iterrows():
        geom_json = json.loads(gpd.GeoSeries([row["geometry"]]).to_json())
        ee_geom = ee.Geometry(geom_json["features"][0]["geometry"])
        
        features = extract_l8_features(ee_geom, start_date, end_date)
        
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
            print(f"    [{len(results)}/{total}] {row['farm']}/{row['field_name']} ✗")
    
    return pd.DataFrame(results)


def main() -> None:
    """Main execution: extract L8 features for all years and seasons."""
    print("="*60)
    print("Landsat 8 Feature Extraction (4 Seasonal Composites)")
    print("="*60)
    
    authenticate_and_initialize()
    
    print("\nLoading seasonal windows...")
    seasonal_windows = load_seasonal_windows()
    
    print("Loading field polygons from database...")
    fields_gdf = get_field_polygons()
    print(f"  Found {len(fields_gdf)} unique field polygons")
    
    config.L8_DIR.mkdir(parents=True, exist_ok=True)
    
    for year in config.YEARS:
        print(f"\n── {year} ──")
        
        if year not in seasonal_windows:
            print(f"  ⚠ No seasonal windows for {year}, skipping")
            continue
        
        for season, (start_date, end_date) in seasonal_windows[year].items():
            # Check if data already exists
            out_path = config.L8_DIR / f"l8_features_{year}_{season}.csv"
            if should_skip_file(out_path):
                print(f"  ✓ Already exists: {out_path.name}")
                continue
            
            df = process_fields_for_season(
                fields_gdf, year, season, start_date, end_date
            )
            
            if not df.empty:
                df.to_csv(out_path, index=False)
                print(f"  ✓ Saved: {out_path.name} ({len(df)} fields)")
            else:
                print(f"  ⚠ No data extracted for {season}")
    
    print("\n" + "="*60)
    print("Landsat 8 extraction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
