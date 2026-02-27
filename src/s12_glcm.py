"""
GLCM Texture Feature Extraction
================================
Extract Gray-Level Co-occurrence Matrix (GLCM) texture features from Sentinel-2.
Uses GEE's built-in glcmTexture() function.

Features: contrast, homogeneity (IDM), entropy, ASM (Angular Second Moment)

Output: data/glcm/glcm_features_{year}_{season}.csv
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
from .s02_sentinel2 import load_seasonal_windows, cloud_mask_s2


@with_retry()
def extract_glcm_features(
    polygon: ee.Geometry,
    start_date: str,
    end_date: str,
) -> dict[str, float] | None:
    """
    Extract GLCM texture features for a single polygon and date range.
    
    Args:
        polygon: Field polygon geometry
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Dictionary of GLCM features, or None if no data
    """
    # Load S2 collection
    s2 = (
        ee.ImageCollection(config.S2_COLLECTION)
        .filterBounds(polygon)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config.S2_CLOUD_THRESHOLD))
        .map(cloud_mask_s2)
    )
    
    count = s2.size().getInfo()
    if count == 0:
        return None
    
    # Create median composite
    composite = s2.median()
    
    # Select bands for GLCM (typically NIR and Red for vegetation/soil)
    # We'll use B4 (Red) and B8 (NIR)
    red = composite.select('B4')
    nir = composite.select('B8')
    
    # Calculate GLCM for Red band
    glcm_red = red.int16().glcmTexture(size=3)  # 3x3 window
    
    # Calculate GLCM for NIR band
    glcm_nir = nir.int16().glcmTexture(size=3)
    
    # Select texture features
    # Available: asm, contrast, corr, var, idm, savg, svar, sent, ent, dvar, dent, imcorr1, imcorr2, maxcorr, diss, inertia, shade, prom
    texture_features = ['contrast', 'ent', 'idm', 'asm']  # contrast, entropy, homogeneity, ASM
    
    # Combine Red and NIR textures
    red_textures = glcm_red.select([f'B4_{feat}' for feat in texture_features])
    nir_textures = glcm_nir.select([f'B8_{feat}' for feat in texture_features])
    
    combined = red_textures.addBands(nir_textures)
    
    # Extract features
    stats = combined.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon,
        scale=10,
        maxPixels=config.GEE_MAX_PIXELS,
    ).getInfo()
    
    # Rename for clarity
    result = {}
    for feat in texture_features:
        red_key = f'B4_{feat}'
        nir_key = f'B8_{feat}'
        if red_key in stats:
            result[f'glcm_red_{feat}'] = stats[red_key]
        if nir_key in stats:
            result[f'glcm_nir_{feat}'] = stats[nir_key]
    
    result['image_count'] = count
    
    return result


def main() -> None:
    """Main execution: extract GLCM features for all years and seasons."""
    print("="*60)
    print("GLCM Texture Feature Extraction")
    print("="*60)
    
    authenticate_and_initialize()
    
    print("\nLoading seasonal windows...")
    seasonal_windows = load_seasonal_windows()
    
    print("Loading field polygons from database...")
    fields_gdf = get_field_polygons()
    print(f"  Found {len(fields_gdf)} unique field polygons")
    
    config.GLCM_DIR.mkdir(parents=True, exist_ok=True)
    
    for year in config.YEARS:
        if year not in seasonal_windows:
            print(f"\n⚠ No seasonal windows for {year}, skipping")
            continue
        
        for season, (start_date, end_date) in seasonal_windows[year].items():
            print(f"\n── {year} / {season} ──")
            
            # Check if data already exists
            out_path = config.GLCM_DIR / f"glcm_features_{year}_{season}.csv"
            if should_skip_file(out_path):
                print(f"  ✓ Already exists: {out_path.name}")
                continue
            
            print(f"  Processing {start_date} to {end_date}...")
            
            year_fields = fields_gdf[fields_gdf["year"] == year].copy()
            results = []
            total = len(year_fields)
            
            for idx, row in year_fields.iterrows():
                geom_json = json.loads(gpd.GeoSeries([row["geometry"]]).to_json())
                ee_geom = ee.Geometry(geom_json["features"][0]["geometry"])
                
                features = extract_glcm_features(ee_geom, start_date, end_date)
                
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
            
            if results:
                df = pd.DataFrame(results)
                df.to_csv(out_path, index=False)
                print(f"  ✓ Saved: {out_path.name} ({len(df)} fields)")
            else:
                print(f"  ⚠ No data extracted for {year}/{season}")
    
    print("\n" + "="*60)
    print("GLCM extraction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
