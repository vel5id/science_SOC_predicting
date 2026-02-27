"""
Extended S1 Debug: Cross-Year Comparison
=========================================
Check S1 availability for same location across all years.
"""
from __future__ import annotations

import json

import ee
import geopandas as gpd

from src import config
from src.db_utils import get_field_polygons
from src.gee_auth import authenticate_and_initialize
from src.s02_sentinel2 import load_seasonal_windows


def check_s1_by_year(polygon: ee.Geometry, year: int, start: str, end: str) -> int:
    """Check S1 image count for a year."""
    s1 = ee.ImageCollection(config.S1_COLLECTION).filterBounds(polygon).filterDate(start, end)
    return s1.size().getInfo()


def check_s2_by_year(polygon: ee.Geometry, year: int, start: str, end: str) -> int:
    """Check S2 image count for a year."""
    s2 = ee.ImageCollection(config.S2_COLLECTION).filterBounds(polygon).filterDate(start, end)
    return s2.size().getInfo()


def main() -> None:
    """Main execution."""
    print("="*70)
    print("Cross-Year S1 Availability Check")
    print("="*70)
    
    authenticate_and_initialize()
    
    # Load seasonal windows
    seasonal_windows = load_seasonal_windows()
    
    # Load fields and get first 2022 field
    fields_gdf = get_field_polygons()
    field_2022 = fields_gdf[fields_gdf["year"] == 2022].iloc[0]
    
    geom_json = json.loads(gpd.GeoSeries([field_2022["geometry"]]).to_json())
    ee_geom = ee.Geometry(geom_json["features"][0]["geometry"])
    
    print(f"\nTest location: {field_2022['farm']}/{field_2022['field_name']}")
    print(f"Coordinates: {field_2022['centroid_lon']:.4f}, {field_2022['centroid_lat']:.4f}")
    
    print(f"\n{'Year':<6} {'Period':<25} {'S1 Images':<12} {'S2 Images'}")
    print("-" * 70)
    
    for year in config.YEARS:
        if year not in seasonal_windows:
            print(f"{year:<6} {'No seasonal windows':<25} {'N/A':<12} {'N/A'}")
            continue
        
        all_dates = [(s, e) for s, e in seasonal_windows[year].values()]
        start_date = min(s for s, _ in all_dates)
        end_date = max(e for _, e in all_dates)
        
        s1_count = check_s1_by_year(ee_geom, year, start_date, end_date)
        s2_count = check_s2_by_year(ee_geom, year, start_date, end_date)
        
        period_str = f"{start_date} to {end_date}"
        status_s1 = "✓" if s1_count > 0 else "✗"
        status_s2 = "✓" if s2_count > 0 else "✗"
        
        print(f"{year:<6} {period_str:<25} {s1_count:<3} {status_s1:<8} {s2_count:<3} {status_s2}")
    
    print("\n" + "="*70)
    print("Analysis:")
    print("- If S1=0 but S2>0 for 2022 → S1 coverage gap in this region")
    print("- If S1>0 for other years → Confirm S1 data exists, just not 2022")
    print("- If S1=0 for ALL years → Region outside S1 coverage area")
    print("="*70)


if __name__ == "__main__":
    main()
