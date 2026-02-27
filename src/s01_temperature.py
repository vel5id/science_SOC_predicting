"""
ERA5 Temperature Extraction
============================
Extract monthly temperature data from ERA5 to determine growing season
(months with mean temperature > 0°C) for each year.

Output: data/temperature/era5_temperature_{year}.csv
"""
from __future__ import annotations

import ee
import pandas as pd

from . import config
from .db_utils import get_region_bbox
from .file_utils import should_skip_file
from .gee_auth import authenticate_and_initialize, with_retry


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert Kelvin to Celsius."""
    return kelvin - 273.15


@with_retry()
def extract_temperature_for_year(
    year: int,
    bbox: tuple[float, float, float, float],
) -> pd.DataFrame:
    """
    Extract ERA5 monthly temperature for a given year and region.
    
    Args:
        year: Year to extract (e.g., 2020)
        bbox: (min_lon, min_lat, max_lon, max_lat)
    
    Returns:
        DataFrame with columns: year, month, mean_temp_c, is_growing_season
    """
    # Define region geometry
    min_lon, min_lat, max_lon, max_lat = bbox
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    
    # Load ERA5 monthly aggregated data
    era5 = (
        ee.ImageCollection(config.ERA5_COLLECTION)
        .filterBounds(region)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select("temperature_2m")
    )
    
    # Extract monthly means
    results = []
    for month in range(1, 13):
        month_start = f"{year}-{month:02d}-01"
        if month == 12:
            month_end = f"{year + 1}-01-01"
        else:
            month_end = f"{year}-{month + 1:02d}-01"
        
        monthly_img = (
            era5.filterDate(month_start, month_end)
            .mean()
        )
        
        # Reduce region to get mean temperature
        stats = monthly_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=11132,  # ERA5 native resolution ~11km
            maxPixels=config.GEE_MAX_PIXELS,
        ).getInfo()
        
        temp_k = stats.get("temperature_2m")
        if temp_k is not None:
            temp_c = kelvin_to_celsius(temp_k)
            is_growing = temp_c > config.TEMP_THRESHOLD_C
            
            results.append({
                "year": year,
                "month": month,
                "mean_temp_c": round(temp_c, 2),
                "is_growing_season": is_growing,
            })
        else:
            print(f"  ⚠ No data for {year}-{month:02d}")
    
    return pd.DataFrame(results)


def determine_seasonal_windows(
    temp_df: pd.DataFrame,
) -> dict[str, tuple[str, str]]:
    """
    Determine date ranges for each season based on growing season mask.
    
    Args:
        temp_df: Temperature DataFrame with is_growing_season column
    
    Returns:
        {season_name: (start_date, end_date)}
    """
    year = temp_df["year"].iloc[0]
    growing_months = temp_df[temp_df["is_growing_season"]]["month"].tolist()
    
    if not growing_months:
        print(f"  ⚠ No growing season detected for {year}")
        return {}
    
    # Map seasons to date ranges
    season_windows = {}
    for season_name, (start_month, end_month) in config.SEASONS.items():
        # Check if season months overlap with growing season
        season_months = list(range(start_month, end_month + 1))
        overlap = set(season_months) & set(growing_months)
        
        if overlap:
            actual_start = min(overlap)
            actual_end = max(overlap)
            
            start_date = f"{year}-{actual_start:02d}-01"
            # End date is last day of end month
            if actual_end == 12:
                end_date = f"{year}-12-31"
            else:
                # Use first day of next month, GEE filterDate is exclusive on end
                end_date = f"{year}-{actual_end + 1:02d}-01"
            
            season_windows[season_name] = (start_date, end_date)
    
    return season_windows


def main() -> None:
    """Main execution: extract temperature for all years."""
    print("="*60)
    print("ERA5 Temperature Extraction")
    print("="*60)
    
    # Initialize GEE
    authenticate_and_initialize()
    
    # Get region bbox from database
    print("\nFetching region bounding box from database...")
    bbox = get_region_bbox()
    print(f"  BBOX: {bbox}")
    
    # Create output directory
    config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract for each year
    all_seasons = {}
    
    for year in config.YEARS:
        print(f"\n=== {year} ===")
        
        # Check if data already exists
        out_path = config.TEMP_DIR / f"era5_temperature_{year}.csv"
        if should_skip_file(out_path):
            print(f"  ✓ Already exists: {out_path.name}")
            # Load existing data for seasonal window calculation
            temp_df = pd.read_csv(out_path)
            season_windows = determine_seasonal_windows(temp_df)
            all_seasons[year] = season_windows
            continue
        
        # Extract temperature
        temp_df = extract_temperature_for_year(year, bbox)
        
        # Save to CSV
        temp_df.to_csv(out_path, index=False)
        print(f"  ✓ Saved: {out_path.name}")
        
        # Determine seasonal windows
        season_windows = determine_seasonal_windows(temp_df)
        all_seasons[year] = season_windows
        
        # Print summary
        growing_months = temp_df[temp_df["is_growing_season"]]["month"].tolist()
        print(f"  Growing season months: {growing_months}")
        for season, (start, end) in season_windows.items():
            print(f"    {season:12s}: {start} to {end}")
    
    # Save seasonal windows summary
    summary_path = config.TEMP_DIR / "seasonal_windows.txt"
    with open(summary_path, "w") as f:
        for year, seasons in all_seasons.items():
            f.write(f"\n{year}:\n")
            for season, (start, end) in seasons.items():
                f.write(f"  {season:12s}: {start} to {end}\n")
    
    print(f"\n✓ Seasonal windows saved: {summary_path.name}")
    print("\n" + "="*60)
    print("Temperature extraction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
