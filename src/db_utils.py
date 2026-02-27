"""Database utilities for soil_analysis.db."""
import sqlite3
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import wkt

from . import config


def get_connection() -> sqlite3.Connection:
    """Get SQLite connection to soil_analysis.db."""
    if not config.DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {config.DB_PATH}")
    return sqlite3.connect(str(config.DB_PATH))


def get_field_polygons() -> gpd.GeoDataFrame:
    """
    Extract unique field polygons from soil_samples table.
    
    Returns:
        GeoDataFrame with columns: year, farm, field_name, geometry
    """
    conn = get_connection()
    
    query = """
    SELECT DISTINCT 
        year, 
        farm, 
        field_name,
        geometry_wkt,
        MIN(centroid_lon) as centroid_lon,
        MIN(centroid_lat) as centroid_lat
    FROM soil_samples
    WHERE geometry_wkt IS NOT NULL
    GROUP BY year, farm, field_name, geometry_wkt
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert WKT to geometry
    df["geometry"] = df["geometry_wkt"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=config.CRS_WGS84)
    
    return gdf


def get_sampling_dates() -> dict[int, list[str]]:
    """
    Get sampling dates grouped by year.
    
    Returns:
        {year: [date1, date2, ...]}
    """
    conn = get_connection()
    
    query = """
    SELECT DISTINCT year, sampling_date
    FROM soil_samples
    WHERE sampling_date IS NOT NULL
    ORDER BY year, sampling_date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    dates_by_year = {}
    for year in df["year"].unique():
        dates_by_year[year] = df[df["year"] == year]["sampling_date"].tolist()
    
    return dates_by_year


def get_region_bbox() -> tuple[float, float, float, float]:
    """
    Calculate bounding box of all sampling points.
    
    Returns:
        (min_lon, min_lat, max_lon, max_lat)
    """
    conn = get_connection()
    
    query = """
    SELECT 
        MIN(centroid_lon) as min_lon,
        MIN(centroid_lat) as min_lat,
        MAX(centroid_lon) as max_lon,
        MAX(centroid_lat) as max_lat
    FROM soil_samples
    """
    
    result = pd.read_sql_query(query, conn)
    conn.close()
    
    return tuple(result.iloc[0].tolist())


def save_features_to_db(
    table_name: str,
    df: pd.DataFrame,
    if_exists: str = "replace",
) -> None:
    """
    Save feature DataFrame to SQLite database.
    
    Args:
        table_name: Name of the table to create/update
        df: DataFrame to save
        if_exists: 'fail', 'replace', or 'append'
    """
    conn = get_connection()
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.close()
    print(f"✓ Saved {len(df)} rows to table '{table_name}'")


def table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    exists = cursor.fetchone() is not None
    conn.close()
    return exists
