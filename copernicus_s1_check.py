"""
Copernicus S1 Data Download
============================
Download Sentinel-1 data directly from Copernicus Data Space Ecosystem.
Alternative to GEE for missing 2022 data.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from src import config
from src.db_utils import get_field_polygons
from src.s02_sentinel2 import load_seasonal_windows

# Copernicus API credentials
CLIENT_ID = "sh-e9f62cd1-88af-4fd7-a376-5abad9560dec"
CLIENT_SECRET = "VYU5BsJETrK16EGhvPUDKdEX9JlYmsJK"

# API endpoints
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


def get_access_token() -> str:
    """Get OAuth2 access token from Copernicus."""
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
    }
    
    response = requests.post(TOKEN_URL, data=data, timeout=30)
    response.raise_for_status()
    
    return response.json()["access_token"]


def search_s1_products(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    token: str,
) -> list[dict]:
    """
    Search for S1 products in the catalog.
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        token: OAuth2 access token
    
    Returns:
        List of product metadata dictionaries
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # OData filter query
    filter_query = (
        f"Collection/Name eq 'SENTINEL-1' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(("
        f"{min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},"
        f"{min_lon} {max_lat},{min_lon} {min_lat}))') and "
        f"ContentDate/Start ge {start_date}T00:00:00.000Z and "
        f"ContentDate/Start lt {end_date}T23:59:59.999Z"
    )
    
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "$filter": filter_query,
        "$top": 1000,  # Max results
        "$orderby": "ContentDate/Start asc",
    }
    
    response = requests.get(CATALOG_URL, headers=headers, params=params, timeout=60)
    response.raise_for_status()
    
    return response.json().get("value", [])


def check_s1_availability_2022(year: int = 2022) -> None:
    """Check S1 product availability for 2022."""
    print("="*70)
    print(f"Copernicus S1 Availability Check for {year}")
    print("="*70)
    
    # Load seasonal windows
    print("\nLoading seasonal windows...")
    seasonal_windows = load_seasonal_windows()
    
    if year not in seasonal_windows:
        print(f"✗ No seasonal windows for {year}!")
        return
    
    # Get date range
    all_dates = [(s, e) for s, e in seasonal_windows[year].values()]
    start_date = min(s for s, _ in all_dates)
    end_date = max(e for _, e in all_dates)
    
    print(f"Period: {start_date} to {end_date}")
    
    # Load fields and get bounding box
    print("\nLoading field polygons...")
    fields_gdf = get_field_polygons()
    year_fields = fields_gdf[fields_gdf["year"] == year]
    
    if len(year_fields) == 0:
        print(f"✗ No fields for {year}!")
        return
    
    # Get overall bounding box for all fields
    bounds = year_fields.total_bounds  # (minx, miny, maxx, maxy)
    bbox = tuple(bounds)
    
    print(f"Fields: {len(year_fields)}")
    print(f"BBOX: ({bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f})")
    
    # Get access token
    print("\nAuthenticating with Copernicus...")
    try:
        token = get_access_token()
        print("✓ Authentication successful")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return
    
    # Search for products
    print(f"\nSearching for S1 products...")
    try:
        products = search_s1_products(bbox, start_date, end_date, token)
        print(f"✓ Found {len(products)} products")
        
        if len(products) == 0:
            print(f"\n⚠ No S1 products found for {year} in this region!")
            print("This confirms GEE data gap is real.")
            return
        
        # Display product summary
        print(f"\nProduct Summary:")
        print(f"{'Date':<12} {'Product Type':<15} {'Mode':<8} {'Polarization'}")
        print("-" * 70)
        
        for product in products[:10]:  # Show first 10
            name = product.get("Name", "")
            date_str = product.get("ContentDate", {}).get("Start", "")[:10]
            
            # Parse product type from name (e.g., S1A_IW_GRDH_1SDV_...)
            parts = name.split("_")
            mode = parts[1] if len(parts) > 1 else "N/A"
            prod_type = parts[2] if len(parts) > 2 else "N/A"
            polarization = parts[3] if len(parts) > 3 else "N/A"
            
            print(f"{date_str:<12} {prod_type:<15} {mode:<8} {polarization}")
        
        if len(products) > 10:
            print(f"... and {len(products) - 10} more products")
        
    except Exception as e:
        print(f"✗ Search failed: {e}")
        return
    
    print("\n" + "="*70)
    print("✓ Copernicus API can access S1 data for 2022!")
    print("Consider implementing direct Copernicus download for s04_sentinel1.py")
    print("="*70)


if __name__ == "__main__":
    check_s1_availability_2022()
