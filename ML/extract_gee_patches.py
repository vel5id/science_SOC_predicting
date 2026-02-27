import os
import sys
import pandas as pd
import numpy as np
import ee
import requests
import rasterio
from rasterio.io import MemoryFile
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import concurrent.futures

# Ensure project root is in path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DATA_PATH = os.path.join(_PROJECT_ROOT, "data/features/master_dataset.csv")
DATA_PATH = os.path.join(_PROJECT_ROOT, "data/features/master_dataset.csv")

try:
    ee.Initialize(project='ee-science-article') # Adjust project name if needed or fallback
except Exception as e:
    try:
        ee.Initialize()
    except Exception as e2:
        print("Please authenticate Earth Engine: `earthengine authenticate`")
        raise e2

def mask_s2_clouds(image):
    """Masks clouds in Sentinel-2 images using the QA60 band."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    # Return scaled reflectance (0-10000 -> 0-1)
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])

def get_gee_patch(lon, lat, target_date_str, size=640, scale=10):
    """
    Downloads a size x size meter patch centered at (lon, lat) around target_date.
    Returns a dictionary of bands (S2 optical + DEM) as numpy arrays.
    """
    try:
        # Parse date and define a time window (-1.5 months to +1.5 months)
        target_date = datetime.strptime(target_date_str, "%d.%m.%Y")
        start_date = (target_date - timedelta(days=45)).strftime("%Y-%m-%d")
        end_date = (target_date + timedelta(days=45)).strftime("%Y-%m-%d")
        
        point = ee.Geometry.Point([lon, lat])
        # A 640m box gives approx 64x64 pixels at 10m scale
        region = point.buffer(size / 2).bounds()
        
        # 1. Get Sentinel-2
        s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(region)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                  .map(mask_s2_clouds))
        
        # We take the median over the window to get a clean patch
        # Another option is to take the most central image, but median is safer for agriculture patches
        s2_img = s2_col.median()
        
        # Select required bands
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        s2_img = s2_img.select(bands)
        
        # Calculate 5 Spectral Indices
        ndvi = s2_img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        bsi = s2_img.expression(
            '((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))', {
                'B11': s2_img.select('B11'),
                'B4': s2_img.select('B4'),
                'B8': s2_img.select('B8'),
                'B2': s2_img.select('B2')
        }).rename('BSI')
        
        ndsi = s2_img.normalizedDifference(['B4', 'B8']).rename('NDSI')
        ndwi = s2_img.normalizedDifference(['B8', 'B11']).rename('NDWI')
        
        reci = s2_img.expression(
            '(B8 / B5) - 1.0', {
                'B8': s2_img.select('B8'),
                'B5': s2_img.select('B5')
        }).rename('RECI')
        
        s2_img = s2_img.addBands([ndvi, bsi, ndsi, ndwi, reci])
        
        # 2. Get DEM (Copernicus 30m)
        dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').mosaic().setDefaultProjection('EPSG:3857', None, 30).select('DEM')
        
        # Combine them
        combined = s2_img.addBands(dem)
        
        # 3. Download GeoTIFF directly to memory
        url = combined.getDownloadURL({
            'region': region,
            'scale': scale,
            'format': 'GEO_TIFF',
            'crs': 'EPSG:3857'  # Web mercator to keep pixels roughly square
        })
        
        resp = requests.get(url)
        resp.raise_for_status()
        
        # 4. Read to numpy arrays using Rasterio
        patch_data = {}
        with MemoryFile(resp.content) as memfile:
            with memfile.open() as dataset:
                # ee combined bands order: B1..B12, NDVI, BSI, NDSI, NDWI, RECI, DEM
                all_bands = bands + ['NDVI', 'BSI', 'NDSI', 'NDWI', 'RECI', 'DEM']
                for i, bname in enumerate(all_bands, start=1):
                    # Fill NaNs with 0 or mean just in case
                    arr = dataset.read(i)
                    arr = np.nan_to_num(arr, nan=0.0)
                    
                    # Ensure it's exactly 64x64 (sometimes EE returns 65x65 or 63x63 depending on boundaries)
                    # We can crop/pad it to exact size. At 10m scale, 640m bounding box usually is 64x64 or 65x65
                    target_shape = (size // scale, size // scale)
                    h, w = arr.shape
                    
                    # Center crop/pad
                    dh = target_shape[0] - h
                    dw = target_shape[1] - w
                    
                    pad_top = max(0, dh // 2)
                    pad_bottom = max(0, dh - pad_top)
                    pad_left = max(0, dw // 2)
                    pad_right = max(0, dw - pad_left)
                    
                    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                        arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
                        
                    crop_top = max(0, -dh // 2)
                    crop_bottom = crop_top + target_shape[0]
                    crop_left = max(0, -dw // 2)
                    crop_right = crop_left + target_shape[1]
                    
                    arr = arr[crop_top:crop_bottom, crop_left:crop_right]
                    patch_data[bname] = arr
                    
        # Stack into [C, H, W]
        stacked = np.stack([patch_data[b] for b in all_bands], axis=0)
        return stacked
        
    except Exception as e:
        print(f"Error extracting patch for ({lon}, {lat}): {e}")
        return None

def main():
    print("Loading master dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df_valid = df.dropna(subset=['ph', 'k', 'p']).reset_index(drop=True)
    
    unique_locations = df_valid[['grid_id', 'centroid_lon', 'centroid_lat', 'sampling_date']].drop_duplicates()
    print(f"Total unique grids to download: {len(unique_locations)}")
    
    # User requested sizes 16x16 and 64x64 (since 32x32 is already downloaded/assumed)
    target_sizes_px = [16, 64]
    
    for px_size in target_sizes_px:
        size_m = px_size * 10
        out_dir = os.path.join(_PROJECT_ROOT, f"data/patches_{px_size}")
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"\n--- Extracting {px_size}x{px_size} patches (size_m={size_m}) ---")
        success_count = 0
        
        def process_row(args):
            idx, row = args
            lon = row['centroid_lon']
            lat = row['centroid_lat']
            date = row['sampling_date']
            out_file = os.path.join(out_dir, f"patch_idx_{idx}.npy")
            if os.path.exists(out_file):
                return 0
            patch = get_gee_patch(lon, lat, date, size=size_m)
            if patch is not None:
                np.save(out_file, patch)
                return 1
            return 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_row, item): item for item in unique_locations.iterrows()}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Size {px_size}x{px_size}"):
                try:
                    success_count += future.result()
                except Exception as e:
                    print(f"Thread failed: {e}")
                    
        print(f"Downloaded {success_count} new patches for {px_size}x{px_size}.")
        print(f"Extraction step completed for {px_size}. Saved to {out_dir}")

if __name__ == "__main__":
    main()
