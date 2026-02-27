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

try:
    ee.Initialize() 
except Exception as e:
    print("Please authenticate Earth Engine: `earthengine authenticate`")
    raise e

def mask_s2_clouds_func():
    def mask_s2_clouds(image):
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])
    return mask_s2_clouds

def get_seasonal_s2(region, year, season_suffix, start_month, end_month):
    start_date = f"{year}-{start_month:02d}-01"
    if end_month == 12:
        end_date = f"{year+1}-01-01"
    else:
        end_date = f"{year}-{end_month+1:02d}-01"
        
    s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(region)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
              .map(mask_s2_clouds_func()))
              
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    
    s2_img = s2_col.median()
    s2_img = s2_img.select(bands)
    
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
    
    final_img = s2_img.addBands([ndvi, bsi, ndsi, ndwi, reci])
    
    old_names = bands + ['NDVI', 'BSI', 'NDSI', 'NDWI', 'RECI']
    new_names = [f"{b}_{season_suffix}" for b in old_names]
    
    return final_img.select(old_names, new_names)

def get_year_s1(region, target_year):
    # Sentinel-1B failed in late 2021, causing massive coverage gaps in 2022-2023.
    # We will use 2020 as a reliable baseline year for radar backscatter (SAR), 
    # since field topology and general roughness do not change rapidly over 2-3 years.
    year = 2020 
    
    start_date = f"{year}-01-01"
    end_date = f"{year+1}-01-01"
    s1_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterBounds(region)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
              .filter(ee.Filter.eq('instrumentMode', 'IW')))
              
    final_s1 = s1_col.median().select(['VV', 'VH'])
    return final_s1

def download_image_as_numpy(ee_img, region, scale, target_shape, num_bands):
    try:
        url = ee_img.getDownloadURL({
            'region': region,
            'scale': scale,
            'format': 'GEO_TIFF',
            'crs': 'EPSG:3857'
        })
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        
        patch_data = []
        with MemoryFile(resp.content) as memfile:
            with memfile.open() as dataset:
                for i in range(1, dataset.count + 1):
                    arr = dataset.read(i)
                    arr = np.nan_to_num(arr, nan=0.0)
                    patch_data.append(arr)
                    
        stacked = np.stack(patch_data, axis=0) # [C, H, W]
        # Ensure we have exactly num_bands
        if stacked.shape[0] < num_bands:
            diff = num_bands - stacked.shape[0]
            zeros = np.zeros((diff, stacked.shape[1], stacked.shape[2]), dtype=stacked.dtype)
            stacked = np.concatenate([stacked, zeros], axis=0)
        elif stacked.shape[0] > num_bands:
            stacked = stacked[:num_bands]
            
        # Center crop / pad to exact target_shape (64, 64)
        _, h, w = stacked.shape
        dh = target_shape[0] - h
        dw = target_shape[1] - w
        
        pad_top = max(0, dh // 2)
        pad_bottom = max(0, dh - pad_top)
        pad_left = max(0, dw // 2)
        pad_right = max(0, dw - pad_left)
        
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            stacked = np.pad(stacked, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
            
        crop_top = max(0, -dh // 2)
        crop_bottom = crop_top + target_shape[0]
        crop_left = max(0, -dw // 2)
        crop_right = crop_left + target_shape[1]
        
        stacked = stacked[:, crop_top:crop_bottom, crop_left:crop_right]
        return stacked
        
    except Exception as e:
        # If extraction fails (e.g. empty collection), return zeros for this specific component
        return np.zeros((num_bands, target_shape[0], target_shape[1]), dtype=np.float32)

def get_gee_multiseason_patch(lon, lat, target_date_str, size=640, scale=10):
    try:
        target_date = datetime.strptime(target_date_str, "%d.%m.%Y")
        year = target_date.year
        
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(size / 2).bounds()
        target_shape = (size // scale, size // scale)
        
        # 1-3. Optical Seasons
        sp_ee = get_seasonal_s2(region, year, 'sp', 3, 5)
        su_ee = get_seasonal_s2(region, year, 'su', 6, 8)
        au_ee = get_seasonal_s2(region, year, 'au', 9, 11)
        
        # 5. DEM
        dem_ee = ee.ImageCollection('COPERNICUS/DEM/GLO30').mosaic().setDefaultProjection('EPSG:3857', None, 30).select('DEM')
        
        # Merge Optical and DEM into 1 request (17*3 + 1 = 52 bands)
        opt_dem_ee = sp_ee.addBands(su_ee).addBands(au_ee).addBands(dem_ee).unmask(0)
        opt_dem_arr = download_image_as_numpy(opt_dem_ee, region, scale, target_shape, 52)
        
        # 4. Radar (keep separate for fallback safety, 2 bands)
        radar_ee = get_year_s1(region, year)
        radar_arr = download_image_as_numpy(radar_ee, region, scale, target_shape, 2)
        
        # Combine all parts: 52 + 2 = 54 channels
        combined_arr = np.concatenate([opt_dem_arr, radar_arr], axis=0)
        
        return combined_arr
        
    except Exception as e:
        print(f"Error extracting patch for ({lon}, {lat}): {e}")
        return None

def process_row(row, out_dir_64):
    idx = row['index']
    lon = row['centroid_lon']
    lat = row['centroid_lat']
    date_str = row['sampling_date']
    
    dest_path_64 = os.path.join(out_dir_64, f"patch_idx_{idx}.npy")
    if os.path.exists(dest_path_64):
        return True # Skip
    
    # Sleep to avoid quota triggers
    time.sleep(np.random.uniform(0.1, 0.4))
    
    patch_64 = get_gee_multiseason_patch(lon, lat, date_str, size=640, scale=10)
    
    if patch_64 is not None:
        np.save(dest_path_64, patch_64)
        return True
    return False

def extract_all():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    df_valid = df.dropna(subset=['ph', 'k', 'p']).reset_index(drop=True)
    unique_locations = df_valid[['grid_id', 'centroid_lon', 'centroid_lat', 'sampling_date']].drop_duplicates()
    
    base_df = df_valid.loc[unique_locations.index].copy()
    base_df = base_df.reset_index(drop=True)
    base_df = base_df.reset_index() # Add 'index' column 0..N
    
    print(f"Total unique locations to process: {len(base_df)}")
    
    out_dir_64 = os.path.join(_PROJECT_ROOT, "data/patches_multiseason_64")
    os.makedirs(out_dir_64, exist_ok=True)
    
    rows = [row for _, row in base_df.iterrows()]
    
    print("Starting concurrent extraction for MULTISEASON (48 channels)...")
    successes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_row, r, out_dir_64): r for r in rows}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            if future.result():
                successes += 1
                
    print(f"Extraction complete! Successfully downloaded: {successes} / {len(rows)}")

if __name__ == "__main__":
    extract_all()
