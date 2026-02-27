"""
Topographic Feature Extraction
===============================
Extract DEM-derived topographic features for each field polygon.

Output: data/topography/topo_features.csv
"""
from __future__ import annotations

import json
import math

import ee
import geopandas as gpd
import pandas as pd

from . import config
from .db_utils import get_field_polygons
from .file_utils import should_skip_file
from .gee_auth import authenticate_and_initialize, with_retry


@with_retry()
def extract_topo_features(polygon: ee.Geometry) -> dict[str, float]:
    """
    Extract topographic features for a single polygon.

    Includes: DEM, slope, aspect (as sin/cos), TWI, plan/profile curvature, TPI

    Notes on approximations:
        - TWI: True TWI = ln(A / tanα) requires upstream contributing area A,
          which needs D8/D-inf flow routing not available in GEE at scale.
          We use a proxy: TWI ≈ −ln(tanα), i.e. A = 1 m²/m. This is standard
          for flat steppe terrain where flow accumulation is near-uniform.
          Cite: Sørensen et al. (2006) Geomorphology 77, 65–76.
        - Aspect: circular variable, so we decompose into sin(α) / cos(α)
          before any mean-aggregation to avoid the circular mean problem
          (e.g. mean(350°, 10°) = 180° in arithmetic, but 0° in circular).
        - Curvature: Second-order derivative kernels (Zevenbergen & Thorne, 1987).
    """
    # Load DEM (COPERNICUS/DEM/GLO30 is an ImageCollection, need to mosaic)
    dem_collection = ee.ImageCollection(config.DEM_COLLECTION)
    dem = dem_collection.select('DEM').mosaic()

    # Basic terrain derivatives
    slope = ee.Terrain.slope(dem)
    aspect = ee.Terrain.aspect(dem)

    # ─── Aspect sin/cos decomposition ───
    # Decompose circular angle into Cartesian components for safe aggregation
    aspect_rad = aspect.multiply(math.pi / 180)
    aspect_sin = aspect_rad.sin().rename("aspect_sin")
    aspect_cos = aspect_rad.cos().rename("aspect_cos")

    # ─── TWI (Topographic Wetness Index) ───
    # Approximation: TWI ≈ −ln(tan(α)), assuming A = 1 m²/m
    slope_rad = slope.multiply(math.pi / 180)
    tan_slope = slope_rad.tan()
    # Avoid ln(0): clamp tan_slope to minimum 0.001 (~0.06°)
    twi = tan_slope.max(0.001).log().multiply(-1).rename("TWI")

    # ─── Curvature (Plan and Profile) ───
    # Second-order derivative kernels (Zevenbergen & Thorne, 1987)
    # Plan curvature: ∂²z/∂x² (horizontal, across slope)
    plan_kernel = ee.Kernel.fixed(3, 3, [
        [0, 0, 0],
        [1, -2, 1],
        [0, 0, 0]
    ])

    # Profile curvature: ∂²z/∂y² (vertical, along slope)
    profile_kernel = ee.Kernel.fixed(3, 3, [
        [0, 1, 0],
        [0, -2, 0],
        [0, 1, 0]
    ])

    plan_curv = dem.convolve(plan_kernel).rename("plan_curvature")
    profile_curv = dem.convolve(profile_kernel).rename("profile_curvature")

    # ─── TPI (Topographic Position Index) ───
    # TPI = DEM - mean(DEM in neighborhood)
    # Use focal_mean with 300m radius (~10 pixels at 30m resolution)
    dem_mean = dem.focal_mean(radius=300, units='meters')
    tpi = dem.subtract(dem_mean).rename("TPI")

    # Combine all layers
    topo = dem.addBands([
        slope, aspect_sin, aspect_cos, twi,
        plan_curv, profile_curv, tpi,
    ])
    topo = topo.select([
        "DEM", "slope", "aspect_sin", "aspect_cos", "TWI",
        "plan_curvature", "profile_curvature", "TPI",
    ])

    # Extract features
    stats = topo.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon,
        scale=30,
        maxPixels=config.GEE_MAX_PIXELS,
    ).getInfo()

    return stats



def main() -> None:
    """Main execution: extract topographic features for all fields."""
    print("="*60)
    print("Topographic Feature Extraction")
    print("="*60)
    
    authenticate_and_initialize()
    
    print("\nLoading field polygons from database...")
    fields_gdf = get_field_polygons()
    print(f"  Found {len(fields_gdf)} unique field polygons")
    
    config.TOPO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    out_path = config.TOPO_DIR / "topo_features.csv"
    if should_skip_file(out_path):
        print(f"\n✓ Already exists: {out_path.name}")
        print(f"Topography data is static. Delete {out_path.name} to re-extract.")
        print("\n" + "="*60)
        print("Topographic extraction complete!")
        print("="*60)
        return
    
    print("\nExtracting topographic features...")
    results = []
    total = len(fields_gdf)
    
    for idx, row in fields_gdf.iterrows():
        geom_json = json.loads(gpd.GeoSeries([row["geometry"]]).to_json())
        ee_geom = ee.Geometry(geom_json["features"][0]["geometry"])
        
        features = extract_topo_features(ee_geom)
        
        result = {
            "year": row["year"],
            "farm": row["farm"],
            "field_name": row["field_name"],
            "centroid_lon": row["centroid_lon"],
            "centroid_lat": row["centroid_lat"],
            **features,
        }
        results.append(result)
        print(f"  [{len(results)}/{total}] {row['farm']}/{row['field_name']} ✓")
    
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_path.name} ({len(df)} fields)")
    
    print("\n" + "="*60)
    print("Topographic extraction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
