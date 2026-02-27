"""
Semivariogram Analysis for Spatial Block CV
============================================
Calculate empirical semivariograms for soil properties to determine
optimal spatial block size for cross-validation.

Output: data/semivariograms/{property}_semivariogram.csv and .png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skgstat import Variogram

from . import config
from .db_utils import get_connection


def load_soil_data() -> pd.DataFrame:
    """Load soil samples with coordinates and target properties."""
    conn = get_connection()
    
    query = """
    SELECT 
        centroid_lon, 
        centroid_lat,
        ph,
        hu,
        no3,
        p,
        k
    FROM soil_samples
    WHERE centroid_lon IS NOT NULL 
      AND centroid_lat IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def calculate_semivariogram(
    coords: np.ndarray,
    values: np.ndarray,
    property_name: str,
    output_dir: Path,
) -> dict[str, float]:
    """
    Calculate empirical semivariogram and fit model.
    
    Args:
        coords: Nx2 array of (lon, lat) coordinates
        values: N array of property values
        property_name: Name of the property (for output files)
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary with fitted parameters: range, sill, nugget, model
    """
    # Remove NaN values
    mask = ~np.isnan(values)
    coords_clean = coords[mask]
    values_clean = values[mask]
    
    if len(values_clean) < 10:
        print(f"  ⚠ Not enough data for {property_name} ({len(values_clean)} samples)")
        return {}
    
    # Calculate variogram
    # Use spherical model by default, maxlag = 50% of max distance
    try:
        V = Variogram(
            coordinates=coords_clean,
            values=values_clean,
            model='spherical',
            maxlag='median',  # Use median of pairwise distances
            n_lags=25,
            normalize=False,
        )
        
        # Extract parameters
        params = {
            'range': V.parameters[0],  # Range (in degrees, ~111km per degree at equator)
            'sill': V.parameters[1],   # Sill
            'nugget': V.parameters[2] if len(V.parameters) > 2 else 0.0,
            'model': 'spherical',
            'n_samples': len(values_clean),
        }
        
        # Convert range from degrees to km.
        # 1° latitude ≈ 111 km (meridional). 1° longitude ≈ 111*cos(lat) km.
        # Variogram range includes both directions, so use the mean_lat correction.
        mean_lat = coords_clean[:, 1].mean()
        km_per_deg = 111.0 * np.cos(np.radians(mean_lat))
        range_km = params['range'] * km_per_deg
        params['range_km'] = round(range_km, 2)
        params['mean_lat_deg'] = round(mean_lat, 2)
        params['km_per_deg'] = round(km_per_deg, 2)
        
        # Recommended block size: 2-3 times the range
        recommended_block_km = range_km * 2.5
        params['recommended_block_km'] = round(recommended_block_km, 2)
        
        # Save variogram plot
        fig = plt.figure(figsize=(10, 6))
        V.plot()
        plt.title(f'Semivariogram: {property_name}')
        plt.xlabel('Distance (degrees)')
        plt.ylabel('Semivariance')
        plt.grid(True, alpha=0.3)
        
        plot_path = output_dir / f"{property_name}_semivariogram.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {property_name}: range={range_km:.1f} km, block={recommended_block_km:.1f} km")
        
        # Save parameters to CSV
        csv_path = output_dir / f"{property_name}_semivariogram.csv"
        pd.DataFrame([params]).to_csv(csv_path, index=False)
        
        return params
        
    except Exception as e:
        print(f"  ✗ Error calculating variogram for {property_name}: {e}")
        return {}


def main() -> None:
    """Main execution: calculate semivariograms for all soil properties."""
    print("="*60)
    print("Semivariogram Analysis for Spatial Block CV")
    print("="*60)
    
    # Create output directory
    config.SEMIVARIOGRAM_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load soil data
    print("\nLoading soil samples...")
    df = load_soil_data()
    print(f"  Loaded {len(df)} samples")
    
    # Prepare coordinates
    coords = df[['centroid_lon', 'centroid_lat']].values
    
    # Properties to analyze
    properties = {
        'ph': df['ph'].values,
        'hu': df['hu'].values,
        'no3': df['no3'].values,
        'p': df['p'].values,
        'k': df['k'].values,
    }
    
    print("\nCalculating semivariograms...")
    all_params = {}
    
    for prop_name, values in properties.items():
        params = calculate_semivariogram(
            coords, values, prop_name, config.SEMIVARIOGRAM_DIR
        )
        if params:
            all_params[prop_name] = params
    
    # Save summary
    if all_params:
        summary_df = pd.DataFrame(all_params).T
        summary_path = config.SEMIVARIOGRAM_DIR / "summary.csv"
        summary_df.to_csv(summary_path)
        print(f"\n✓ Summary saved: {summary_path.name}")
        
        # Print recommendations
        print("\n" + "="*60)
        print("Spatial Block CV Recommendations:")
        print("="*60)
        for prop_name, params in all_params.items():
            print(f"  {prop_name:20s}: {params['recommended_block_km']:.1f} km")
        
        # Overall recommendation (median)
        median_block = np.median([p['recommended_block_km'] for p in all_params.values()])
        print(f"\n  Median recommendation: {median_block:.1f} km")
        print(f"  → Use block size: ~{int(median_block)} km for Spatial Block CV")
    
    print("\n" + "="*60)
    print("Semivariogram analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
