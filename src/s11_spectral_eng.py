"""
Spectral Engineering (Post-processing)
=======================================
Calculate additional spectral indices and features from existing Sentinel-2 data.
No additional GEE calls required - pure post-processing.

Includes: EVI, Band Ratios, PCA components

Output: data/spectral_eng/spectral_eng_{year}_{season}.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config


def calculate_evi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate EVI (Enhanced Vegetation Index) from S2 bands.

    EVI = 2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))

    Note: If s02_sentinel2 already computed EVI on the GEE side,
    that column will already exist and this function skips it
    to avoid overwriting with a differently-scaled version.
    """
    result = df.copy()

    # Skip if EVI already computed by s02 (GEE-side)
    if 'EVI' in df.columns:
        return result

    # Check if required bands exist
    required = ['B2', 'B4', 'B8']
    if not all(col in df.columns for col in required):
        print("  ⚠ Missing bands for EVI calculation")
        return result

    nir = df['B8']
    red = df['B4']
    blue = df['B2']

    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    # Guard against zero/negative denominator
    safe_denom = denom.where(denom.abs() > 1e-6, np.nan)
    result['EVI'] = 2.5 * ((nir - red) / safe_denom)

    return result


def calculate_band_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate band ratios for soil analysis.

    Ratios: B3/B4, B8/B4, B11/B8

    Uses np.where to produce NaN where the denominator is near-zero
    instead of an epsilon hack that creates extreme outliers.
    """
    result = df.copy()
    THRESHOLD = 1e-4  # minimum denominator value

    # B3/B4 (Green/Red)
    if 'B3' in df.columns and 'B4' in df.columns:
        denom = df['B4']
        result['B3_B4'] = np.where(
            denom.abs() > THRESHOLD, df['B3'] / denom, np.nan
        )

    # B8/B4 (NIR/Red)
    if 'B8' in df.columns and 'B4' in df.columns:
        denom = df['B4']
        result['B8_B4'] = np.where(
            denom.abs() > THRESHOLD, df['B8'] / denom, np.nan
        )

    # B11/B8 (SWIR1/NIR)
    if 'B11' in df.columns and 'B8' in df.columns:
        denom = df['B8']
        result['B11_B8'] = np.where(
            denom.abs() > THRESHOLD, df['B11'] / denom, np.nan
        )

    return result


def calculate_pca(
    df: pd.DataFrame,
    n_components: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculate PCA components from spectral bands.

    Applies StandardScaler before PCA to ensure rotation invariance.
    Without scaling, PCA is dominated by bands with highest variance
    (typically NIR), making PC1 ≈ NIR.

    Args:
        df: DataFrame with S2 bands
        n_components: Number of PCA components to extract (default: 5)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with added PCA_1, PCA_2, ..., PCA_n columns
    """
    result = df.copy()

    # Select spectral bands for PCA
    band_cols = [col for col in df.columns if col.startswith('B') and col[1:].isdigit()]

    if len(band_cols) < 3:
        print("  ⚠ Not enough bands for PCA")
        return result

    # Extract band data
    X = df[band_cols].values

    # Check for NaN values
    if pd.isna(X).any():
        print("  ⚠ NaN values in bands, skipping PCA")
        return result

    # Standardize: zero mean, unit variance per band
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA with deterministic seed
    pca = PCA(
        n_components=min(n_components, len(band_cols)),
        random_state=random_state,
    )
    X_pca = pca.fit_transform(X_scaled)

    # Add PCA components to result
    for i in range(X_pca.shape[1]):
        result[f'PCA_{i+1}'] = X_pca[:, i]

    return result


def process_s2_file(s2_path: str, output_path: str) -> None:
    """
    Process a single S2 CSV file and add spectral engineering features.
    
    Args:
        s2_path: Path to input S2 features CSV
        output_path: Path to output spectral engineering CSV
    """
    # Load S2 data
    df = pd.read_csv(s2_path)
    
    # Calculate EVI
    df = calculate_evi(df)
    
    # Calculate band ratios
    df = calculate_band_ratios(df)
    
    # Calculate PCA
    df = calculate_pca(df, n_components=5)
    
    # Save result
    df.to_csv(output_path, index=False)


def main() -> None:
    """Main execution: process all S2 files and add spectral engineering features."""
    print("="*60)
    print("Spectral Engineering (Post-processing)")
    print("="*60)
    
    # Create output directory
    config.SPECTRAL_ENG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all S2 feature files
    s2_files = list(config.S2_DIR.glob("s2_features_*.csv"))
    
    if not s2_files:
        print("\n⚠ No S2 feature files found. Run s02_sentinel2.py first.")
        return
    
    print(f"\nProcessing {len(s2_files)} S2 files...")
    
    for s2_file in s2_files:
        # Parse filename to get year and season
        # Format: s2_features_{year}_{season}.csv
        parts = s2_file.stem.split('_')
        if len(parts) >= 4:
            year = parts[2]
            season = '_'.join(parts[3:])  # handles 'late_summer'
            
            output_file = config.SPECTRAL_ENG_DIR / f"spectral_eng_{year}_{season}.csv"
            
            # Check if already exists
            if output_file.exists():
                print(f"  ✓ Already exists: {output_file.name}")
                continue
            
            print(f"  Processing {s2_file.name}...")
            process_s2_file(str(s2_file), str(output_file))
            print(f"    ✓ Saved: {output_file.name}")
    
    print("\n" + "="*60)
    print("Spectral engineering complete!")
    print("="*60)


if __name__ == "__main__":
    main()
