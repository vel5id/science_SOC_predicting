import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def run_sigma_audit():
    data_path = Path("c:/Claude/science_article/data/features/full_dataset.csv")
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    print("="*60)
    print("📊 SIGMA AUDIT REPORT: data/features/full_dataset.csv")
    print("="*60)
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    
    # 1. Missingness Analysis
    print("\n🧐 1. MISSINGNESS ANALYSIS")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    if len(missing_cols) > 0:
        print(f"Features with missing data: {len(missing_cols)}")
        print("Top 10 highest missingness:")
        for col, pct in missing_cols.head(10).items():
            print(f"  - {col}: {pct:.2f}% ({missing[col]} rows)")
    else:
        print("No missing data found.")
        
    print("\n🔍 2. EXACT ZEROS AND OUTLIER SENTINELS")
    # Zeros where zeros shouldn't usually exist for continuous environmental variables (except certain spectral indices)
    suspect_zeros = (df == 0.0).sum()
    suspect_zeros = suspect_zeros[suspect_zeros > 0].sort_values(ascending=False)
    if len(suspect_zeros) > 0:
        print("Columns with exact 0.0 values (potential imputation artifacts or pure zero):")
        for col, count in suspect_zeros.head(10).items():
            print(f"  - {col}: {count} rows ({(count/len(df))*100:.2f}%)")

    # Sentinels
    sentinels = [-9999, -999, 9999, 999]
    for s in sentinels:
        count = (df == s).sum().sum()
        if count > 0:
            print(f"  WARNING: Found {count} instances of potential sentinel value {s}")
            
    print("\n📏 3. PHYSICAL BOUNDS (SANITY ASSERTIONS)")
    bounds = {
        "ph": (3.5, 9.5),
        "soc": (0.01, 20.0),
        "climate_MAT": (-10.0, 30.0), # Celsius
        "sg_sand_pct_0_5cm": (0, 100),
    }
    
    for col, (min_val, max_val) in bounds.items():
        if col in df.columns:
            # Check range
            actual_min = df[col].min()
            actual_max = df[col].max()
            print(f"  - {col}: Expected [{min_val}, {max_val}], Actual [{actual_min:.2f}, {actual_max:.2f}]")
            out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(out_of_bounds) > 0:
                print(f"    ⚠️ WARNING: {len(out_of_bounds)} rows out of strict physical bounds!")
    
    # Check NDVI indices (-1 to +1)
    ndvi_cols = [c for c in df.columns if "NDVI" in c]
    if ndvi_cols:
        print(f"\n🌱 4. VEGETATION INDICES VALIDATION")
        for col in ndvi_cols[:5]: # sample top 5
            c_min, c_max = df[col].min(), df[col].max()
            if c_min < -1.0 or c_max > 1.0:
                print(f"  - {col}: [INVALID] Range [{c_min:.3f}, {c_max:.3f}] - NDVI must be strictly [-1, 1]")
            else:
                print(f"  - {col}: [VALID] Range [{c_min:.3f}, {c_max:.3f}]")

    print("\n📉 5. DISTRIBUTIONAL SKEWNESS (SELECT FEATURES)")
    # Select important numeric features representing soil and climate
    target_cols = ["ph", "soc", "climate_MAT", "topo_DEM"]
    for col in target_cols:
        if col in df.columns:
            clean_s = df[col].dropna()
            if len(clean_s) > 10:
                skew = stats.skew(clean_s)
                kurt = stats.kurtosis(clean_s)
                print(f"  - {col}: Skewness = {skew:.2f}, Kurtosis = {kurt:.2f}")
                if abs(skew) > 1:
                    print(f"    ⚠️ High skewness detected. Consider non-parametric tests or RobustScaler.")
                
    # Check soil texture properties summing to 100
    print("\n🧱 6. TEXTURE VALIDATION")
    sand_col = "sg_sand_pct_0_5cm"
    silt_col = "sg_silt_pct_0_5cm"
    clay_col = "sg_clay_pct_0_5cm"
    
    if all(c in df.columns for c in [sand_col, silt_col, clay_col]):
        total = df[sand_col] + df[silt_col] + df[clay_col]
        total_mean = total.mean()
        total_std = total.std()
        print(f"  - Sum of Sand + Silt + Clay (0-5cm): Mean = {total_mean:.2f}%, StdDev = {total_std:.2f}%")
        if abs(total_mean - 100) > 2.0:
            print("    ⚠️ WARNING: Soil textures do not sum to ~100%. Check SoilGrids extraction logic.")

if __name__ == "__main__":
    run_sigma_audit()
