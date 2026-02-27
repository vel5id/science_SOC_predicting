"""
Feature Merging and Database Integration
=========================================
Merge all extracted satellite features with soil_samples table.

Fixes applied:
  - SOC conversion (hu * 0.58, Van Bemmelen factor)
  - Static features (S1, topo, climate) aggregated and merged into full_dataset
  - Removed destructive replace(0.0, NA) — legitimate zeros preserved
  - Topography data aggregated by (year, farm, field_name) via mean

Output:
  - data/features/full_dataset.csv
  - data/features/field_static_features.csv
  - SQLite tables: temperature_profile, sentinel2_features, etc.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config
from .db_utils import get_connection, save_features_to_db

logger = logging.getLogger(__name__)

# Van Bemmelen factor for humus -> SOC conversion
# SOC = Humus * 0.58  (standard; some use 0.55-0.60 for chernozems)
VAN_BEMMELEN_FACTOR = 0.58

# Columns that should never be included in mean-aggregation
# because they are metadata, not features.
_AGGREGATION_EXCLUDE = {"image_count", "centroid_lon", "centroid_lat"}


def load_temperature_data() -> pd.DataFrame:
    """Load all temperature CSV files."""
    dfs = []
    for csv_file in config.TEMP_DIR.glob("era5_temperature_*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_sentinel2_data() -> pd.DataFrame:
    """Load all Sentinel-2 CSV files."""
    dfs = []
    for csv_file in config.S2_DIR.glob("s2_features_*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_landsat8_data() -> pd.DataFrame:
    """Load all Landsat 8 CSV files."""
    dfs = []
    for csv_file in config.L8_DIR.glob("l8_features_*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_sentinel1_data() -> pd.DataFrame:
    """Load all Sentinel-1 CSV files."""
    dfs = []
    for csv_file in config.S1_DIR.glob("s1_features_*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_topography_data() -> pd.DataFrame:
    """Load topography CSV."""
    topo_file = config.TOPO_DIR / "topo_features.csv"
    if topo_file.exists():
        return pd.read_csv(topo_file)
    return pd.DataFrame()


def load_climate_data() -> pd.DataFrame:
    """Load all climate CSV files."""
    dfs = []
    for csv_file in config.CLIMATE_DIR.glob("climate_features_*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_soilgrids_data() -> pd.DataFrame:
    """Load SoilGrids texture/properties CSV."""
    sg_file = config.SOIL_DIR / "soilgrids_features.csv"
    if sg_file.exists():
        return pd.read_csv(sg_file)
    return pd.DataFrame()


def load_spectral_eng_data() -> pd.DataFrame:
    """Load all spectral engineering CSV files."""
    dfs = []
    for csv_file in config.SPECTRAL_ENG_DIR.glob("spectral_eng_*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_glcm_data() -> pd.DataFrame:
    """Load all GLCM CSV files."""
    dfs = []
    for csv_file in config.GLCM_DIR.glob("glcm_features_*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_soil_samples() -> pd.DataFrame:
    """Load soil_samples table from database."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM soil_samples", conn)
    conn.close()
    return df


def add_soc_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert humus (hu, %) to Soil Organic Carbon (soc, %).

    Uses the Van Bemmelen factor: SOC = Humus * 0.58
    This is the standard conversion used in soil science literature.
    For chernozems, some studies use 0.55-0.60; 0.58 is the most common.
    """
    if "hu" in df.columns:
        df["soc"] = pd.to_numeric(df["hu"], errors="coerce") * VAN_BEMMELEN_FACTOR
        # Round to 3 decimal places
        df["soc"] = df["soc"].round(3)
        non_null = df["soc"].notna().sum()
        print(f"  SOC conversion: {non_null} values computed (hu * {VAN_BEMMELEN_FACTOR})")
    else:
        print("  WARNING: 'hu' column not found, SOC not computed")
    return df


def _aggregate_to_field_level(
    df: pd.DataFrame,
    group_keys: list[str],
    prefix: str,
) -> pd.DataFrame:
    """
    Aggregate pixel-level data to field-level by computing mean per group.

    Args:
        df: Raw data with potentially multiple rows per field (pixel-level)
        group_keys: Columns to group by, e.g. ["year", "farm", "field_name"]
        prefix: Column prefix for renamed features, e.g. "s1_"

    Returns:
        Aggregated DataFrame with one row per unique group
    """
    drop_cols = _AGGREGATION_EXCLUDE
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Separate group keys from feature columns
    feature_cols = [c for c in df_clean.columns if c not in group_keys]

    # Aggregate numeric features by mean
    agg_dict = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            agg_dict[col] = "mean"

    if not agg_dict:
        return df_clean[group_keys].drop_duplicates().reset_index(drop=True)

    aggregated = df_clean.groupby(group_keys, as_index=False).agg(agg_dict)

    # Rename feature columns with prefix
    rename_map = {c: f"{prefix}{c}" for c in agg_dict if c not in group_keys}
    aggregated = aggregated.rename(columns=rename_map)

    return aggregated


def merge_static_features() -> pd.DataFrame:
    """
    Merge static field-level features (S1 + Topography + Climate).

    S1 and Topo data may have multiple rows per field (pixel-level).
    They are aggregated to field-level (mean) before merging.

    Returns:
        DataFrame with static features keyed by (year, farm, field_name)
    """
    print("\nMerging static field-level features...")

    # Load S1, Topography, and Climate data
    s1_df = load_sentinel1_data()
    topo_df = load_topography_data()
    climate_df = load_climate_data()

    # Get unique fields from soil samples
    soil_df = load_soil_samples()
    unique_fields = soil_df[["year", "farm", "field_name"]].drop_duplicates()
    print(f"  Unique fields: {len(unique_fields)} rows")

    static_merged = unique_fields.copy()

    # Merge S1 (annual, may have multiple rows per field)
    if not s1_df.empty:
        s1_agg = _aggregate_to_field_level(
            s1_df, ["year", "farm", "field_name"], "s1_"
        )
        static_merged = static_merged.merge(
            s1_agg,
            on=["year", "farm", "field_name"],
            how="left",
        )
        matched = static_merged[
            static_merged.columns[static_merged.columns.str.startswith("s1_")]
        ].notna().any(axis=1).sum()
        print(f"  Merged Sentinel-1: {matched}/{len(static_merged)} fields matched")

    # Merge Topography (static, may have multiple rows per field)
    if not topo_df.empty:
        # Topo has year column but values are static - aggregate by (farm, field_name)
        topo_clean = topo_df.drop(columns=["year"], errors="ignore")
        topo_agg = _aggregate_to_field_level(
            topo_clean, ["farm", "field_name"], "topo_"
        )
        static_merged = static_merged.merge(
            topo_agg,
            on=["farm", "field_name"],
            how="left",
        )
        matched = static_merged[
            static_merged.columns[static_merged.columns.str.startswith("topo_")]
        ].notna().any(axis=1).sum()
        print(f"  Merged Topography: {matched}/{len(static_merged)} fields matched")

    # Merge Climate (annual, may have multiple rows per field)
    if not climate_df.empty:
        climate_agg = _aggregate_to_field_level(
            climate_df, ["year", "farm", "field_name"], "climate_"
        )
        static_merged = static_merged.merge(
            climate_agg,
            on=["year", "farm", "field_name"],
            how="left",
        )
        matched = static_merged[
            static_merged.columns[static_merged.columns.str.startswith("climate_")]
        ].notna().any(axis=1).sum()
        print(f"  Merged Climate: {matched}/{len(static_merged)} fields matched")

    return static_merged


def merge_all_features() -> pd.DataFrame:
    """
    Merge ALL features (seasonal + static) with soil samples into a single dataset.

    Includes:
      - Soil samples (base table) with SOC conversion
      - Sentinel-2 (seasonal, pivoted to wide)
      - Landsat 8 (seasonal, pivoted to wide)
      - Spectral Engineering (seasonal, pivoted to wide)
      - GLCM (seasonal, pivoted to wide)
      - Sentinel-1 (annual, aggregated to field-level)
      - Topography (static, aggregated to field-level)
      - Climate (annual, aggregated to field-level)

    Returns:
        Merged DataFrame with all features
    """
    print("\nLoading data...")

    # Load soil samples (base table)
    soil_df = load_soil_samples()
    print(f"  Soil samples: {len(soil_df)} rows")

    # Add SOC column from humus
    soil_df = add_soc_column(soil_df)

    # Load satellite features
    s2_df = load_sentinel2_data()
    print(f"  Sentinel-2: {len(s2_df)} rows")

    l8_df = load_landsat8_data()
    print(f"  Landsat 8: {len(l8_df)} rows")

    s1_df = load_sentinel1_data()
    print(f"  Sentinel-1: {len(s1_df)} rows")

    topo_df = load_topography_data()
    print(f"  Topography: {len(topo_df)} rows")

    climate_df = load_climate_data()
    print(f"  Climate: {len(climate_df)} rows")

    spectral_eng_df = load_spectral_eng_data()
    print(f"  Spectral Engineering: {len(spectral_eng_df)} rows")

    glcm_df = load_glcm_data()
    print(f"  GLCM: {len(glcm_df)} rows")

    print("\nMerging features...")

    merged = soil_df.copy()

    # ── Seasonal features (S2, L8, Spectral, GLCM) ──
    # These are pivoted from long format (one row per season) to wide format

    if not s2_df.empty:
        # C-3 guard: assert no duplicate (year, farm, field, season)
        dup_key = ["year", "farm", "field_name", "season"]
        dupes = s2_df.duplicated(subset=dup_key, keep=False)
        if dupes.any():
            n = dupes.sum()
            logger.warning(
                f"S2: {n} duplicate rows for (year, farm, field, season) detected. "
                "pivot_table will average them — check s02 output."
            )
        _pivot_exclude = {"year", "farm", "field_name", "season"} | _AGGREGATION_EXCLUDE
        s2_feat_cols = [
            c for c in s2_df.columns
            if c not in _pivot_exclude
        ]
        s2_pivot = s2_df.pivot_table(
            index=["year", "farm", "field_name"],
            columns="season",
            values=s2_feat_cols,
        )
        s2_pivot.columns = [f"s2_{col[0]}_{col[1]}" for col in s2_pivot.columns]
        s2_pivot = s2_pivot.reset_index()

        merged = merged.merge(
            s2_pivot,
            on=["year", "farm", "field_name"],
            how="left",
        )
        print(f"  Merged Sentinel-2 (4 seasons)")

    if not l8_df.empty:
        dup_key = ["year", "farm", "field_name", "season"]
        dupes = l8_df.duplicated(subset=dup_key, keep=False)
        if dupes.any():
            logger.warning(
                f"L8: {dupes.sum()} duplicate rows detected — will be averaged."
            )
        _pivot_exclude = {"year", "farm", "field_name", "season"} | _AGGREGATION_EXCLUDE
        l8_feat_cols = [
            c for c in l8_df.columns
            if c not in _pivot_exclude
        ]
        l8_pivot = l8_df.pivot_table(
            index=["year", "farm", "field_name"],
            columns="season",
            values=l8_feat_cols,
        )
        l8_pivot.columns = [f"l8_{col[0]}_{col[1]}" for col in l8_pivot.columns]
        l8_pivot = l8_pivot.reset_index()

        merged = merged.merge(
            l8_pivot,
            on=["year", "farm", "field_name"],
            how="left",
        )
        print(f"  Merged Landsat 8 (4 seasons)")

    if not spectral_eng_df.empty:
        dup_key = ["year", "farm", "field_name", "season"]
        dupes = spectral_eng_df.duplicated(subset=dup_key, keep=False)
        if dupes.any():
            logger.warning(
                f"SpectralEng: {dupes.sum()} duplicate rows detected."
            )
        _pivot_exclude = {"year", "farm", "field_name", "season"} | _AGGREGATION_EXCLUDE
        se_feat_cols = [
            c for c in spectral_eng_df.columns
            if c not in _pivot_exclude
        ]
        spectral_pivot = spectral_eng_df.pivot_table(
            index=["year", "farm", "field_name"],
            columns="season",
            values=se_feat_cols,
        )
        spectral_pivot.columns = [f"spectral_{col[0]}_{col[1]}" for col in spectral_pivot.columns]
        spectral_pivot = spectral_pivot.reset_index()

        merged = merged.merge(
            spectral_pivot,
            on=["year", "farm", "field_name"],
            how="left",
        )
        print(f"  Merged Spectral Engineering (4 seasons)")

    if not glcm_df.empty:
        dup_key = ["year", "farm", "field_name", "season"]
        dupes = glcm_df.duplicated(subset=dup_key, keep=False)
        if dupes.any():
            logger.warning(
                f"GLCM: {dupes.sum()} duplicate rows detected."
            )
        _pivot_exclude = {"year", "farm", "field_name", "season"} | _AGGREGATION_EXCLUDE
        glcm_feat_cols = [
            c for c in glcm_df.columns
            if c not in _pivot_exclude
        ]
        glcm_pivot = glcm_df.pivot_table(
            index=["year", "farm", "field_name"],
            columns="season",
            values=glcm_feat_cols,
        )
        glcm_pivot.columns = [f"glcm_{col[0]}_{col[1]}" for col in glcm_pivot.columns]
        glcm_pivot = glcm_pivot.reset_index()

        merged = merged.merge(
            glcm_pivot,
            on=["year", "farm", "field_name"],
            how="left",
        )
        print(f"  Merged GLCM (4 seasons)")

    # ── Static / annual features (S1, Topo, Climate) ──
    # These are aggregated to field-level (mean) and merged by field key

    if not s1_df.empty:
        s1_agg = _aggregate_to_field_level(
            s1_df, ["year", "farm", "field_name"], "s1_"
        )
        merged = merged.merge(
            s1_agg,
            on=["year", "farm", "field_name"],
            how="left",
        )
        matched = merged["s1_VV"].notna().sum() if "s1_VV" in merged.columns else 0
        print(f"  Merged Sentinel-1: {matched}/{len(merged)} rows matched")

    if not topo_df.empty:
        topo_clean = topo_df.drop(columns=["year"], errors="ignore")
        topo_agg = _aggregate_to_field_level(
            topo_clean, ["farm", "field_name"], "topo_"
        )
        merged = merged.merge(
            topo_agg,
            on=["farm", "field_name"],
            how="left",
        )
        matched = merged["topo_DEM"].notna().sum() if "topo_DEM" in merged.columns else 0
        print(f"  Merged Topography: {matched}/{len(merged)} rows matched")

    if not climate_df.empty:
        climate_agg = _aggregate_to_field_level(
            climate_df, ["year", "farm", "field_name"], "climate_"
        )
        merged = merged.merge(
            climate_agg,
            on=["year", "farm", "field_name"],
            how="left",
        )
        matched = merged["climate_MAT"].notna().sum() if "climate_MAT" in merged.columns else 0
        print(f"  Merged Climate: {matched}/{len(merged)} rows matched")

    # SoilGrids (static, keyed by farm + field_name, prefixed sg_)
    soilgrids_df = load_soilgrids_data()
    if not soilgrids_df.empty:
        print(f"  SoilGrids: {len(soilgrids_df)} rows")
        sg_agg = _aggregate_to_field_level(
            soilgrids_df, ["farm", "field_name"], "sg_"
        )
        merged = merged.merge(
            sg_agg,
            on=["farm", "field_name"],
            how="left",
        )
        # Check a representative column
        check_col = "sg_sand_pct_0_5cm"
        matched = merged[check_col].notna().sum() if check_col in merged.columns else 0
        print(f"  Merged SoilGrids: {matched}/{len(merged)} rows matched")

    return merged


def _print_coverage_report(df: pd.DataFrame) -> None:
    """Print a coverage report for key feature groups."""
    total = len(df)

    groups = {
        "Soil (ph)": "ph",
        "Soil (hu)": "hu",
        "SOC": "soc",
        "Soil (no3)": "no3",
        "S2 (NDVI)": "s2_NDVI_summer",
        "L8 (NDVI)": "l8_NDVI_summer",
        "S1 (VV)": "s1_VV",
        "Topo (DEM)": "topo_DEM",
        "Climate (MAT)": "climate_MAT",
        "SG sand": "sg_sand_pct_0_5cm",
        "SG clay": "sg_clay_pct_0_5cm",
    }

    print("\n  Feature coverage:")
    for label, col in groups.items():
        if col in df.columns:
            filled = pd.to_numeric(df[col], errors="coerce").notna().sum()
            pct = filled / total * 100
            bar = "#" * int(pct / 5)
            print(f"    {label:>16s}: {filled:4d}/{total} ({pct:5.1f}%) {bar}")
        else:
            print(f"    {label:>16s}: -- column missing --")


def main() -> None:
    """Main execution: merge all features and save to database."""
    print("="*60)
    print("Feature Merging and Database Integration")
    print("="*60)

    # Create output directory
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================
    # 1. Generate field-level static features
    # ========================================
    static_df = merge_static_features()

    print(f"\nStatic features dataset shape: {static_df.shape}")
    print(f"  Rows: {len(static_df)}")
    print(f"  Columns: {len(static_df.columns)}")

    # Save static features to CSV (no destructive transformations)
    static_csv_path = config.FEATURES_DIR / "field_static_features.csv"
    static_df.to_csv(static_csv_path, index=False)
    print(f"Saved CSV: {static_csv_path.name}")

    # Save static features to Excel
    static_excel_path = config.FEATURES_DIR / "field_static_features.xlsx"
    static_df.to_excel(static_excel_path, index=False, engine='openpyxl')
    print(f"Saved Excel: {static_excel_path.name}")

    # ========================================
    # 2. Generate full merged dataset
    # ========================================
    merged_df = merge_all_features()

    print(f"\nFull dataset shape: {merged_df.shape}")
    print(f"  Rows: {len(merged_df)}")
    print(f"  Columns: {len(merged_df.columns)}")

    _print_coverage_report(merged_df)

    # Save to CSV (preserve NaN as empty, do NOT replace 0.0)
    csv_path = config.FEATURES_DIR / "full_dataset.csv"
    merged_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path.name}")

    # Save to Excel
    excel_path = config.FEATURES_DIR / "full_dataset.xlsx"
    merged_df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"Saved Excel: {excel_path.name}")

    # Save to database
    print("\nSaving to database...")

    temp_df = load_temperature_data()
    if not temp_df.empty:
        save_features_to_db("temperature_profile", temp_df)

    s2_df = load_sentinel2_data()
    if not s2_df.empty:
        save_features_to_db("sentinel2_features", s2_df)

    l8_df = load_landsat8_data()
    if not l8_df.empty:
        save_features_to_db("landsat8_features", l8_df)

    s1_df = load_sentinel1_data()
    if not s1_df.empty:
        save_features_to_db("sentinel1_features", s1_df)

    topo_df = load_topography_data()
    if not topo_df.empty:
        save_features_to_db("topography_features", topo_df)

    climate_df = load_climate_data()
    if not climate_df.empty:
        save_features_to_db("climate_features", climate_df)

    soilgrids_df = load_soilgrids_data()
    if not soilgrids_df.empty:
        save_features_to_db("soilgrids_features", soilgrids_df)

    # Save merged dataset
    save_features_to_db("merged_dataset", merged_df)

    print("\n" + "="*60)
    print("Feature merging complete!")
    print(f"Final dataset: {len(merged_df)} rows x {len(merged_df.columns)} columns")
    print("="*60)


if __name__ == "__main__":
    main()
