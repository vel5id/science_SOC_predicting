"""
Composite (compound) spectral feature computation (Section 2.4 of article v2).

Three types of composite features:
  (a) Inter-index combinations: pairwise ratios, products, differences
      of 7 vegetation indices × 4 seasons → 96 features
  (b) Multi-seasonal deltas: ΔNDVI(summer−spring), amplitude, seasonal mean → 42 features
  (c) Normalised band differences: (B5−B4)/(B5+B4), SWIR1/NIR, etc. → 10 features

Total: ~148 composite features.
"""

import pandas as pd
import numpy as np

from .config import SEASONS, OUTPUT_DIR


# Base S2 index names (without season suffix)
_S2_INDICES = ["NDVI", "GNDVI", "NDRE", "EVI", "SAVI", "BSI", "Cl_Red_Edge"]

# Spectral band columns: spectral_B*_season
_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def _col(prefix: str, name: str, season: str) -> str:
    """Build column name, trying both 's2_' and 'spectral_' prefixes."""
    return f"{prefix}{name}_{season}"


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Division with 0-protection."""
    return a / b.replace(0, np.nan)


def _safe_norm_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    """Normalised difference: (a-b)/(a+b)."""
    denom = a + b
    return (a - b) / denom.replace(0, np.nan)


def compute_inter_index_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """(a) Pairwise products, ratios, differences of S2 indices per season."""
    result = pd.DataFrame(index=df.index)
    pairs = [
        ("GNDVI", "BSI", "product"),
        ("NDVI", "BSI", "product"),
        ("EVI", "BSI", "product"),
        ("GNDVI", "NDRE", "diff"),
        ("NDVI", "NDRE", "diff"),
        ("EVI", "NDRE", "diff"),
        ("EVI", "NDRE", "ratio"),
        ("GNDVI", "NDVI", "ratio"),
        ("NDVI", "SAVI", "ratio"),
        ("SAVI", "BSI", "product"),
        ("NDVI", "Cl_Red_Edge", "product"),
        ("GNDVI", "Cl_Red_Edge", "diff"),
    ]
    for idx1, idx2, op in pairs:
        for season in SEASONS:
            # Try s2_ prefix first, fall back to spectral_
            for prefix in ["s2_", "spectral_"]:
                c1 = _col(prefix, idx1, season)
                c2 = _col(prefix, idx2, season)
                if c1 in df.columns and c2 in df.columns:
                    a, b = df[c1], df[c2]
                    if op == "product":
                        result[f"comp_{idx1}x{idx2}_{season}"] = a * b
                    elif op == "diff":
                        result[f"comp_{idx1}-{idx2}_{season}"] = a - b
                    elif op == "ratio":
                        result[f"comp_{idx1}/{idx2}_{season}"] = _safe_div(a, b)
                    break
    return result


def compute_multiseasonal_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """(b) Multi-seasonal deltas: Δ(summer−spring), amplitude, seasonal mean."""
    result = pd.DataFrame(index=df.index)
    season_pairs = [
        ("summer", "spring", "sum-spr"),
        ("late_summer", "spring", "ls-spr"),
        ("late_summer", "summer", "ls-sum"),
        ("autumn", "summer", "aut-sum"),
    ]
    for idx_name in _S2_INDICES:
        # Seasonal means and amplitude
        season_vals = {}
        for season in SEASONS:
            for prefix in ["s2_", "spectral_"]:
                c = _col(prefix, idx_name, season)
                if c in df.columns:
                    season_vals[season] = df[c]
                    break

        if len(season_vals) < 2:
            continue

        # Deltas
        for s1, s2, label in season_pairs:
            if s1 in season_vals and s2 in season_vals:
                result[f"delta_{idx_name}_{label}"] = season_vals[s1] - season_vals[s2]

        # Amplitude (max - min across seasons)
        all_vals = pd.concat(season_vals.values(), axis=1)
        result[f"amp_{idx_name}"] = all_vals.max(axis=1) - all_vals.min(axis=1)

        # Seasonal mean
        result[f"mean_{idx_name}"] = all_vals.mean(axis=1)

    return result


def compute_normalised_band_differences(df: pd.DataFrame) -> pd.DataFrame:
    """(c) Normalised band differences and ratios."""
    result = pd.DataFrame(index=df.index)
    combos = [
        ("B5", "B4", "norm_diff"),    # Red-edge NDI
        ("B11", "B12", "norm_diff"),   # SWIR NDI
        ("B11", "B8", "ratio"),        # SWIR1/NIR
        ("B6", "B5", "norm_diff"),     # Red-edge 2 NDI
        ("B8A", "B11", "norm_diff"),   # NIRn/SWIR1 NDI
    ]
    for b1, b2, op in combos:
        for season in SEASONS:
            for prefix in ["s2_", "spectral_"]:
                c1 = _col(prefix, b1, season)
                c2 = _col(prefix, b2, season)
                if c1 in df.columns and c2 in df.columns:
                    a, b = df[c1], df[c2]
                    if op == "norm_diff":
                        result[f"ndi_{b1}_{b2}_{season}"] = _safe_norm_diff(a, b)
                    elif op == "ratio":
                        result[f"ratio_{b1}_{b2}_{season}"] = _safe_div(a, b)
                    break
    return result


def compute_all_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 148 composite features and return concatenated."""
    inter = compute_inter_index_combinations(df)
    deltas = compute_multiseasonal_deltas(df)
    ndi = compute_normalised_band_differences(df)

    composites = pd.concat([inter, deltas, ndi], axis=1)
    return composites


def run(df: pd.DataFrame) -> dict:
    """Compute composites and report summary."""
    composites = compute_all_composites(df)
    summary = pd.DataFrame({
        "Type": ["Inter-index combinations", "Multi-seasonal deltas",
                 "Normalised band differences", "TOTAL"],
        "n_features": [
            len([c for c in composites.columns if c.startswith("comp_")]),
            len([c for c in composites.columns if c.startswith(("delta_", "amp_", "mean_"))]),
            len([c for c in composites.columns if c.startswith(("ndi_", "ratio_"))]),
            composites.shape[1],
        ],
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_excel(OUTPUT_DIR / "composite_features_summary.xlsx", index=False)

    return {"composites": composites, "summary": summary}
