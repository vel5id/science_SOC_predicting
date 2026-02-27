"""
Spatial autocorrelation analysis (Moran's I).

Checks whether soil properties exhibit spatial autocorrelation,
which is critical for validating the article's use of spatial block CV.
Also verifies the spatial distribution patterns described in section 3.7.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from .config import SOIL_TARGETS, SOIL_LABELS, OUTPUT_DIR


def _inverse_distance_weights(coords: np.ndarray, bandwidth: float = 50_000) -> np.ndarray:
    """Compute inverse-distance weight matrix (meters).

    bandwidth: distance beyond which weight = 0 (in meters-like units).
    Coords assumed to be in decimal degrees — rough conversion: 1° ≈ 100km.
    """
    # Convert degrees to approximate meters for distance.
    # Longitude scaling uses each point's own latitude (not mean) to avoid
    # systematic distortion across wide latitude ranges (e.g. 45-55N in Kazakhstan).
    coords_m = coords.copy()
    coords_m[:, 0] *= 111_320 * np.cos(np.radians(coords[:, 1]))
    coords_m[:, 1] *= 110_540

    dist = squareform(pdist(coords_m))
    np.fill_diagonal(dist, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where((dist < bandwidth) & (dist > 0), 1.0 / dist, 0.0)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0)
    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_row_std = W / row_sums
    # Return both: row-standardized (for I computation) and raw (for S1/S2 variance)
    return W_row_std, W


def morans_i(values: np.ndarray, W: np.ndarray, W_raw: np.ndarray = None) -> dict:
    """Compute Moran's I statistic with z-test.

    I = (n / S0) * (z' W z) / (z' z)  — uses row-standardized W.
    S1, S2 variance components use raw (non-standardized) W to avoid
    bias from row-standardization (Cliff & Ord, 1981).
    """
    n = len(values)
    z = values - values.mean()
    S0 = W.sum()

    if S0 == 0 or np.all(z == 0):
        return {"I": np.nan, "E_I": np.nan, "z_score": np.nan, "p_value": np.nan}

    numerator = n * (z @ W @ z)
    denominator = S0 * (z @ z)
    I = numerator / denominator if denominator != 0 else np.nan

    E_I = -1.0 / (n - 1)

    # Variance under normality assumption using raw weights (Cliff & Ord, 1981).
    # Row-standardized W gives biased S1/S2 — use original W_raw if available.
    W_v = W_raw if W_raw is not None else W
    S0_v = W_v.sum()
    S1 = 0.5 * ((W_v + W_v.T) ** 2).sum()
    S2 = ((W_v.sum(axis=1) + W_v.sum(axis=0)) ** 2).sum()
    A = n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * S0_v**2)
    D = (z**4).sum() / ((z**2).sum() ** 2 / n)  # kurtosis
    B = D * ((n**2 - n) * S1 - 2 * n * S2 + 6 * S0_v**2)
    C = (n - 1) * (n - 2) * (n - 3) * S0_v**2
    var_I = (A - B) / C - E_I**2 if C != 0 else np.nan

    if var_I is not None and var_I > 0:
        z_score = (I - E_I) / np.sqrt(var_I)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        z_score = np.nan
        p_value = np.nan

    return {
        "I": I,
        "E_I": E_I,
        "Var_I": var_I,
        "z_score": z_score,
        "p_value": p_value,
    }


def compute_morans_i_all(df: pd.DataFrame, max_samples: int = 2000) -> pd.DataFrame:
    """Moran's I for all soil targets.

    Subsamples to max_samples for computational tractability (O(n²) weights).
    """
    coords_cols = ["centroid_lon", "centroid_lat"]
    valid = df[coords_cols + SOIL_TARGETS].dropna(subset=coords_cols)

    if len(valid) > max_samples:
        valid = valid.sample(max_samples, random_state=42)

    # Reset index so positional alignment works cleanly
    valid = valid.reset_index(drop=True)
    coords = valid[coords_cols].values
    W, W_raw = _inverse_distance_weights(coords)

    rows = []
    for col in SOIL_TARGETS:
        not_null = valid[col].notna()
        mask_arr = not_null.to_numpy()
        vals = valid.loc[mask_arr, col].values
        w_sub = W[np.ix_(mask_arr, mask_arr)]
        w_raw_sub = W_raw[np.ix_(mask_arr, mask_arr)]

        result = morans_i(vals, w_sub, w_raw_sub)
        rows.append({
            "Property": SOIL_LABELS[col],
            "n": len(vals),
            "Morans_I": round(result["I"], 4) if not np.isnan(result["I"]) else np.nan,
            "Expected_I": round(result["E_I"], 6),
            "z_score": round(result["z_score"], 2) if not np.isnan(result["z_score"]) else np.nan,
            "p_value": result["p_value"],
            "Spatial_autocorrelation": (
                "Positive" if result.get("I", 0) > 0 and result.get("p_value", 1) < 0.05
                else "Negative" if result.get("I", 0) < 0 and result.get("p_value", 1) < 0.05
                else "None"
            ),
        })
    return pd.DataFrame(rows)


def latitudinal_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """Check article claim: pH increases from north to south, SOC decreases.

    Section 3.7: pH shows latitudinal gradient.
    """
    valid = df[["centroid_lat", "ph", "soc"]].dropna()
    rows = []

    for col, label in [("ph", "pH"), ("soc", "SOC")]:
        rho, p = stats.spearmanr(valid["centroid_lat"], valid[col])
        rows.append({
            "Property": label,
            "rho_with_latitude": round(rho, 4),
            "p_value": p,
            "Direction": "increases northward" if rho > 0 else "decreases northward",
            # Article: pH increases southward (= negative rho with lat)
            # SOC decreases southward (= positive rho with lat)
        })
    return pd.DataFrame(rows)


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run spatial analysis."""
    moran = compute_morans_i_all(df)
    gradient = latitudinal_gradient(df)

    results = {
        "morans_i": moran,
        "latitudinal_gradient": gradient,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_DIR / "spatial_analysis.xlsx") as writer:
        for name, tbl in results.items():
            tbl.to_excel(writer, sheet_name=name, index=False)

    return results
