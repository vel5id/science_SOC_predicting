"""
Full Spearman correlation analysis: soil properties vs all RS features.

Key verifications:
- pH ↔ L8 GNDVI spring:  ρ = -0.67
- pH ↔ MAP:              ρ = 0.66
- pH ↔ slope:            ρ = 0.55
- K  ↔ BSI spring:       ρ = -0.48
- P  ↔ GS_temp:          ρ = 0.48
- P  ↔ aspect_cos:       ρ = 0.47
- pH ↔ aspect_sin:       ρ = -0.47
- pH ↔ S2 GNDVI spring:  ρ ≈ -0.49..-0.52  (article: range for veg indices)
- SOC ↔ summer NDVI:     ρ ≈ 0.20-0.30

Also applies Benjamini-Hochberg FDR correction.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import (
    SOIL_TARGETS, SOIL_LABELS, ARTICLE_CLAIMS, ALPHA, OUTPUT_DIR,
    TOPO_COLS, CLIMATE_COLS, SEASONS,
)


def _get_rs_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all remote sensing / covariate feature columns."""
    exclude = {"id", "year", "farm", "field_name", "grid_id",
               "centroid_lon", "centroid_lat", "geometry_wkt",
               "protocol_number", "analysis_date", "sampling_date",
               "hu"}  # hu is the raw humus, soc is derived
    return [c for c in df.columns
            if c not in exclude and c not in SOIL_TARGETS
            and df[c].dtype in ("float64", "int64")]


def compute_all_spearman(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations: each soil target vs each RS feature.

    Returns long-form DataFrame with columns:
      target, feature, rho, p_value, n, abs_rho
    """
    features = _get_rs_feature_columns(df)
    rows = []
    for target in SOIL_TARGETS:
        for feat in features:
            mask = df[[target, feat]].notna().all(axis=1)
            n = mask.sum()
            if n < 10:
                continue
            rho, p = stats.spearmanr(df.loc[mask, target], df.loc[mask, feat])
            rows.append({
                "target": target,
                "target_label": SOIL_LABELS[target],
                "feature": feat,
                "rho": rho,
                "abs_rho": abs(rho),
                "p_value": p,
                "n": n,
            })
    result = pd.DataFrame(rows)
    return result


def apply_bh_correction(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction per target."""
    df = corr_df.copy()
    df["p_adjusted"] = np.nan
    df["significant_bh"] = False

    for target in SOIL_TARGETS:
        mask = df["target"] == target
        pvals = df.loc[mask, "p_value"].values
        if len(pvals) == 0:
            continue
        reject, p_adj, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
        df.loc[mask, "p_adjusted"] = p_adj
        df.loc[mask, "significant_bh"] = reject

    return df


def top_correlations(corr_df: pd.DataFrame, n_top: int = 20) -> pd.DataFrame:
    """Top-N strongest correlations per soil target."""
    frames = []
    for target in SOIL_TARGETS:
        sub = corr_df[corr_df["target"] == target].nlargest(n_top, "abs_rho")
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def verify_article_claims(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Verify specific rho values stated in the article."""
    rows = []
    for claim_id, info in ARTICLE_CLAIMS.items():
        target, feature, article_rho = info["target"], info["feature"], info["rho"]

        # For inter-soil correlations, skip (handled in intercorrelation.py)
        if feature in SOIL_TARGETS:
            continue

        match = corr_df[(corr_df["target"] == target) & (corr_df["feature"] == feature)]
        if match.empty:
            rows.append({
                "claim": claim_id,
                "target": target,
                "feature": feature,
                "article_rho": article_rho,
                "computed_rho": np.nan,
                "difference": np.nan,
                "p_value": np.nan,
                "n": 0,
                "MATCH_within_0.05": False,
                "NOTE": "Feature not found in dataset",
            })
            continue

        row = match.iloc[0]
        diff = abs(row["rho"] - article_rho)
        rows.append({
            "claim": claim_id,
            "target": target,
            "feature": feature,
            "article_rho": article_rho,
            "computed_rho": round(row["rho"], 4),
            "difference": round(diff, 4),
            "p_value": row["p_value"],
            "n": row["n"],
            "MATCH_within_0.05": diff < 0.05,
            "NOTE": "",
        })
    return pd.DataFrame(rows)


def seasonal_comparison(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Compare spring vs summer correlations for vegetation indices.

    Article claims: spring NDVI/GNDVI/NDRE correlate stronger with pH than summer.
    """
    indices = ["NDVI", "GNDVI", "NDRE", "EVI", "SAVI"]
    rows = []
    for idx in indices:
        for target in SOIL_TARGETS:
            for prefix in ["s2_", "l8_"]:
                season_vals = {}
                for season in SEASONS:
                    feat = f"{prefix}{idx}_{season}"
                    match = corr_df[(corr_df["target"] == target) & (corr_df["feature"] == feat)]
                    if not match.empty:
                        season_vals[season] = match.iloc[0]["rho"]

                if len(season_vals) >= 2:
                    rows.append({
                        "target": target,
                        "index": f"{prefix}{idx}",
                        **{f"rho_{s}": round(v, 4) for s, v in season_vals.items()},
                        "spring_stronger_than_summer": (
                            abs(season_vals.get("spring", 0)) > abs(season_vals.get("summer", 0))
                            if "spring" in season_vals and "summer" in season_vals
                            else None
                        ),
                    })
    return pd.DataFrame(rows)


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run the full correlation analysis."""
    all_corr = compute_all_spearman(df)
    all_corr = apply_bh_correction(all_corr)
    top = top_correlations(all_corr)
    claims = verify_article_claims(all_corr)
    seasonal = seasonal_comparison(all_corr)

    results = {
        "all_correlations": all_corr,
        "top_correlations": top,
        "article_claims_verification": claims,
        "seasonal_comparison": seasonal,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_DIR / "correlation_analysis.xlsx") as writer:
        top.to_excel(writer, sheet_name="top20_per_target", index=False)
        claims.to_excel(writer, sheet_name="claims_verification", index=False)
        seasonal.to_excel(writer, sheet_name="seasonal_comparison", index=False)
        # Full matrix too large for one sheet — save as CSV
    all_corr.to_csv(OUTPUT_DIR / "all_spearman_correlations.csv", index=False)

    return results
