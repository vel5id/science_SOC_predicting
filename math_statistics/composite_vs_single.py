"""
Composite vs Single feature comparison (Table 2 / Figure 5 of article v2).

Verifies article claims:
- GNDVI×BSI(spring) → K₂O: ρ = -0.488 vs BSI alone ρ = -0.478
- GNDVI−NDRE(spring) → NO₃: ρ = 0.416 vs L8 SAVI ρ = 0.419
- Multi-seasonal deltas do NOT outperform peak single-season features
"""

import pandas as pd
import numpy as np
from scipy import stats

from .config import SOIL_TARGETS, SOIL_LABELS, OUTPUT_DIR


def compare_composite_vs_single(df: pd.DataFrame,
                                 composites: pd.DataFrame,
                                 all_single_corr: pd.DataFrame) -> pd.DataFrame:
    """For each soil target, find best composite and best single, then compare."""
    rows = []
    for target in SOIL_TARGETS:
        # Best single (from precomputed all_single_corr)
        single_sub = all_single_corr[all_single_corr["target"] == target]
        if single_sub.empty:
            continue
        best_single = single_sub.loc[single_sub["abs_rho"].idxmax()]

        # Best composite
        best_comp_rho = 0
        best_comp_name = ""
        best_comp_p = 1
        best_comp_n = 0
        for ccol in composites.columns:
            mask = df[target].notna() & composites[ccol].notna()
            n = mask.sum()
            if n < 10:
                continue
            rho, p = stats.spearmanr(df.loc[mask, target], composites.loc[mask, ccol])
            if abs(rho) > abs(best_comp_rho):
                best_comp_rho = rho
                best_comp_name = ccol
                best_comp_p = p
                best_comp_n = n

        improvement = abs(best_comp_rho) - best_single["abs_rho"]
        rows.append({
            "Target": SOIL_LABELS[target],
            "Best_single": best_single["feature"],
            "Single_rho": round(best_single["rho"], 4),
            "Best_composite": best_comp_name,
            "Composite_rho": round(best_comp_rho, 4),
            "Improvement": round(improvement, 4),
            "Composite_wins": improvement > 0,
        })

    return pd.DataFrame(rows)


def verify_specific_claims(df: pd.DataFrame, composites: pd.DataFrame) -> pd.DataFrame:
    """Verify the specific composite feature claims in the article."""
    claims = []

    # Claim 1: GNDVI×BSI(spring) → K₂O: ρ = -0.488
    ccol = "comp_GNDVIxBSI_spring"
    if ccol in composites.columns:
        mask = df["k"].notna() & composites[ccol].notna()
        rho, p = stats.spearmanr(df.loc[mask, "k"], composites.loc[mask, ccol])
        claims.append({
            "Claim": "GNDVI×BSI(spring) → K₂O: ρ = -0.488",
            "Feature": ccol,
            "Article_rho": -0.488,
            "Computed_rho": round(rho, 4),
            "Difference": round(abs(rho - (-0.488)), 4),
            "MATCH": abs(rho - (-0.488)) < 0.05,
            "n": mask.sum(),
        })

    # Claim 2: GNDVI−NDRE(spring) → NO₃: ρ = -0.416
    ccol2 = "comp_GNDVI-NDRE_spring"
    if ccol2 in composites.columns:
        mask = df["no3"].notna() & composites[ccol2].notna()
        rho, p = stats.spearmanr(df.loc[mask, "no3"], composites.loc[mask, ccol2])
        claims.append({
            "Claim": "GNDVI−NDRE(spring) → NO₃: ρ = -0.416",
            "Feature": ccol2,
            "Article_rho": -0.416,
            "Computed_rho": round(rho, 4),
            "Difference": round(abs(rho - (-0.416)), 4),
            "MATCH": abs(rho - (-0.416)) < 0.05,
            "n": mask.sum(),
        })

    # Claim 3: BSI alone → K₂O: ρ = -0.478
    bsi_col = "s2_BSI_spring"
    if bsi_col in df.columns:
        mask = df[["k", bsi_col]].notna().all(axis=1)
        rho, p = stats.spearmanr(df.loc[mask, "k"], df.loc[mask, bsi_col])
        claims.append({
            "Claim": "BSI(spring) alone → K₂O: ρ = -0.478",
            "Feature": bsi_col,
            "Article_rho": -0.478,
            "Computed_rho": round(rho, 4),
            "Difference": round(abs(rho - (-0.478)), 4),
            "MATCH": abs(rho - (-0.478)) < 0.05,
            "n": mask.sum(),
        })

    # Claim 4: GNDVI×BSI > BSI alone for K₂O
    if len(claims) >= 2:
        gndvi_bsi_rho = next((c["Computed_rho"] for c in claims
                              if "GNDVI×BSI" in c["Claim"]), None)
        bsi_alone_rho = next((c["Computed_rho"] for c in claims
                              if "BSI(spring) alone" in c["Claim"]), None)
        if gndvi_bsi_rho is not None and bsi_alone_rho is not None:
            claims.append({
                "Claim": "GNDVI×BSI > BSI alone for K₂O",
                "Feature": "comparison",
                "Article_rho": np.nan,
                "Computed_rho": np.nan,
                "Difference": round(abs(gndvi_bsi_rho) - abs(bsi_alone_rho), 4),
                "MATCH": abs(gndvi_bsi_rho) > abs(bsi_alone_rho),
                "n": 0,
            })

    # Claim 5: EVI/NDRE(summer) → SOC: ρ = -0.177
    ccol5 = "comp_EVI/NDRE_summer"
    if ccol5 in composites.columns:
        mask = df["soc"].notna() & composites[ccol5].notna()
        if mask.sum() > 10:
            rho, p = stats.spearmanr(df.loc[mask, "soc"], composites.loc[mask, ccol5])
            claims.append({
                "Claim": "EVI/NDRE(summer) → SOC: ρ = -0.177",
                "Feature": ccol5,
                "Article_rho": -0.177,
                "Computed_rho": round(rho, 4),
                "Difference": round(abs(rho - (-0.177)), 4),
                "MATCH": abs(rho - (-0.177)) < 0.05,
                "n": mask.sum(),
            })

    return pd.DataFrame(claims)


def seasonal_delta_vs_peak(df: pd.DataFrame, composites: pd.DataFrame) -> pd.DataFrame:
    """Check: multi-seasonal deltas do NOT outperform peak single-season.

    Article claims deltas are weaker than peak seasonal values.
    """
    delta_cols = [c for c in composites.columns if c.startswith("delta_") or c.startswith("amp_")]
    rows = []

    for target in SOIL_TARGETS:
        # Best delta/amplitude
        best_delta_rho = 0
        best_delta_name = ""
        for dc in delta_cols:
            mask = df[target].notna() & composites[dc].notna()
            n = mask.sum()
            if n < 10:
                continue
            rho, _ = stats.spearmanr(df.loc[mask, target], composites.loc[mask, dc])
            if abs(rho) > abs(best_delta_rho):
                best_delta_rho = rho
                best_delta_name = dc

        # Best single-season index
        best_single_rho = 0
        best_single_name = ""
        for idx_name in ["NDVI", "GNDVI", "NDRE", "EVI", "SAVI", "BSI"]:
            for season in ["spring", "summer", "late_summer", "autumn"]:
                for prefix in ["s2_", "spectral_"]:
                    col = f"{prefix}{idx_name}_{season}"
                    if col not in df.columns:
                        continue
                    mask = df[[target, col]].notna().all(axis=1)
                    if mask.sum() < 10:
                        continue
                    rho, _ = stats.spearmanr(df.loc[mask, target], df.loc[mask, col])
                    if abs(rho) > abs(best_single_rho):
                        best_single_rho = rho
                        best_single_name = col

        rows.append({
            "Target": SOIL_LABELS[target],
            "Best_delta": best_delta_name,
            "Delta_rho": round(best_delta_rho, 4),
            "Best_single_season": best_single_name,
            "Single_rho": round(best_single_rho, 4),
            "Delta_weaker": abs(best_delta_rho) < abs(best_single_rho),
        })

    return pd.DataFrame(rows)


def run(df: pd.DataFrame, composites: pd.DataFrame,
        all_single_corr: pd.DataFrame) -> dict:
    """Run composite vs single comparison."""
    comparison = compare_composite_vs_single(df, composites, all_single_corr)
    claims = verify_specific_claims(df, composites)
    delta_check = seasonal_delta_vs_peak(df, composites)

    results = {
        "comparison_table": comparison,
        "claims_verification": claims,
        "delta_vs_peak": delta_check,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_DIR / "composite_vs_single.xlsx") as writer:
        comparison.to_excel(writer, sheet_name="comparison", index=False)
        claims.to_excel(writer, sheet_name="claims_verification", index=False)
        delta_check.to_excel(writer, sheet_name="delta_vs_peak", index=False)

    return results
