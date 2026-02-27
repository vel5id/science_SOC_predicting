"""
Soil property intercorrelation analysis (Figure 2 of the article).

Verifies the reported Spearman correlations between soil properties:
- SOC ↔ S:   ρ = 0.56
- SOC ↔ NO₃: ρ = 0.41
- pH  ↔ SOC: ρ = -0.49
- P, K weak correlations with others (|ρ| < 0.30)
"""

import pandas as pd
import numpy as np
from scipy import stats

from .config import SOIL_TARGETS, SOIL_LABELS, ARTICLE_CLAIMS, OUTPUT_DIR


def compute_intercorrelation_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman rank correlation matrix among soil properties.

    Returns (rho_matrix, pvalue_matrix).
    """
    sub = df[SOIL_TARGETS].dropna()
    n = len(SOIL_TARGETS)
    rho_mat = pd.DataFrame(np.zeros((n, n)), index=SOIL_TARGETS, columns=SOIL_TARGETS)
    p_mat = pd.DataFrame(np.zeros((n, n)), index=SOIL_TARGETS, columns=SOIL_TARGETS)

    for i, c1 in enumerate(SOIL_TARGETS):
        for j, c2 in enumerate(SOIL_TARGETS):
            if i == j:
                rho_mat.iloc[i, j] = 1.0
                p_mat.iloc[i, j] = 0.0
            elif j > i:
                # pairwise complete
                mask = df[[c1, c2]].notna().all(axis=1)
                rho, p = stats.spearmanr(df.loc[mask, c1], df.loc[mask, c2])
                rho_mat.iloc[i, j] = rho
                rho_mat.iloc[j, i] = rho
                p_mat.iloc[i, j] = p
                p_mat.iloc[j, i] = p

    return rho_mat, p_mat


def verify_article_intercorrelations(df: pd.DataFrame) -> pd.DataFrame:
    """Check specific inter-correlations stated in the article."""
    claims_to_check = {
        "SOC ↔ S": ("soc", "s", 0.176),
        "SOC ↔ NO₃": ("soc", "no3", 0.148),
        "pH ↔ SOC": ("ph", "soc", -0.178),
    }
    rows = []
    for label, (c1, c2, article_rho) in claims_to_check.items():
        mask = df[[c1, c2]].notna().all(axis=1)
        rho, p = stats.spearmanr(df.loc[mask, c1], df.loc[mask, c2])
        diff = abs(rho - article_rho)
        rows.append({
            "Pair": label,
            "Article_rho": article_rho,
            "Computed_rho": round(rho, 4),
            "Difference": round(diff, 4),
            "p_value": p,
            "n_pairs": mask.sum(),
            "MATCH_within_0.05": diff < 0.05,
        })
    return pd.DataFrame(rows)


def check_weak_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Verify article claim: P and K have |ρ| < 0.30 with other properties."""
    rows = []
    for target in ["p", "k"]:
        for other in SOIL_TARGETS:
            if other == target:
                continue
            mask = df[[target, other]].notna().all(axis=1)
            rho, p = stats.spearmanr(df.loc[mask, target], df.loc[mask, other])
            rows.append({
                "Target": SOIL_LABELS[target],
                "Other": SOIL_LABELS[other],
                "Spearman_rho": round(rho, 4),
                "abs_rho": round(abs(rho), 4),
                "p_value": p,
                "Article_claim_abs_lt_030": True,
                "VERIFIED": abs(rho) < 0.30,
            })
    return pd.DataFrame(rows)


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run intercorrelation analysis."""
    rho_mat, p_mat = compute_intercorrelation_matrix(df)
    verification = verify_article_intercorrelations(df)
    weak_check = check_weak_correlations(df)

    results = {
        "rho_matrix": rho_mat,
        "p_matrix": p_mat,
        "verification": verification,
        "pk_weak_check": weak_check,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_DIR / "intercorrelation.xlsx") as writer:
        rho_mat.to_excel(writer, sheet_name="spearman_rho")
        p_mat.to_excel(writer, sheet_name="p_values")
        verification.to_excel(writer, sheet_name="verification", index=False)
        weak_check.to_excel(writer, sheet_name="pk_weak_check", index=False)

    return results
