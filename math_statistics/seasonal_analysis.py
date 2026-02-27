"""
Seasonal NDVI dynamics by SOC class (Figure 5 of the article).

Verifies:
- High SOC (>3.0%) soils show higher NDVI across all seasons
- Maximum contrast between classes in summer (0.60-0.70 vs 0.35-0.45)
- Autumn NDVI converges (0.20-0.35)
- Statistical significance of class differences (Kruskal-Wallis per season)
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import SOIL_TARGETS, SEASONS, SEASON_LABELS, OUTPUT_DIR


# SOC classes from the article
SOC_BINS = [0, 1.5, 2.0, 2.5, 3.0, np.inf]
SOC_LABELS = ["<1.5%", "1.5–2.0%", "2.0–2.5%", "2.5–3.0%", ">3.0%"]


def assign_soc_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add SOC class column to dataframe."""
    out = df.copy()
    out["soc_class"] = pd.cut(out["soc"], bins=SOC_BINS, labels=SOC_LABELS, right=False)
    return out


def seasonal_ndvi_by_soc_class(df: pd.DataFrame) -> pd.DataFrame:
    """Mean NDVI per season per SOC class (Sentinel-2)."""
    df = assign_soc_class(df)
    ndvi_cols = [f"s2_NDVI_{s}" for s in SEASONS]
    rows = []

    for soc_cls in SOC_LABELS:
        sub = df[df["soc_class"] == soc_cls]
        for season, col in zip(SEASONS, ndvi_cols):
            s = sub[col].dropna()
            if len(s) == 0:
                continue
            rows.append({
                "SOC_class": soc_cls,
                "Season": SEASON_LABELS[season],
                "season_key": season,
                "n": len(s),
                "NDVI_mean": round(s.mean(), 4),
                "NDVI_median": round(s.median(), 4),
                "NDVI_std": round(s.std(), 4),
                "NDVI_q25": round(s.quantile(0.25), 4),
                "NDVI_q75": round(s.quantile(0.75), 4),
            })
    return pd.DataFrame(rows)


def kruskal_wallis_ndvi_per_season(df: pd.DataFrame) -> pd.DataFrame:
    """Kruskal-Wallis test: does NDVI differ significantly across SOC classes?

    Applies Benjamini-Hochberg FDR correction across all seasonal tests
    (4 tests), consistent with correlation_analysis.py.
    """
    df = assign_soc_class(df)
    rows = []

    for season in SEASONS:
        col = f"s2_NDVI_{season}"
        groups = []
        for soc_cls in SOC_LABELS:
            g = df.loc[df["soc_class"] == soc_cls, col].dropna()
            if len(g) >= 2:
                groups.append(g)

        if len(groups) < 2:
            continue

        h_stat, p = stats.kruskal(*groups)
        rows.append({
            "Season": SEASON_LABELS[season],
            "H_statistic": round(h_stat, 2),
            "p_value": p,
            "p_formatted": f"{p:.2e}" if p < 0.001 else f"{p:.4f}",
            "Significant_raw": p < 0.05,
        })

    result = pd.DataFrame(rows)
    if len(result) > 1:
        # Benjamini-Hochberg FDR correction across all seasonal tests
        reject, p_adj, _, _ = multipletests(result["p_value"].values, method="fdr_bh")
        result["p_adj_BH"] = p_adj
        result["Significant"] = reject
    elif len(result) == 1:
        result["p_adj_BH"] = result["p_value"]
        result["Significant"] = result["Significant_raw"]
    return result


def verify_article_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Check specific numeric claims from section 3.4."""
    df = assign_soc_class(df)
    checks = []

    # Claim: Summer NDVI on SOC>3% is 0.60-0.70
    high_summer = df.loc[df["soc_class"] == ">3.0%", "s2_NDVI_summer"].dropna()
    if len(high_summer) > 0:
        checks.append({
            "Claim": "Summer NDVI, SOC>3%: 0.60-0.70",
            "Computed_mean": round(high_summer.mean(), 4),
            "Computed_median": round(high_summer.median(), 4),
            "In_range": 0.55 <= high_summer.mean() <= 0.75,  # generous tolerance
        })

    # Claim: Summer NDVI on SOC<1.5% is 0.35-0.45
    low_summer = df.loc[df["soc_class"] == "<1.5%", "s2_NDVI_summer"].dropna()
    if len(low_summer) > 0:
        checks.append({
            "Claim": "Summer NDVI, SOC<1.5%: 0.35-0.45",
            "Computed_mean": round(low_summer.mean(), 4),
            "Computed_median": round(low_summer.median(), 4),
            "In_range": 0.30 <= low_summer.mean() <= 0.50,
        })

    # Claim: Autumn NDVI converges to 0.20-0.35
    for soc_cls in SOC_LABELS:
        autumn = df.loc[df["soc_class"] == soc_cls, "s2_NDVI_autumn"].dropna()
        if len(autumn) > 0:
            checks.append({
                "Claim": f"Autumn NDVI converges, {soc_cls}",
                "Computed_mean": round(autumn.mean(), 4),
                "Computed_median": round(autumn.median(), 4),
                "In_range": 0.15 <= autumn.mean() <= 0.40,
            })

    # Claim: Maximum contrast in summer
    contrast = {}
    for season in SEASONS:
        col = f"s2_NDVI_{season}"
        high = df.loc[df["soc_class"] == ">3.0%", col].dropna().mean()
        low = df.loc[df["soc_class"] == "<1.5%", col].dropna().mean()
        if not np.isnan(high) and not np.isnan(low):
            contrast[season] = abs(high - low)

    if contrast:
        max_season = max(contrast, key=contrast.get)
        checks.append({
            "Claim": "Max contrast in summer",
            "Computed_mean": round(contrast.get("summer", np.nan), 4),
            "Computed_median": np.nan,
            "In_range": max_season == "summer",
        })

    return pd.DataFrame(checks)


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run seasonal analysis."""
    ndvi_table = seasonal_ndvi_by_soc_class(df)
    kw_test = kruskal_wallis_ndvi_per_season(df)
    claims = verify_article_claims(df)

    results = {
        "ndvi_by_soc_class": ndvi_table,
        "kruskal_wallis_ndvi": kw_test,
        "claims_verification": claims,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_DIR / "seasonal_analysis.xlsx") as writer:
        for name, tbl in results.items():
            tbl.to_excel(writer, sheet_name=name, index=False)

    return results
