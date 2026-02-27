"""
Descriptive statistics & distribution tests.

Reproduces Table 1 of the article and verifies:
- Summary statistics (mean, median, SD, CV, skewness, kurtosis)
- Shapiro-Wilk normality test for each soil property
- Kruskal-Wallis test for year-to-year differences
"""

import pandas as pd
import numpy as np
from scipy import stats

from .config import SOIL_TARGETS, SOIL_LABELS, ALPHA, OUTPUT_DIR


def compute_descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Table 1: descriptive statistics for soil targets."""
    rows = []
    for col in SOIL_TARGETS:
        s = df[col].dropna()
        rows.append({
            "Property": SOIL_LABELS[col],
            "n": len(s),
            "Mean": round(s.mean(), 2),
            "Median": round(s.median(), 2),
            "SD": round(s.std(), 2),
            "Min": round(s.min(), 2),
            "Max": round(s.max(), 2),
            "CV_%": round(s.std() / abs(s.mean()) * 100, 1) if s.mean() > 0 else np.nan,
            "Skewness": round(s.skew(), 2),
            "Kurtosis": round(s.kurtosis(), 2),
        })
    return pd.DataFrame(rows)


def shapiro_wilk_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Shapiro-Wilk normality test for each soil property.

    Article claims: all 6 properties reject normality at p < 0.001.
    Note: Shapiro-Wilk has n <= 5000 limit; we subsample if necessary.
    """
    rows = []
    for col in SOIL_TARGETS:
        s = df[col].dropna()
        # Shapiro-Wilk limited to 5000 samples
        if len(s) > 5000:
            s = s.sample(5000, random_state=42)
        stat, p = stats.shapiro(s)
        rows.append({
            "Property": SOIL_LABELS[col],
            "n": len(df[col].dropna()),
            "W_statistic": round(stat, 4),
            "p_value": p,
            "p_formatted": f"{p:.2e}" if p < 0.001 else f"{p:.4f}",
            "Normal_at_005": p >= ALPHA,
            "Article_claim_non_normal": True,  # article says all p < 0.001
            "VERIFIED": p < 0.001,
        })
    return pd.DataFrame(rows)


def kruskal_wallis_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Kruskal-Wallis test: are soil properties significantly different across years?

    Article claims: all properties show p < 0.001 across years.
    """
    rows = []
    for col in SOIL_TARGETS:
        groups = []
        year_ns = {}
        for year in sorted(df["year"].unique()):
            g = df.loc[df["year"] == year, col].dropna()
            if len(g) > 0:
                groups.append(g)
                year_ns[year] = len(g)

        if len(groups) < 2:
            rows.append({
                "Property": SOIL_LABELS[col],
                "year_sample_sizes": str(year_ns),
                "H_statistic": np.nan,
                "p_value": np.nan,
                "p_formatted": "N/A",
                "Significant_at_005": False,
                "Article_claim_p_lt_001": True,
                "VERIFIED": False,
            })
            continue

        h_stat, p = stats.kruskal(*groups)
        rows.append({
            "Property": SOIL_LABELS[col],
            "year_sample_sizes": str(year_ns),
            "H_statistic": round(h_stat, 2),
            "p_value": p,
            "p_formatted": f"{p:.2e}" if p < 0.001 else f"{p:.4f}",
            "Significant_at_005": p < ALPHA,
            "Article_claim_p_lt_001": True,
            "VERIFIED": p < 0.001,
        })
    return pd.DataFrame(rows)


def descriptive_stats_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year descriptive statistics to check for temporal bias."""
    rows = []
    for year in sorted(df["year"].unique()):
        sub = df[df["year"] == year]
        for col in SOIL_TARGETS:
            s = sub[col].dropna()
            if len(s) == 0:
                continue
            rows.append({
                "Year": year,
                "Property": SOIL_LABELS[col],
                "n": len(s),
                "Mean": round(s.mean(), 2),
                "Median": round(s.median(), 2),
                "SD": round(s.std(), 2),
            })
    return pd.DataFrame(rows)


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run all descriptive statistics analyses."""
    results = {
        "table1_descriptive": compute_descriptive_table(df),
        "shapiro_wilk": shapiro_wilk_tests(df),
        "kruskal_wallis_year": kruskal_wallis_by_year(df),
        "descriptive_by_year": descriptive_stats_by_year(df),
    }

    # Save to Excel
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_DIR / "descriptive_stats.xlsx") as writer:
        for name, tbl in results.items():
            tbl.to_excel(writer, sheet_name=name, index=False)

    return results
