"""
Variance decomposition: between-field vs within-field (Section 3.6.1 of article v2).

Article claims:
- pH: 94% between-field, 9.8% within-field variance
  (these sum to >100% because of rounding / possible overlap)
- SOC: ~20% within-field variance (double that of pH ~10%)

This is a classic one-way ANOVA-type decomposition using field_name as the grouping variable.
"""

import pandas as pd
import numpy as np

from .config import SOIL_TARGETS, SOIL_LABELS, OUTPUT_DIR


def decompose_variance(df: pd.DataFrame, group_col: str = "field_name") -> pd.DataFrame:
    """One-way variance decomposition for each soil property.

    Total variance = Between-group variance + Within-group variance
    SS_total = SS_between + SS_within
    """
    rows = []
    for col in SOIL_TARGETS:
        valid = df[[group_col, col]].dropna()
        if valid.empty:
            continue

        grand_mean = valid[col].mean()
        ss_total = ((valid[col] - grand_mean) ** 2).sum()

        ss_between = 0
        ss_within = 0
        groups = valid.groupby(group_col)
        for _, group in groups:
            n_g = len(group)
            group_mean = group[col].mean()
            ss_between += n_g * (group_mean - grand_mean) ** 2
            ss_within += ((group[col] - group_mean) ** 2).sum()

        n_groups = groups.ngroups
        n_total = len(valid)

        # Percentages
        pct_between = (ss_between / ss_total * 100) if ss_total > 0 else 0
        pct_within = (ss_within / ss_total * 100) if ss_total > 0 else 0

        # Intraclass Correlation Coefficient (ICC)
        ms_between = ss_between / max(n_groups - 1, 1)
        ms_within = ss_within / max(n_total - n_groups, 1)
        n_mean = n_total / n_groups  # average group size
        icc = (ms_between - ms_within) / (ms_between + (n_mean - 1) * ms_within) if (ms_between + (n_mean - 1) * ms_within) > 0 else 0

        rows.append({
            "Property": SOIL_LABELS[col],
            "n_samples": n_total,
            "n_fields": n_groups,
            "SS_total": round(ss_total, 2),
            "SS_between": round(ss_between, 2),
            "SS_within": round(ss_within, 2),
            "Pct_between": round(pct_between, 1),
            "Pct_within": round(pct_within, 1),
            "ICC": round(icc, 4),
        })
    return pd.DataFrame(rows)


def verify_article_claims(decomp: pd.DataFrame) -> pd.DataFrame:
    """Verify specific article claims about variance decomposition."""
    checks = []

    # pH: 73.2% between-field
    ph_row = decomp[decomp["Property"] == SOIL_LABELS["ph"]]
    if not ph_row.empty:
        pct = ph_row.iloc[0]["Pct_between"]
        checks.append({
            "Claim": "pH: 73.2% between-field variance",
            "Article_value": 73.2,
            "Computed_value": pct,
            "Difference": round(abs(pct - 73.2), 1),
            "MATCH_within_5pct": abs(pct - 73.2) < 5,
        })

    # pH within-field: 26.8%
    if not ph_row.empty:
        pct_w = ph_row.iloc[0]["Pct_within"]
        checks.append({
            "Claim": "pH: 26.8% within-field variance",
            "Article_value": 26.8,
            "Computed_value": pct_w,
            "Difference": round(abs(pct_w - 26.8), 1),
            "MATCH_within_5pct": abs(pct_w - 26.8) < 5,
        })

    # SOC within-field ~43.3%
    soc_row = decomp[decomp["Property"] == SOIL_LABELS["soc"]]
    if not soc_row.empty:
        pct_soc = soc_row.iloc[0]["Pct_within"]
        checks.append({
            "Claim": "SOC within-field variance ~43.3%",
            "Article_value": 43.3,
            "Computed_value": pct_soc,
            "Difference": round(abs(pct_soc - 43.3), 1),
            "MATCH_within_5pct": abs(pct_soc - 43.3) < 5,
        })

    return pd.DataFrame(checks)


def run(df: pd.DataFrame) -> dict:
    """Run variance decomposition analysis."""
    decomp = decompose_variance(df)
    claims = verify_article_claims(decomp)

    results = {
        "decomposition": decomp,
        "claims_verification": claims,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_DIR / "variance_decomposition.xlsx") as writer:
        decomp.to_excel(writer, sheet_name="decomposition", index=False)
        claims.to_excel(writer, sheet_name="claims_verification", index=False)

    return results
