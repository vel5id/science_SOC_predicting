"""
Derived (compound) soil indicators (Section 2.5 of article v2).

Nine derived indicators reflecting functional soil processes:
  1. SOC/NO₃  (proxy C:N ratio — humification degree)
  2. SOC×NO₃  (organic nitrogen pool)
  3. S/NO₃    (S-N balance in mineralisation cycle)
  4. NO₃+0.5·S  (mineralisation index)
  5. P₂O₅/K₂O  (macronutrient balance)
  6. |pH − 7.0|  (deviation from neutrality)
  7. SOC×pH
  8. Normalised fertility index
  9. Sum of nutrients
"""

import pandas as pd
import numpy as np
from scipy import stats

from .config import SOIL_TARGETS, SOIL_LABELS, OUTPUT_DIR, ALPHA


DERIVED_DEFINITIONS = {
    "SOC_NO3_ratio": {
        "formula": "SOC / NO₃",
        "description": "Proxy C:N ratio (humification degree)",
    },
    "SOC_x_NO3": {
        "formula": "SOC × NO₃",
        "description": "Organic nitrogen pool",
    },
    "S_NO3_ratio": {
        "formula": "S / NO₃",
        "description": "S-N balance in mineralisation",
    },
    "mineral_index": {
        "formula": "NO₃ + 0.5·S",
        "description": "Mineralisation index",
    },
    "P_K_ratio": {
        "formula": "P₂O₅ / K₂O",
        "description": "Macronutrient balance",
    },
    "pH_deviation": {
        "formula": "|pH − 7.0|",
        "description": "Deviation from neutrality",
    },
    "SOC_x_pH": {
        "formula": "SOC × pH",
        "description": "SOC-pH interaction",
    },
    "fertility_index": {
        "formula": "norm(SOC) + norm(NO₃) + norm(P₂O₅) + norm(K₂O)",
        "description": "Normalised fertility index",
    },
    "nutrient_sum": {
        "formula": "NO₃ + P₂O₅ + K₂O + S",
        "description": "Sum of nutrients (mg/kg)",
    },
}

DERIVED_LABELS = {k: v["formula"] for k, v in DERIVED_DEFINITIONS.items()}


def compute_derived_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 9 derived soil indicators."""
    out = pd.DataFrame(index=df.index)

    out["SOC_NO3_ratio"] = df["soc"] / df["no3"].replace(0, np.nan)
    out["SOC_x_NO3"] = df["soc"] * df["no3"]
    out["S_NO3_ratio"] = df["s"] / df["no3"].replace(0, np.nan)
    out["mineral_index"] = df["no3"] + 0.5 * df["s"]
    out["P_K_ratio"] = df["p"] / df["k"].replace(0, np.nan)
    out["pH_deviation"] = (df["ph"] - 7.0).abs()
    out["SOC_x_pH"] = df["soc"] * df["ph"]

    # Normalised fertility: z-score each then sum available z-scores per row.
    # Fix #22: guard against std=0 (constant column) to avoid inf/NaN.
    # Fix #21: use nanmean across available z-scores so partial NaN rows are
    #          preserved rather than producing all-NaN fertility_index.
    z_cols = []
    for col in ["soc", "no3", "p", "k"]:
        s = df[col]
        std_val = s.std()
        if std_val > 1e-10:
            z = (s - s.mean()) / std_val
        else:
            z = pd.Series(0.0, index=df.index)
        z_col = f"_z_{col}"
        out[z_col] = z
        z_cols.append(z_col)
    # Sum available z-scores (NaN-safe): rows with partial data get partial sum
    out["fertility_index"] = out[z_cols].sum(axis=1, min_count=1)
    out.drop(columns=z_cols, inplace=True)

    out["nutrient_sum"] = df["no3"] + df["p"] + df["k"] + df["s"]

    return out


def correlate_derived_with_rs(df: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations of derived indicators with all RS features."""
    exclude = {"id", "year", "farm", "field_name", "grid_id",
               "centroid_lon", "centroid_lat", "geometry_wkt",
               "protocol_number", "analysis_date", "sampling_date", "hu"}
    rs_cols = [c for c in df.columns
               if c not in exclude and c not in SOIL_TARGETS
               and df[c].dtype in ("float64", "int64")]

    rows = []
    for dcol in derived.columns:
        for feat in rs_cols:
            mask = derived[dcol].notna() & df[feat].notna()
            n = mask.sum()
            if n < 10:
                continue
            rho, p = stats.spearmanr(derived.loc[mask, dcol], df.loc[mask, feat])
            rows.append({
                "derived": dcol,
                "derived_label": DERIVED_LABELS.get(dcol, dcol),
                "feature": feat,
                "rho": rho,
                "abs_rho": abs(rho),
                "p_value": p,
                "n": n,
            })
    return pd.DataFrame(rows)


def correlate_derived_with_composites(derived: pd.DataFrame,
                                       composites: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations of derived soil with composite spectral features."""
    rows = []
    for dcol in derived.columns:
        for ccol in composites.columns:
            mask = derived[dcol].notna() & composites[ccol].notna()
            n = mask.sum()
            if n < 10:
                continue
            rho, p = stats.spearmanr(derived.loc[mask, dcol], composites.loc[mask, ccol])
            rows.append({
                "derived": dcol,
                "composite": ccol,
                "rho": rho,
                "abs_rho": abs(rho),
                "p_value": p,
                "n": n,
            })
    return pd.DataFrame(rows)


def verify_article_claims(corr_rs: pd.DataFrame, corr_comp: pd.DataFrame) -> pd.DataFrame:
    """Verify specific article v2 claims about derived indicators."""
    claims = [
        ("P_K_ratio", "topo_slope", -0.56, "P₂O₅/K₂O → slope"),
        ("S_NO3_ratio", "mean_GNDVI", 0.49, "S/NO₃ → GNDVI mean-season"),
        ("SOC_x_NO3", "comp_GNDVI-NDRE_spring", -0.46, "SOC×NO₃ → GNDVI−NDRE spring"),
        ("mineral_index", "l8_SR_B5_spring", -0.47, "mineral_index → L8 NIR spring"),
    ]
    rows = []
    for derived_col, feat, article_rho, label in claims:
        # Search in RS correlations first
        match = corr_rs[(corr_rs["derived"] == derived_col) & (corr_rs["feature"] == feat)]
        if match.empty and corr_comp is not None:
            match = corr_comp[(corr_comp["derived"] == derived_col) & (corr_comp["composite"] == feat)]
            if not match.empty:
                match = match.rename(columns={"composite": "feature"})
        if match.empty:
            # Try partial match
            if corr_comp is not None:
                partial = corr_comp[corr_comp["derived"] == derived_col]
                partial = partial[partial["composite"].str.contains(
                    feat.replace("comp_", "").replace("mean_", ""),
                    case=False, na=False)]
                if not partial.empty:
                    match = partial.nlargest(1, "abs_rho").rename(columns={"composite": "feature"})

        if match.empty:
            rows.append({
                "Claim": label,
                "derived": derived_col,
                "feature": feat,
                "Article_rho": article_rho,
                "Computed_rho": np.nan,
                "Difference": np.nan,
                "MATCH_within_0.05": False,
                "NOTE": "Feature not found",
            })
        else:
            r = match.iloc[0]
            computed = r["rho"]
            diff = abs(computed - article_rho)
            rows.append({
                "Claim": label,
                "derived": derived_col,
                "feature": r["feature"],
                "Article_rho": article_rho,
                "Computed_rho": round(computed, 4),
                "Difference": round(diff, 4),
                "MATCH_within_0.05": diff < 0.05,
                "NOTE": "",
            })
    return pd.DataFrame(rows)


def top_derived_correlations(corr_df: pd.DataFrame, n_top: int = 5) -> pd.DataFrame:
    """Top N correlations for each derived indicator."""
    frames = []
    for dcol in corr_df["derived"].unique():
        sub = corr_df[corr_df["derived"] == dcol].nlargest(n_top, "abs_rho")
        frames.append(sub)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def run(df: pd.DataFrame, composites: pd.DataFrame = None) -> dict:
    """Run derived soil analysis."""
    derived = compute_derived_indicators(df)
    corr_rs = correlate_derived_with_rs(df, derived)
    corr_comp = None
    if composites is not None and composites.shape[1] > 0:
        corr_comp = correlate_derived_with_composites(derived, composites)
    claims = verify_article_claims(corr_rs, corr_comp)
    top_rs = top_derived_correlations(corr_rs)
    top_comp = top_derived_correlations(corr_comp) if corr_comp is not None else pd.DataFrame()

    results = {
        "derived": derived,
        "corr_with_rs": corr_rs,
        "corr_with_composites": corr_comp,
        "claims_verification": claims,
        "top_rs": top_rs,
        "top_composites": top_comp,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_DIR / "derived_soil.xlsx") as writer:
        claims.to_excel(writer, sheet_name="claims_verification", index=False)
        top_rs.to_excel(writer, sheet_name="top_derived_vs_rs", index=False)
        if not top_comp.empty:
            top_comp.to_excel(writer, sheet_name="top_derived_vs_composites", index=False)

    return results
