"""
Orchestrator: run all statistical analyses and generate a verification report.

Usage:
    python -m math_statistics.run_all
"""

import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from math_statistics.config import FEATURES_CSV, OUTPUT_DIR, SOIL_TARGETS
from math_statistics import (
    descriptive_stats,
    intercorrelation,
    correlation_analysis,
    seasonal_analysis,
    spatial_analysis,
    composite_features,
    derived_soil,
    variance_decomposition,
    confounding_analysis,
    composite_vs_single,
    plots,
)


def load_data() -> pd.DataFrame:
    """Load and validate the full dataset."""
    print(f"Loading data from {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)
    print(f"  Shape: {df.shape}")
    print(f"  Soil targets available: {[c for c in SOIL_TARGETS if c in df.columns]}")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Missing values in targets:")
    for col in SOIL_TARGETS:
        n_miss = df[col].isna().sum()
        print(f"    {col}: {n_miss} / {len(df)} ({n_miss/len(df)*100:.1f}%)")
    return df


def generate_text_report(all_results: dict) -> str:
    """Generate a human-readable verification report."""
    lines = [
        "=" * 80,
        "STATISTICAL VERIFICATION REPORT (Article v2)",
        "Article: Soil properties - RS correlations, Northern Kazakhstan",
        "Single and composite spectral indices",
        "=" * 80,
        "",
    ]

    # 1. Descriptive stats
    lines.append("1. DESCRIPTIVE STATISTICS (Table 1)")
    lines.append("-" * 60)
    desc = all_results.get("descriptive", {})
    if "table1_descriptive" in desc:
        lines.append(desc["table1_descriptive"].to_string(index=False))
    lines.append("")

    # 2. Normality
    lines.append("2. SHAPIRO-WILK NORMALITY TEST")
    lines.append("-" * 60)
    if "shapiro_wilk" in desc:
        sw = desc["shapiro_wilk"]
        all_verified = sw["VERIFIED"].all()
        lines.append(f"Article claims all properties non-normal (p<0.001): "
                      f"{'ALL VERIFIED' if all_verified else 'SOME FAILED'}")
        lines.append(sw[["Property", "W_statistic", "p_formatted", "VERIFIED"]].to_string(index=False))
    lines.append("")

    # 3. Kruskal-Wallis
    lines.append("3. KRUSKAL-WALLIS: YEAR DIFFERENCES")
    lines.append("-" * 60)
    if "kruskal_wallis_year" in desc:
        kw = desc["kruskal_wallis_year"]
        lines.append(kw[["Property", "H_statistic", "p_formatted", "VERIFIED"]].to_string(index=False))
    lines.append("")

    # 4. Intercorrelation
    lines.append("4. SOIL PROPERTY INTERCORRELATIONS (Figure 2)")
    lines.append("-" * 60)
    inter = all_results.get("intercorrelation", {})
    if "verification" in inter:
        lines.append(inter["verification"].to_string(index=False))
    lines.append("")

    # 5. Key article claims (single features)
    lines.append("5. SINGLE FEATURE CORRELATION CLAIMS (Section 3.3)")
    lines.append("-" * 60)
    corr = all_results.get("correlation", {})
    if "article_claims_verification" in corr:
        claims = corr["article_claims_verification"]
        lines.append(claims.to_string(index=False))
        n_match = claims["MATCH_within_0.05"].sum()
        n_total = len(claims)
        lines.append(f"\n  Matched within +/-0.05: {n_match}/{n_total}")
    lines.append("")

    # 6. Seasonal comparison
    lines.append("6. SEASONAL COMPARISON: SPRING vs SUMMER")
    lines.append("-" * 60)
    if "seasonal_comparison" in corr:
        sc = corr["seasonal_comparison"]
        ph_rows = sc[sc["target"] == "ph"]
        if not ph_rows.empty:
            spring_stronger = ph_rows["spring_stronger_than_summer"].sum()
            lines.append(f"  For pH: spring stronger than summer in {spring_stronger}/{len(ph_rows)} indices")
    lines.append("")

    # 7. Composite features
    lines.append("7. COMPOSITE SPECTRAL FEATURES (Section 3.4, Table 2)")
    lines.append("-" * 60)
    comp = all_results.get("composite_features", {})
    if "summary" in comp:
        lines.append("Feature counts:")
        lines.append(comp["summary"].to_string(index=False))
    lines.append("")

    comp_vs = all_results.get("composite_vs_single", {})
    if "comparison_table" in comp_vs:
        lines.append("Best single vs best composite per target:")
        lines.append(comp_vs["comparison_table"].to_string(index=False))
    lines.append("")
    if "claims_verification" in comp_vs:
        lines.append("Specific composite claims:")
        lines.append(comp_vs["claims_verification"].to_string(index=False))
    lines.append("")
    if "delta_vs_peak" in comp_vs:
        lines.append("Seasonal delta vs peak single-season:")
        lines.append(comp_vs["delta_vs_peak"].to_string(index=False))
    lines.append("")

    # 8. Derived soil indicators
    lines.append("8. DERIVED SOIL INDICATORS (Section 3.5)")
    lines.append("-" * 60)
    derived = all_results.get("derived_soil", {})
    if "claims_verification" in derived:
        lines.append(derived["claims_verification"].to_string(index=False))
    lines.append("")
    if "top_rs" in derived and not derived["top_rs"].empty:
        lines.append("Top derived-soil correlations with RS:")
        top5 = derived["top_rs"].nlargest(10, "abs_rho")
        lines.append(top5[["derived", "feature", "rho", "p_value"]].to_string(index=False))
    lines.append("")

    # 9. Variance decomposition
    lines.append("9. VARIANCE DECOMPOSITION (Section 3.6.1)")
    lines.append("-" * 60)
    vd = all_results.get("variance_decomposition", {})
    if "decomposition" in vd:
        lines.append(vd["decomposition"].to_string(index=False))
    lines.append("")
    if "claims_verification" in vd:
        lines.append("Claims verification:")
        lines.append(vd["claims_verification"].to_string(index=False))
    lines.append("")

    # 10. Confounding analysis
    lines.append("10. pH-CONFOUNDING & SOC MECHANISMS (Section 3.6.2)")
    lines.append("-" * 60)
    conf = all_results.get("confounding", {})
    if "confounding_verification" in conf:
        lines.append(conf["confounding_verification"].to_string(index=False))
    lines.append("")
    if "cv_vs_rho" in conf:
        lines.append("CV vs max |rho|:")
        lines.append(conf["cv_vs_rho"].to_string(index=False))
    lines.append("")

    # 11. Spatial
    lines.append("11. SPATIAL AUTOCORRELATION (MORAN'S I)")
    lines.append("-" * 60)
    spat = all_results.get("spatial", {})
    if "morans_i" in spat:
        lines.append(spat["morans_i"].to_string(index=False))
    lines.append("")
    if "latitudinal_gradient" in spat:
        lines.append("Latitudinal gradient:")
        lines.append(spat["latitudinal_gradient"].to_string(index=False))
    lines.append("")

    # Summary
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)

    # Count verified claims
    all_verified_claims = 0
    all_total_claims = 0
    if "article_claims_verification" in corr:
        c = corr["article_claims_verification"]
        all_verified_claims += c["MATCH_within_0.05"].sum()
        all_total_claims += len(c)
    if "verification" in inter:
        v = inter["verification"]
        all_verified_claims += v["MATCH_within_0.05"].sum()
        all_total_claims += len(v)
    if "claims_verification" in comp_vs:
        cv = comp_vs["claims_verification"]
        if "MATCH" in cv.columns:
            all_verified_claims += cv["MATCH"].sum()
            all_total_claims += len(cv)
    if "claims_verification" in derived:
        dv = derived["claims_verification"]
        if "MATCH_within_0.05" in dv.columns:
            all_verified_claims += dv["MATCH_within_0.05"].sum()
            all_total_claims += len(dv)

    lines.append(f"Total claims verified (within +/-0.05): {all_verified_claims}/{all_total_claims}")
    if desc.get("shapiro_wilk") is not None:
        lines.append(f"Normality tests verified: {desc['shapiro_wilk']['VERIFIED'].sum()}/{len(desc['shapiro_wilk'])}")
    if "decomposition" in vd:
        ph_row = vd["decomposition"][vd["decomposition"]["Property"] == "pH (KCl)"]
        if not ph_row.empty:
            lines.append(f"pH between-field variance: {ph_row.iloc[0]['Pct_between']}% (article claims 78.7%)")

    return "\n".join(lines)


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data()

    # ── v1 analyses ────────────────────────────────────────────────
    print("\n--- Running descriptive statistics ---")
    desc_results = descriptive_stats.run(df)

    print("--- Running intercorrelation analysis ---")
    inter_results = intercorrelation.run(df)

    print("--- Running full correlation analysis ---")
    corr_results = correlation_analysis.run(df)

    print("--- Running seasonal analysis ---")
    seas_results = seasonal_analysis.run(df)

    print("--- Running spatial analysis ---")
    spat_results = spatial_analysis.run(df)

    # ── v2 new analyses ────────────────────────────────────────────
    print("--- Computing composite spectral features ---")
    comp_results = composite_features.run(df)
    composites_df = comp_results["composites"]

    print("--- Running composite vs single comparison ---")
    comp_vs_results = composite_vs_single.run(
        df, composites_df, corr_results["all_correlations"])

    print("--- Computing derived soil indicators ---")
    derived_results = derived_soil.run(df, composites_df)

    print("--- Running variance decomposition ---")
    vd_results = variance_decomposition.run(df)

    print("--- Running confounding analysis ---")
    conf_results = confounding_analysis.run(df, corr_results.get("all_correlations"))

    # Gather all results
    all_results = {
        "descriptive": desc_results,
        "intercorrelation": inter_results,
        "correlation": corr_results,
        "seasonal": seas_results,
        "spatial": spat_results,
        "composite_features": comp_results,
        "composite_vs_single": comp_vs_results,
        "derived_soil": derived_results,
        "variance_decomposition": vd_results,
        "confounding": conf_results,
    }

    # Generate plots
    print("--- Generating plots ---")
    plots.run_all_plots(
        df=df,
        rho_matrix=inter_results.get("rho_matrix"),
        p_matrix=inter_results.get("p_matrix"),
        corr_df=corr_results.get("all_correlations"),
        ndvi_table=seas_results.get("ndvi_by_soc_class"),
        claims_df=corr_results.get("article_claims_verification"),
        # v2:
        comparison_df=comp_vs_results.get("comparison_table"),
        decomp_df=vd_results.get("decomposition"),
        confound_df=conf_results.get("partial_correlations"),
        sat_curve=conf_results.get("saturation_curve"),
        cv_rho_df=conf_results.get("cv_vs_rho"),
        top_derived_df=derived_results.get("top_rs"),
        delta_df=comp_vs_results.get("delta_vs_peak"),
    )

    # Generate report
    print("--- Generating report ---")
    report = generate_text_report(all_results)
    report_path = OUTPUT_DIR / "verification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    elapsed = time.time() - t0
    print(f"\nAll analyses completed in {elapsed:.1f}s")
    print(f"Output directory: {OUTPUT_DIR}")

    # Print report to console (handle Windows encoding)
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\n" + report.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
