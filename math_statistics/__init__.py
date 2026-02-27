"""
Math-statistics verification package for the soil remote sensing article.

Modules:
    config          – shared constants (soil targets, feature groups, paths)
    descriptive_stats – Table 1 reproduction, Shapiro-Wilk, Kruskal-Wallis
    intercorrelation  – soil-property intercorrelation matrix (Fig. 2)
    correlation_analysis – Spearman ρ with all RS features, BH correction
    seasonal_analysis   – NDVI seasonal dynamics by SOC class (Fig. 5)
    spatial_analysis    – Moran's I spatial autocorrelation check
    plots              – all visualisation routines
    run_all            – orchestrator: run every analysis + save report
"""
