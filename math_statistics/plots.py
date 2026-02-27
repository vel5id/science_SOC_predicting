"""
Visualization module: all plots for statistical verification.

Generates:
1. Histograms + KDE for soil properties (Figure 1 verification)
2. Soil intercorrelation heatmap (Figure 2)
3. Spearman heatmap: soil × S2 indices × season (Figure 3)
4. Scatter plots for top correlations (Figure 4)
5. Seasonal NDVI by SOC class (Figure 5)
6. Band-level correlation bar chart (Figure 6)
7. Topo/climate correlation bar chart (Figure 7)
8. Spatial maps of soil properties (Figure 8)
9. QQ-plots for normality check
10. Boxplots by year
11. VIF bar chart (multicollinearity check)
12. Bootstrap CI for key correlations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

from .config import (
    SOIL_TARGETS, SOIL_LABELS, SEASONS, SEASON_LABELS,
    TOPO_COLS, CLIMATE_COLS, OUTPUT_DIR,
)

# Global plot style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


def _save(fig, name: str):
    """Save figure to output directory."""
    out = OUTPUT_DIR / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# 1. Histograms + KDE (Figure 1)
# ──────────────────────────────────────────────────────────────────
def plot_histograms(df: pd.DataFrame):
    """Distribution histograms for 6 soil properties."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for i, col in enumerate(SOIL_TARGETS):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=40, density=True, alpha=0.6, color="steelblue", edgecolor="white")
        # KDE overlay
        try:
            kde_x = np.linspace(data.min(), data.max(), 200)
            kde = stats.gaussian_kde(data)
            ax.plot(kde_x, kde(kde_x), color="darkred", lw=2)
        except Exception:
            pass
        ax.set_title(SOIL_LABELS[col])
        ax.set_ylabel("Density")
        # Add statistics annotation
        ax.text(0.97, 0.95,
                f"n={len(data)}\nMean={data.mean():.2f}\nMedian={data.median():.2f}\nCV={data.std()/data.mean()*100:.1f}%",
                transform=ax.transAxes, va="top", ha="right", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Figure 1 verification: Distribution of soil properties", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "01_histograms")


# ──────────────────────────────────────────────────────────────────
# 2. Intercorrelation heatmap (Figure 2)
# ──────────────────────────────────────────────────────────────────
def plot_intercorrelation_heatmap(rho_matrix: pd.DataFrame, p_matrix: pd.DataFrame):
    """Heatmap of soil property intercorrelations."""
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = [SOIL_LABELS.get(c, c) for c in rho_matrix.index]
    mask = np.triu(np.ones_like(rho_matrix, dtype=bool), k=1)

    sns.heatmap(
        rho_matrix.values, mask=mask, annot=True, fmt=".2f",
        xticklabels=labels, yticklabels=labels,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        ax=ax, square=True, linewidths=0.5,
    )
    # Add significance stars
    for i in range(len(SOIL_TARGETS)):
        for j in range(i):
            p = p_matrix.iloc[i, j]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if stars:
                ax.text(j + 0.5, i + 0.75, stars, ha="center", va="center",
                        fontsize=7, color="black")

    ax.set_title("Figure 2 verification: Spearman intercorrelations")
    fig.tight_layout()
    _save(fig, "02_intercorrelation_heatmap")


# ──────────────────────────────────────────────────────────────────
# 3. Soil × S2 indices × season heatmap (Figure 3)
# ──────────────────────────────────────────────────────────────────
def plot_s2_index_heatmap(corr_df: pd.DataFrame):
    """Heatmap: soil properties vs S2 spectral indices by season."""
    indices = ["NDVI", "NDRE", "GNDVI", "EVI", "SAVI", "BSI", "Cl_Red_Edge"]
    seasons = ["spring", "summer", "late_summer", "autumn"]

    # Build pivot: rows = index_season, columns = soil target
    rows_data = []
    for idx in indices:
        for season in seasons:
            feat = f"s2_{idx}_{season}"
            row = {"index_season": f"{idx} {SEASON_LABELS[season]}"}
            for target in SOIL_TARGETS:
                match = corr_df[(corr_df["target"] == target) & (corr_df["feature"] == feat)]
                row[SOIL_LABELS[target]] = match.iloc[0]["rho"] if not match.empty else np.nan
            rows_data.append(row)

    pivot = pd.DataFrame(rows_data).set_index("index_season")

    fig, ax = plt.subplots(figsize=(10, 14))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-0.7, vmax=0.7, ax=ax, linewidths=0.3,
    )
    ax.set_title("Figure 3 verification: Spearman ρ — S2 indices × season vs soil properties")
    fig.tight_layout()
    _save(fig, "03_s2_index_season_heatmap")


# ──────────────────────────────────────────────────────────────────
# 4. Top scatter plots (Figure 4)
# ──────────────────────────────────────────────────────────────────
def plot_top_scatters(df: pd.DataFrame, corr_df: pd.DataFrame, n_top: int = 9):
    """Scatter plots for strongest correlations."""
    # Get top N overall
    top = corr_df.nlargest(n_top, "abs_rho")
    ncols = 3
    nrows = (n_top + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.ravel()

    for i, (_, row) in enumerate(top.iterrows()):
        ax = axes[i]
        target, feat = row["target"], row["feature"]
        mask = df[[target, feat]].notna().all(axis=1)
        x = df.loc[mask, feat]
        y = df.loc[mask, target]
        ax.scatter(x, y, s=5, alpha=0.3, color="steelblue")
        # Lowess trend (or simple linear fit for speed)
        z = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, np.polyval(z, xline), color="red", lw=2)

        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel(SOIL_LABELS.get(target, target))
        ax.set_title(f"ρ = {row['rho']:.3f}  (p = {row['p_value']:.1e})", fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Figure 4 verification: Strongest correlations (scatter)", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "04_top_scatter_plots")


# ──────────────────────────────────────────────────────────────────
# 5. Seasonal NDVI by SOC class (Figure 5)
# ──────────────────────────────────────────────────────────────────
def plot_seasonal_ndvi(ndvi_table: pd.DataFrame):
    """Line plot: seasonal NDVI trajectory per SOC class."""
    fig, ax = plt.subplots(figsize=(10, 6))

    season_order = ["Spring", "Summer", "Late summer", "Autumn"]
    palette = sns.color_palette("viridis", n_colors=ndvi_table["SOC_class"].nunique())

    for i, soc_cls in enumerate(ndvi_table["SOC_class"].unique()):
        sub = ndvi_table[ndvi_table["SOC_class"] == soc_cls]
        sub = sub.set_index("Season").reindex(season_order)
        ax.errorbar(
            season_order, sub["NDVI_mean"],
            yerr=(sub["NDVI_mean"] - sub["NDVI_q25"], sub["NDVI_q75"] - sub["NDVI_mean"]),
            marker="o", label=f"SOC {soc_cls}", color=palette[i],
            capsize=4, lw=2,
        )

    ax.set_ylabel("NDVI (S2)")
    ax.set_title("Figure 5 verification: Seasonal NDVI dynamics by SOC class")
    ax.legend(title="SOC class")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "05_seasonal_ndvi_by_soc")


# ──────────────────────────────────────────────────────────────────
# 6. Band-level correlations (Figure 6)
# ──────────────────────────────────────────────────────────────────
def plot_band_correlations(corr_df: pd.DataFrame):
    """Grouped bar chart: S2 summer band correlations with soil properties."""
    bands = ["s2_B2", "s2_B3", "s2_B4", "s2_B5", "s2_B6",
             "s2_B7", "s2_B8", "s2_B8A", "s2_B11", "s2_B12"]

    data = []
    for band_prefix in bands:
        feat = f"{band_prefix}_summer"
        for target in SOIL_TARGETS:
            match = corr_df[(corr_df["target"] == target) & (corr_df["feature"] == feat)]
            if not match.empty:
                data.append({
                    "Band": band_prefix.replace("s2_", ""),
                    "Soil property": SOIL_LABELS[target],
                    "rho": match.iloc[0]["rho"],
                })

    plot_df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=plot_df, x="Band", y="rho", hue="Soil property", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Figure 6 verification: S2 summer band correlations with soil properties")
    ax.set_ylabel("Spearman ρ")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, "06_band_correlations_summer")


# ──────────────────────────────────────────────────────────────────
# 7. Topo / climate correlation chart (Figure 7)
# ──────────────────────────────────────────────────────────────────
def plot_topo_climate_correlations(corr_df: pd.DataFrame):
    """Grouped bar chart: topo + climate feature correlations."""
    features = TOPO_COLS + CLIMATE_COLS
    data = []
    for feat in features:
        for target in SOIL_TARGETS:
            match = corr_df[(corr_df["target"] == target) & (corr_df["feature"] == feat)]
            if not match.empty:
                data.append({
                    "Feature": feat.replace("topo_", "").replace("climate_", ""),
                    "Soil property": SOIL_LABELS[target],
                    "rho": match.iloc[0]["rho"],
                })

    plot_df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=plot_df, x="Feature", y="rho", hue="Soil property", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Figure 7 verification: Topographic & climate correlations")
    ax.set_ylabel("Spearman ρ")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, "07_topo_climate_correlations")


# ──────────────────────────────────────────────────────────────────
# 8. Spatial maps (Figure 8)
# ──────────────────────────────────────────────────────────────────
def plot_spatial_maps(df: pd.DataFrame):
    """Scatter-based spatial maps of soil properties."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for i, col in enumerate(SOIL_TARGETS):
        ax = axes[i]
        valid = df[["centroid_lon", "centroid_lat", col]].dropna()
        sc = ax.scatter(
            valid["centroid_lon"], valid["centroid_lat"],
            c=valid[col], s=4, alpha=0.6,
            cmap="viridis", edgecolors="none",
        )
        fig.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_title(SOIL_LABELS[col])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")

    fig.suptitle("Figure 8 verification: Spatial distribution of soil properties", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "08_spatial_maps")


# ──────────────────────────────────────────────────────────────────
# 9. QQ-plots
# ──────────────────────────────────────────────────────────────────
def plot_qq(df: pd.DataFrame):
    """QQ-plots for normality assessment."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for i, col in enumerate(SOIL_TARGETS):
        ax = axes[i]
        data = df[col].dropna().values
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f"QQ-plot: {SOIL_LABELS[col]}")

    fig.suptitle("QQ-plots for normality verification", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "09_qq_plots")


# ──────────────────────────────────────────────────────────────────
# 10. Boxplots by year
# ──────────────────────────────────────────────────────────────────
def plot_boxplots_by_year(df: pd.DataFrame):
    """Boxplots of soil properties split by year."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for i, col in enumerate(SOIL_TARGETS):
        ax = axes[i]
        data_list = []
        labels = []
        for year in sorted(df["year"].unique()):
            d = df.loc[df["year"] == year, col].dropna()
            if len(d) > 0:
                data_list.append(d.values)
                labels.append(str(year))
        if data_list:
            bp = ax.boxplot(data_list, labels=labels, patch_artist=True)
            colors = sns.color_palette("Set2", len(data_list))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
        ax.set_title(SOIL_LABELS[col])
        ax.set_xlabel("Year")

    fig.suptitle("Soil properties by sampling year (Kruskal-Wallis check)", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "10_boxplots_by_year")


# ──────────────────────────────────────────────────────────────────
# 11. Bootstrap CI for key correlations
# ──────────────────────────────────────────────────────────────────
def plot_bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000):
    """Bootstrap 95% CI for the key article correlations."""
    from .config import ARTICLE_CLAIMS

    claims = {k: v for k, v in ARTICLE_CLAIMS.items() if v["feature"] not in SOIL_TARGETS}
    fig, ax = plt.subplots(figsize=(12, 6))

    names = []
    rhos = []
    ci_low = []
    ci_high = []
    article_vals = []

    rng = np.random.default_rng(42)

    for claim_id, info in claims.items():
        target, feat, art_rho = info["target"], info["feature"], info["rho"]
        mask = df[[target, feat]].notna().all(axis=1)
        x = df.loc[mask, feat].values
        y = df.loc[mask, target].values
        n = len(x)

        if n < 10:
            continue

        boot_rhos = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            r, _ = stats.spearmanr(x[idx], y[idx])
            boot_rhos.append(r)

        boot_rhos = np.array(boot_rhos)
        names.append(claim_id)
        rhos.append(np.median(boot_rhos))
        ci_low.append(np.percentile(boot_rhos, 2.5))
        ci_high.append(np.percentile(boot_rhos, 97.5))
        article_vals.append(art_rho)

    y_pos = np.arange(len(names))
    ax.barh(y_pos, rhos, xerr=[np.array(rhos) - np.array(ci_low),
                                np.array(ci_high) - np.array(rhos)],
            color="steelblue", alpha=0.7, capsize=4)
    ax.scatter(article_vals, y_pos, color="red", marker="D", s=80,
               zorder=5, label="Article ρ")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Spearman ρ")
    ax.set_title("Bootstrap 95% CI for key article correlations")
    ax.legend()
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    _save(fig, "11_bootstrap_ci")


# ──────────────────────────────────────────────────────────────────
# 12. Correlation difference heatmap: article vs computed
# ──────────────────────────────────────────────────────────────────
def plot_claim_verification(claims_df: pd.DataFrame):
    """Visual comparison: article rho vs computed rho."""
    if claims_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(claims_df))
    width = 0.35

    ax.bar(x - width/2, claims_df["article_rho"], width, label="Article ρ", color="coral")
    ax.bar(x + width/2, claims_df["computed_rho"], width, label="Computed ρ", color="steelblue")

    ax.set_xticks(x)
    ax.set_xticklabels(claims_df["claim"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Article claims vs computed values")
    ax.legend()
    ax.axhline(0, color="gray", lw=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "12_claim_verification")


# ──────────────────────────────────────────────────────────────────
# 13. VIF check (multicollinearity)
# ──────────────────────────────────────────────────────────────────
def plot_vif(df: pd.DataFrame, top_n: int = 30):
    """Variance Inflation Factor for top correlated features.

    High VIF (>10) signals multicollinearity that could affect model stability.
    """
    from sklearn.linear_model import LinearRegression

    # Select numeric RS features with no NaN
    exclude = {"id", "year", "grid_id", "centroid_lon", "centroid_lat"}
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c not in exclude and c not in SOIL_TARGETS and c != "hu"]
    sub = df[num_cols].dropna(axis=1, how="any").dropna()

    if sub.shape[1] < 2 or sub.shape[0] < 50:
        return

    # Subsample for speed
    if len(sub) > 2000:
        sub = sub.sample(2000, random_state=42)

    # Compute VIF for each column
    vifs = []
    X = sub.values
    for i in range(min(X.shape[1], 50)):  # limit to 50 features
        y_i = X[:, i]
        X_rest = np.delete(X, i, axis=1)
        lr = LinearRegression().fit(X_rest, y_i)
        r2 = lr.score(X_rest, y_i)
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        vifs.append({"Feature": sub.columns[i], "VIF": vif})

    vif_df = pd.DataFrame(vifs).sort_values("VIF", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["red" if v > 10 else "orange" if v > 5 else "steelblue" for v in vif_df["VIF"]]
    ax.barh(range(len(vif_df)), vif_df["VIF"].values, color=colors)
    ax.set_yticks(range(len(vif_df)))
    ax.set_yticklabels(vif_df["Feature"].values, fontsize=7)
    ax.set_xlabel("VIF")
    ax.set_title(f"Top-{top_n} Variance Inflation Factors (VIF > 10 = high multicollinearity)")
    ax.axvline(10, color="red", ls="--", lw=1, label="VIF = 10")
    ax.axvline(5, color="orange", ls="--", lw=1, label="VIF = 5")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "13_vif_multicollinearity")


# ──────────────────────────────────────────────────────────────────
# 14. Composite vs Single comparison (Figure 5 of v2)
# ──────────────────────────────────────────────────────────────────
def plot_composite_vs_single(comparison_df: pd.DataFrame):
    """Side-by-side bars: best single vs best composite per soil target."""
    if comparison_df is None or comparison_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(comparison_df))
    width = 0.35

    ax.barh(x - width/2, comparison_df["Single_rho"].abs(), width,
            label="Best single", color="steelblue", alpha=0.8)
    ax.barh(x + width/2, comparison_df["Composite_rho"].abs(), width,
            label="Best composite", color="coral", alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(comparison_df["Target"], fontsize=9)
    ax.set_xlabel("|Spearman rho|")
    ax.set_title("Figure 5 verification: Single vs Composite spectral features")
    ax.legend()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    _save(fig, "14_composite_vs_single")


# ──────────────────────────────────────────────────────────────────
# 15. Variance decomposition bar chart (Figure 8b of v2)
# ──────────────────────────────────────────────────────────────────
def plot_variance_decomposition(decomp_df: pd.DataFrame):
    """Stacked bar chart: between-field vs within-field variance."""
    if decomp_df is None or decomp_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(decomp_df))

    ax.bar(x, decomp_df["Pct_between"], color="steelblue", label="Between-field")
    ax.bar(x, decomp_df["Pct_within"], bottom=decomp_df["Pct_between"],
           color="coral", label="Within-field")

    ax.set_xticks(x)
    ax.set_xticklabels(decomp_df["Property"], rotation=30, ha="right")
    ax.set_ylabel("Variance (%)")
    ax.set_title("Figure 8b verification: Variance decomposition (between-field vs within-field)")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "15_variance_decomposition")


# ──────────────────────────────────────────────────────────────────
# 16. pH confounding: raw vs partial correlations (Figure 9b of v2)
# ──────────────────────────────────────────────────────────────────
def plot_confounding(confound_df: pd.DataFrame):
    """Grouped bar: raw SOC-VI correlation vs partial (controlling pH)."""
    if confound_df is None or confound_df.empty:
        return

    # Filter to main indices for readability
    main_vis = confound_df[confound_df["VI"].str.contains(
        r"s2_(NDVI|GNDVI|NDRE|EVI)_(summer|spring)", regex=True)]
    if main_vis.empty:
        main_vis = confound_df.head(12)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(main_vis))
    width = 0.35

    ax.bar(x - width/2, main_vis["rho_raw"], width, label="Raw rho(SOC, VI)", color="steelblue")
    ax.bar(x + width/2, main_vis["rho_partial_given_pH"], width,
           label="Partial rho(SOC, VI | pH)", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(main_vis["VI"].str.replace("s2_", ""), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Figure 9b verification: pH confounding in SOC-VI correlations")
    ax.legend()
    ax.axhline(0, color="gray", lw=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "16_ph_confounding")


# ──────────────────────────────────────────────────────────────────
# 17. NDVI saturation curve (Figure 9a of v2)
# ──────────────────────────────────────────────────────────────────
def plot_ndvi_saturation(sat_curve: pd.DataFrame):
    """NDVI(summer) as function of SOC — saturation plateau."""
    if sat_curve is None or sat_curve.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sat_curve["soc_mid"], sat_curve["ndvi_mean"],
                yerr=sat_curve["ndvi_std"], marker="o", capsize=3,
                color="steelblue", lw=2)
    ax.axvline(2.5, color="red", ls="--", lw=1.5, label="SOC = 2.5% (saturation)")
    ax.set_xlabel("SOC (%)")
    ax.set_ylabel("Mean NDVI (summer)")
    ax.set_title("Figure 9a verification: NDVI saturation curve vs SOC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "17_ndvi_saturation")


# ──────────────────────────────────────────────────────────────────
# 18. CV vs max correlation strength (Figure 9c of v2)
# ──────────────────────────────────────────────────────────────────
def plot_cv_vs_rho(cv_rho_df: pd.DataFrame):
    """Scatter: CV of soil property vs max |rho| with RS features."""
    if cv_rho_df is None or cv_rho_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(cv_rho_df["CV_%"], cv_rho_df["Max_abs_rho"], s=100, color="steelblue",
               edgecolors="black", zorder=5)
    for _, row in cv_rho_df.iterrows():
        ax.annotate(row["Property"], (row["CV_%"], row["Max_abs_rho"]),
                    fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("CV (%)")
    ax.set_ylabel("Max |Spearman rho|")
    ax.set_title("Figure 9c verification: CV vs correlation strength")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "18_cv_vs_rho")


# ──────────────────────────────────────────────────────────────────
# 19. Derived soil indicators: top correlations (Figure 7 of v2)
# ──────────────────────────────────────────────────────────────────
def plot_derived_soil_top(top_derived_df: pd.DataFrame):
    """Bar chart of top correlations for derived soil indicators."""
    if top_derived_df is None or top_derived_df.empty:
        return

    # Get best for each derived indicator
    best = top_derived_df.loc[top_derived_df.groupby("derived")["abs_rho"].idxmax()]

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(best))
    colors = ["coral" if r < 0 else "steelblue" for r in best["rho"]]
    ax.barh(y, best["rho"], color=colors, alpha=0.8)
    labels = [f"{row['derived']} ~ {row['feature']}" for _, row in best.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Spearman rho")
    ax.set_title("Figure 7 verification: Derived soil indicators — top correlations")
    ax.axvline(0, color="gray", lw=0.8)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    _save(fig, "19_derived_soil_top")


# ──────────────────────────────────────────────────────────────────
# 20. Delta vs peak single-season
# ──────────────────────────────────────────────────────────────────
def plot_delta_vs_peak(delta_df: pd.DataFrame):
    """Paired bars: seasonal delta vs peak single-season."""
    if delta_df is None or delta_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(delta_df))
    width = 0.35

    ax.bar(x - width/2, delta_df["Single_rho"].abs(), width,
           label="Best single-season", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, delta_df["Delta_rho"].abs(), width,
           label="Best delta/amplitude", color="orange", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(delta_df["Target"], rotation=30, ha="right")
    ax.set_ylabel("|Spearman rho|")
    ax.set_title("Multi-seasonal deltas vs peak single-season features")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "20_delta_vs_peak")


def run_all_plots(df: pd.DataFrame,
                  rho_matrix: pd.DataFrame = None,
                  p_matrix: pd.DataFrame = None,
                  corr_df: pd.DataFrame = None,
                  ndvi_table: pd.DataFrame = None,
                  claims_df: pd.DataFrame = None,
                  # v2 additions:
                  comparison_df: pd.DataFrame = None,
                  decomp_df: pd.DataFrame = None,
                  confound_df: pd.DataFrame = None,
                  sat_curve: pd.DataFrame = None,
                  cv_rho_df: pd.DataFrame = None,
                  top_derived_df: pd.DataFrame = None,
                  delta_df: pd.DataFrame = None):
    """Generate all plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(exist_ok=True)

    plot_histograms(df)
    plot_qq(df)
    plot_boxplots_by_year(df)
    plot_spatial_maps(df)

    if rho_matrix is not None and p_matrix is not None:
        plot_intercorrelation_heatmap(rho_matrix, p_matrix)

    if corr_df is not None:
        plot_s2_index_heatmap(corr_df)
        plot_top_scatters(df, corr_df)
        plot_band_correlations(corr_df)
        plot_topo_climate_correlations(corr_df)

    if ndvi_table is not None:
        plot_seasonal_ndvi(ndvi_table)

    if claims_df is not None:
        plot_claim_verification(claims_df)

    plot_bootstrap_ci(df)
    plot_vif(df)

    # v2 additions
    plot_composite_vs_single(comparison_df)
    plot_variance_decomposition(decomp_df)
    plot_confounding(confound_df)
    plot_ndvi_saturation(sat_curve)
    plot_cv_vs_rho(cv_rho_df)
    plot_delta_vs_peak(delta_df)

if __name__ == "__main__":
    import os
    print("Loading data for plot generation...")
    # Load raw data
    df_path = Path("data/features/full_dataset.csv")
    if not df_path.exists():
        print("Error: data/features/full_dataset.csv not found.")
        exit(1)
    df = pd.read_csv(df_path)
    
    # Load computed matrices if they exist
    out = OUTPUT_DIR
    def load_sheet(file, sheet):
        p = out / file
        if p.exists():
            try:
                if file.endswith(".csv"):
                    return pd.read_csv(p)
                return pd.read_excel(p, sheet_name=sheet)
            except Exception as e:
                print(f"Warning: could not load {sheet} from {file}: {e}")
        return None
        
    # Read the data
    rho_matrix = load_sheet("intercorrelation.xlsx", "spearman_rho")
    if rho_matrix is not None:
        rho_matrix = rho_matrix.set_index(rho_matrix.columns[0])
    p_matrix = load_sheet("intercorrelation.xlsx", "p_values")
    if p_matrix is not None:
        p_matrix = p_matrix.set_index(p_matrix.columns[0])
        
    corr_df = load_sheet("all_spearman_correlations.csv", None)
    ndvi_table = load_sheet("seasonal_ndvi.xlsx", "ndvi_by_soc") if (out / "seasonal_ndvi.xlsx").exists() else None
    
    claims_df = load_sheet("correlation_analysis.xlsx", "claims_verification")
    comparison_df = load_sheet("composite_vs_single.xlsx", "comparison")
    delta_df = load_sheet("composite_vs_single.xlsx", "delta_vs_peak")
    decomp_df = load_sheet("variance_decomposition.xlsx", "decomposition")
    confound_df = load_sheet("confounding_analysis.xlsx", "partial_correlations")
    sat_curve = load_sheet("confounding_analysis.xlsx", "saturation_curve")
    cv_rho_df = load_sheet("confounding_analysis.xlsx", "cv_vs_rho")
    top_derived_df = load_sheet("derived_soil_indicators.xlsx", "top_correlations") if (out / "derived_soil_indicators.xlsx").exists() else None

    print("Generating plots (this might take a few seconds)...")
    run_all_plots(
        df=df,
        rho_matrix=rho_matrix,
        p_matrix=p_matrix,
        corr_df=corr_df,
        ndvi_table=ndvi_table,
        claims_df=claims_df,
        comparison_df=comparison_df,
        decomp_df=decomp_df,
        confound_df=confound_df,
        sat_curve=sat_curve,
        cv_rho_df=cv_rho_df,
        top_derived_df=top_derived_df,
        delta_df=delta_df
    )
    print(f"Plots generated successfully in {out}/plots/")

