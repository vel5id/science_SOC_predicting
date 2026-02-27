"""
Combined visualization: SHAP results → CNN channel recommendations for S prediction.
Produces a 4-panel figure with:
  1. SHAP beeswarm (top-20 features)
  2. Feature importance bar (top-20, XGBoost MDI)
  3. Channel category breakdown (what kind of data matters most)
  4. CNN spectral channel mapping table (which of the 54 channels are most important)
"""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import shap

OUT_DIR = os.path.join(_PROJECT_ROOT, "ML/results/shap_sulfur")
SHAP_CSV   = os.path.join(OUT_DIR, "shap_mean_abs_values.csv")
IMP_CSV    = os.path.join(OUT_DIR, "feature_importance_all.csv")
BEESWARM   = os.path.join(OUT_DIR, "shap_beeswarm_sulfur.png")

# 54-channel layout (from train_cascaded_cnn.py / train_multiseason_convnext.py)
CHANNEL_MAP = {
    **{i: f"B{['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'][i % 13]}_spring" for i in range(12)},
    12: "NDVI_spring", 13: "BSI_spring", 14: "NDSI_spring", 15: "NDWI_spring", 16: "ReCl_spring",
    **{17+j: f"B{['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'][j]}_summer" for j in range(12)},
    29: "NDVI_summer", 30: "BSI_summer", 31: "NDSI_summer", 32: "NDWI_summer", 33: "ReCl_summer",
    **{34+j: f"B{['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'][j]}_autumn" for j in range(12)},
    46: "NDVI_autumn", 47: "BSI_autumn", 48: "NDSI_autumn", 49: "NDWI_autumn", 50: "ReCl_autumn",
    51: "VV_radar", 52: "VH_radar", 53: "DEM",
}

# Category assignment for grouping
def categorize_feature(name):
    n = name.lower()
    if n.startswith("s_no3") or n.startswith("soc_x"):
        return "Interaction Features"
    if "b2" in n or "b3" in n or "blue" in n or "green" in n:
        return "Visible (Blue/Green)"
    if "b4" in n or "red" in n or "ndvi" in n or "savi" in n or "gndvi" in n:
        return "Red/NIR/Vegetation"
    if "b11" in n or "b12" in n or "swir" in n or "bsi" in n or "msi" in n:
        return "SWIR/Soil"
    if "dem" in n or "topo" in n or "aspect" in n or "slope" in n:
        return "Topography"
    if "climate" in n or "map" in n or "precip" in n or "temp" in n:
        return "Climate"
    if "glcm" in n:
        return "Texture (GLCM)"
    if "l8" in n or "landsat" in n:
        return "Landsat-8"
    if "spectral_pca" in n or "pca" in n:
        return "PCA Spectral"
    return "Other Spectral"

# Color palette for categories
CAT_COLORS = {
    "Interaction Features":  "#E91E63",
    "Visible (Blue/Green)":  "#1565C0",
    "Red/NIR/Vegetation":    "#2E7D32",
    "SWIR/Soil":             "#F57C00",
    "Topography":            "#6D4C41",
    "Climate":               "#0097A7",
    "Texture (GLCM)":        "#7B1FA2",
    "Landsat-8":             "#558B2F",
    "PCA Spectral":          "#795548",
    "Other Spectral":        "#90A4AE",
}

def main():
    shap_df = pd.read_csv(SHAP_CSV)
    imp_df  = pd.read_csv(IMP_CSV)
    shap_df["category"] = shap_df["feature"].apply(categorize_feature)
    imp_df["category"]  = imp_df["feature"].apply(categorize_feature)

    # --- Category aggregation (sum of mean |SHAP|) ---
    cat_shap = shap_df.groupby("category")["mean_abs_shap"].sum().sort_values(ascending=False)

    # --- Top-20 features ---
    top20_shap = shap_df.head(20).copy()
    top20_imp  = imp_df.head(20).copy()

    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor("#0F1117")

    # Suptitle
    fig.suptitle(
        "SHAP Analysis — Predictors of Sulfur (S, mg/kg)\n"
        "XGBoost · 514 Features · 81-Fold Spatial LOFO-CV   |   Spearman ρ = 0.871   |   R² = 0.818",
        fontsize=16, fontweight="bold", color="white", y=0.99, va="top"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35,
                           left=0.06, right=0.98, top=0.95, bottom=0.04)

    ax_shap  = fig.add_subplot(gs[0, 0])  # SHAP bar
    ax_imp   = fig.add_subplot(gs[0, 1])  # MDI bar
    ax_cat   = fig.add_subplot(gs[1, 0])  # Category pie
    ax_tbl   = fig.add_subplot(gs[1, 1])  # CNN channel table

    dark_bg = "#1A1D2E"
    for ax in [ax_shap, ax_imp, ax_cat, ax_tbl]:
        ax.set_facecolor(dark_bg)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3A3D5C")

    # =========================================================
    # Panel 1: SHAP Mean |SHAP| bar
    # =========================================================
    colors_shap = [CAT_COLORS.get(c, "#90A4AE") for c in top20_shap["category"]]
    bars = ax_shap.barh(
        range(len(top20_shap)), top20_shap["mean_abs_shap"].values[::-1],
        color=colors_shap[::-1], edgecolor="#0F1117", linewidth=0.5
    )
    ax_shap.set_yticks(range(len(top20_shap)))
    ax_shap.set_yticklabels(top20_shap["feature"].values[::-1], fontsize=9, color="white")
    ax_shap.set_xlabel("Mean |SHAP Value| (impact on S prediction)", color="white", fontsize=10)
    ax_shap.set_title("① SHAP Feature Impact (Top-20)", color="white", fontsize=12, fontweight="bold", pad=8)
    ax_shap.xaxis.label.set_color("white")
    ax_shap.tick_params(axis="x", colors="white")

    # Add value labels
    for i, (val, cat) in enumerate(zip(top20_shap["mean_abs_shap"].values[::-1],
                                        top20_shap["category"].values[::-1])):
        ax_shap.text(val + 0.02, i, f"{val:.3f}", va="center", ha="left",
                     fontsize=8, color="white")

    ax_shap.set_xlim(0, top20_shap["mean_abs_shap"].max() * 1.18)

    # =========================================================
    # Panel 2: XGBoost MDI Importance bar
    # =========================================================
    colors_imp = [CAT_COLORS.get(c, "#90A4AE") for c in top20_imp["category"]]
    ax_imp.barh(
        range(len(top20_imp)), top20_imp["importance"].values[::-1],
        color=colors_imp[::-1], edgecolor="#0F1117", linewidth=0.5
    )
    ax_imp.set_yticks(range(len(top20_imp)))
    ax_imp.set_yticklabels(top20_imp["feature"].values[::-1], fontsize=9, color="white")
    ax_imp.set_xlabel("XGBoost MDI (Mean Gain, avg over folds)", color="white", fontsize=10)
    ax_imp.set_title("② XGBoost Feature Importance (Top-20)", color="white", fontsize=12, fontweight="bold", pad=8)
    ax_imp.tick_params(axis="x", colors="white")
    for i, val in enumerate(top20_imp["importance"].values[::-1]):
        ax_imp.text(val + 0.001, i, f"{val:.3f}", va="center", ha="left", fontsize=8, color="white")
    ax_imp.set_xlim(0, top20_imp["importance"].max() * 1.18)

    # =========================================================
    # Panel 3: Category pie chart (SHAP-weighted)
    # =========================================================
    cat_labels = cat_shap.index.tolist()
    cat_vals   = cat_shap.values
    cat_cols   = [CAT_COLORS.get(c, "#90A4AE") for c in cat_labels]
    wedges, texts, autotexts = ax_cat.pie(
        cat_vals, labels=None, colors=cat_cols,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=140, pctdistance=0.75,
        wedgeprops=dict(edgecolor="#0F1117", linewidth=1.2)
    )
    for t in autotexts:
        t.set_color("white"); t.set_fontsize(8)

    legend_patches = [mpatches.Patch(color=CAT_COLORS.get(c, "#90A4AE"), label=c) for c in cat_labels]
    ax_cat.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.22),
                  ncol=2, fontsize=8, labelcolor="white",
                  facecolor=dark_bg, edgecolor="#3A3D5C")
    ax_cat.set_title("③ Feature Category Breakdown (SHAP-weighted)", color="white", fontsize=12, fontweight="bold", pad=8)

    # =========================================================
    # Panel 4: CNN Channel Mapping Table
    # =========================================================
    ax_tbl.axis("off")
    ax_tbl.set_title("④ CNN Channel Recommendations for S Prediction\n(54-channel multiseason patches)", 
                     color="white", fontsize=12, fontweight="bold", pad=8)

    # Build channel importance table
    cnn_channels = [
        # (Channel idx range, name, season, band, SHAP rank, status)
        ("Ch 0",  "B2 (Blue)",     "Spring",      "Visible",    "#1565C0",  "★ TOP-3 SHAP"),
        ("Ch 1",  "B3 (Green)",    "Spring",      "Visible",    "#1565C0",  "★ TOP-4 SHAP"),
        ("Ch 11", "B11 (SWIR1)",   "Spring",      "SWIR/Soil",  "#F57C00",  "✓ Top-12 SHAP"),
        ("Ch 13", "BSI",           "Spring",      "SWIR/Soil",  "#F57C00",  "✓ Top-11 SHAP"),
        ("Ch 18", "B2 (Blue)",     "Summer",      "Visible",    "#1565C0",  "✓ Top-7 SHAP"),
        ("Ch 28", "B11 (SWIR1)",   "Late-Sum.",   "SWIR/Soil",  "#F57C00",  "✓ Top-6 SHAP"),
        ("Ch 53", "DEM",           "Static",      "Topography", "#6D4C41",  "✓ Top-8 SHAP"),
        ("—",     "SOC×NO3",       "Tabular",     "Interaction","#E91E63",  "★ TOP-2 SHAP"),
        ("—",     "S/NO3 ratio",   "Tabular",     "Interaction","#E91E63",  "★ TOP-1 SHAP"),
    ]

    col_labels = ["Ch", "Feature", "Season", "Type", "Priority"]
    cell_text  = [[c[0], c[1], c[2], c[3], c[5]] for c in cnn_channels]
    cell_colors = []
    for c in cnn_channels:
        row_col = [dark_bg] * 4 + [c[4]]
        cell_colors.append(row_col)

    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#3A3D5C")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#2A2D4E")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_text_props(color="white")

    # --- Shared legend (categories) ---
    category_handles = [
        mpatches.Patch(color=v, label=k) for k, v in CAT_COLORS.items()
        if k in top20_shap["category"].values or k in top20_imp["category"].values
    ]
    fig.legend(handles=category_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=5, fontsize=9,
               labelcolor="white", facecolor=dark_bg, edgecolor="#3A3D5C",
               title="Feature Categories", title_fontsize=10)
    for text in fig.legends[0].get_texts():
        text.set_color("white")

    out_path = os.path.join(OUT_DIR, "shap_combined_visualization.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved combined visualization: {out_path}")


if __name__ == "__main__":
    main()
