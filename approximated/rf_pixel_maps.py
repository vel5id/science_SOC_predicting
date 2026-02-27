"""
rf_pixel_maps.py
================
Stage 3 of the RF pipeline: pixel-level prediction on farm + Ridge comparison.

Steps:
  1. Load pixels_Агро_Парасат_2023_real.csv  (32,434 pixels)
  2. Load trained RF models (rf_models/rf_{element}.pkl)
  3. Join field-level features from rf_dataset.csv via field_name
  4. Build pixel feature matrix (40 features per element)
  5. RF predict for all 6 elements  (expm1 back-transform for P,S,NO3)
  6. Re-build Ridge predictions (same logic as pixel_geo_approx.py)
  7. Build comparison visualisations:
       rf_pixel_maps_comparison.png   — 2×6 Ridge vs RF maps
       rf_vs_ridge_diff.png           — 1×6 RF-minus-Ridge difference maps
       rf_pixel_detail_{element}.png  — per-element full-detail maps
  8. Save pixel predictions CSV

Run: python approximated/rf_pixel_maps.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
PIXELS_CSV = BASE / "approximated" / "tiff" / "pixels_Агро_Парасат_2023_real.csv"
RF_DATASET = BASE / "data" / "features" / "rf_dataset.csv"
FULL_DS    = BASE / "data" / "features" / "full_dataset.csv"
MODELS_DIR = BASE / "math_statistics" / "output" / "rf" / "rf_models"
OUT_RF     = BASE / "math_statistics" / "output" / "rf"
OUT_PLOTS  = BASE / "math_statistics" / "output" / "plots"
OUT_RF.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

FARM = "Агро Парасат"
YEAR = 2023

# ─── Target config ────────────────────────────────────────────────────────────
TARGETS    = ["ph", "k", "p", "hu", "s", "no3"]
LOG_TARGETS = {"p", "s", "no3"}

TARGET_LABELS = {
    "ph":  "pH",
    "k":   "K, mg/kg",
    "p":   "P, mg/kg",
    "hu":  "Humus, %",
    "s":   "S, mg/kg",
    "no3": "NO₃, mg/kg",
}

TARGET_CMAPS = {
    "ph":  "RdYlGn",
    "k":   "YlOrRd",
    "p":   "BuGn",
    "hu":  "YlGn",
    "s":   "PuRd",
    "no3": "Blues",
}

# ─── Ridge BEST_PREDICTOR config (from pixel_geo_approx.py) ──────────────────
# (csv_col_in_full_dataset, pixel_col_in_pixel_csv)
BEST_PREDICTOR = {
    "ph":  [("s2_NDRE_spring", "ndre"),  ("s2_BSI_spring",  "bsi")],
    "k":   [("s2_BSI_spring",  "bsi"),   ("s2_NDRE_spring", "ndre")],
    "p":   [("s2_GNDVI_spring","gndvi"), ("s2_BSI_spring",  "bsi")],
    "hu":  [("s2_EVI_summer",  "evi"),   ("s2_NDRE_spring", "ndre")],
    "s":   [("s2_GNDVI_autumn","gndvi"), ("s2_BSI_spring",  "bsi")],
    "no3": [("s2_GNDVI_spring","gndvi"), ("s2_EVI_summer",  "evi")],
}

RIDGE_ALPHA = 10.0

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("rf_pixel_maps.py  —  RF Pixel Prediction + Ridge Comparison")
print("=" * 70)

# ─── 1. Load data ─────────────────────────────────────────────────────────────
print(f"\nLoading pixel data: {PIXELS_CSV.name} ...")
px_df = pd.read_csv(PIXELS_CSV)
print(f"  Pixel shape: {px_df.shape}")
print(f"  Fields:  {sorted(px_df['field_name'].unique().tolist())}")
print(f"  Pixel columns: {list(px_df.columns)}")

print(f"\nLoading rf_dataset.csv ...")
rf_df = pd.read_csv(RF_DATASET)
# Keep only Agro Parasat 2023
parasat_mask = (
    (rf_df["farm"].str.contains("Parasat|Парасат", case=False, na=False)) &
    (rf_df["year"] == YEAR)
)
farm_rf = rf_df[parasat_mask].copy()
print(f"  rf_dataset shape: {rf_df.shape}  |  Parasat 2023: {len(farm_rf)} rows")

print(f"\nLoading full_dataset.csv (for Ridge training) ...")
full_df = pd.read_csv(FULL_DS)
print(f"  Full dataset shape: {full_df.shape}")

# ─── 2. Build field-level feature map for RF ──────────────────────────────────
print("\nBuilding field-level feature lookup ...")
# For each field_name, get median of all RF features
# (Parasat 2023 may have multiple soil samples per field → median per field)
all_rf_features = set()
for tgt in TARGETS:
    with open(MODELS_DIR / f"rf_{tgt}.pkl", "rb") as f:
        bundle = pickle.load(f)
    all_rf_features.update(bundle["features"])

all_rf_features = sorted(all_rf_features)
print(f"  Total unique RF features needed: {len(all_rf_features)}")

# Compute per-field median of all features from Parasat 2023 rf_dataset
# (each field may have multiple sample rows → take median)
field_feat_map = (
    farm_rf.groupby("field_name")[all_rf_features]
    .median()
    .reset_index()
)
print(f"  Fields with feature data: {len(field_feat_map)}")

# Farm-level fallback for pixels outside any field
farm_feat_median = farm_rf[all_rf_features].median()

# ─── 3. Join features to pixels ───────────────────────────────────────────────
print("\nJoining field-level features to pixels ...")
px_feat = px_df.merge(field_feat_map, on="field_name", how="left")

# Fill any remaining NaN with farm median
n_missing_field = px_feat[all_rf_features].isna().any(axis=1).sum()
if n_missing_field > 0:
    print(f"  Pixels without field match: {n_missing_field} → filling with farm median")
    for col in all_rf_features:
        px_feat[col] = px_feat[col].fillna(farm_feat_median[col])

# Verify no NaN remains
remaining_nan = px_feat[all_rf_features].isna().sum().sum()
print(f"  NaN remaining in features: {remaining_nan}")
print(f"  Feature matrix ready: {len(px_feat)} pixels × {len(all_rf_features)} features")

# ─── 4. RF Pixel Predictions ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  RF PIXEL PREDICTIONS")
print("=" * 50)

rf_predictions = {}

for tgt in TARGETS:
    with open(MODELS_DIR / f"rf_{tgt}.pkl", "rb") as f:
        bundle = pickle.load(f)

    rf_model    = bundle["rf"]
    imputer     = bundle["imputer"]
    features    = bundle["features"]
    log_transform = bundle["log_transform"]

    # Build feature matrix
    X = px_feat[features].values
    X = imputer.transform(X)

    # Predict
    y_pred = rf_model.predict(X)

    # Back-transform if log
    if log_transform:
        y_pred = np.expm1(y_pred)
        y_pred = np.maximum(y_pred, 0.0)   # clip negatives

    rf_predictions[tgt] = y_pred
    pmin, pmax, pmean = y_pred.min(), y_pred.max(), y_pred.mean()
    print(f"  {tgt:<5}: min={pmin:.3f}  max={pmax:.3f}  mean={pmean:.3f}"
          f"  (log_transform={log_transform})")

# ─── 5. Ridge Pixel Predictions ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("  RIDGE PIXEL PREDICTIONS")
print("=" * 50)

def build_ridge_predictions(full_df, px_df, tgt, best_pred_pairs, alpha=10.0):
    """
    Train Ridge on field-level full_dataset, predict on pixel CSV.
    Mirrors the logic from pixel_geo_approx.py.
    """
    # ── Training data: all farms (global model, same as pixel_geo_approx.py) ──
    train = full_df[full_df[tgt].notna()].copy()
    if len(train) < 20:
        return np.full(len(px_df), np.nan)

    # Spectral features
    csv_cols  = [p[0] for p in best_pred_pairs]
    px_cols   = [p[1] for p in best_pred_pairs]

    # Drop rows where spectral features are missing
    train = train.dropna(subset=csv_cols + ["centroid_lon", "centroid_lat"])
    if len(train) < 20:
        return np.full(len(px_df), np.nan)

    # UTM normalisation (approximate, using min/max of training set)
    # Convert lon/lat → UTM-like (simple linear scaling for stability)
    lon_min, lon_max = train["centroid_lon"].min(), train["centroid_lon"].max()
    lat_min, lat_max = train["centroid_lat"].min(), train["centroid_lat"].max()
    # Avoid division by zero
    lon_rng = lon_max - lon_min if lon_max > lon_min else 1.0
    lat_rng = lat_max - lat_min if lat_max > lat_min else 1.0

    nx_tr = (train["centroid_lon"] - lon_min) / lon_rng
    ny_tr = (train["centroid_lat"] - lat_min) / lat_rng

    X_spec_tr = train[csv_cols].values
    X_tr = np.column_stack([X_spec_tr, nx_tr, ny_tr, nx_tr * ny_tr])
    y_tr = train[tgt].values

    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    pipe.fit(X_tr, y_tr)

    # ── Pixel prediction ──────────────────────────────────────────────────────
    # Normalise pixel UTM coords using same scale as training
    nx_px = (px_df["lon"] - lon_min) / lon_rng
    ny_px = (px_df["lat"] - lat_min) / lat_rng

    X_spec_px = px_df[px_cols].values
    X_px = np.column_stack([X_spec_px, nx_px, ny_px, nx_px * ny_px])

    return pipe.predict(X_px)


ridge_predictions = {}

for tgt in TARGETS:
    y_ridge = build_ridge_predictions(
        full_df, px_df, tgt, BEST_PREDICTOR[tgt], alpha=RIDGE_ALPHA
    )
    ridge_predictions[tgt] = y_ridge
    pmin, pmax, pmean = np.nanmin(y_ridge), np.nanmax(y_ridge), np.nanmean(y_ridge)
    print(f"  {tgt:<5}: min={pmin:.3f}  max={pmax:.3f}  mean={pmean:.3f}")

# ─── 6. Save pixel predictions CSV ───────────────────────────────────────────
print("\nSaving pixel predictions CSV ...")
pred_df = px_df[["pixel_id", "field_name", "lon", "lat", "utm_x", "utm_y",
                  "ndvi", "ndre", "gndvi", "evi", "bsi"]].copy()

for tgt in TARGETS:
    pred_df[f"rf_{tgt}"]    = rf_predictions[tgt]
    pred_df[f"ridge_{tgt}"] = ridge_predictions[tgt]
    pred_df[f"diff_{tgt}"]  = rf_predictions[tgt] - ridge_predictions[tgt]

pred_df.to_csv(OUT_RF / "rf_pixel_predictions.csv", index=False)
print(f"  Saved: rf_pixel_predictions.csv  ({len(pred_df)} rows)")

# ─── 7. Comparison of RF vs Ridge predictions ─────────────────────────────────
print("\nComputing RF vs Ridge comparison stats ...")
for tgt in TARGETS:
    rf_v = rf_predictions[tgt]
    ri_v = ridge_predictions[tgt]
    diff = rf_v - ri_v
    rho, _ = spearmanr(rf_v, ri_v)
    print(f"  {tgt:<5}: RF mean={rf_v.mean():.3f}  Ridge mean={ri_v.mean():.3f}"
          f"  diff mean={diff.mean():.3f}  rho(RF,Ridge)={rho:.3f}")

# ─── Helper: extract 2D grid coords ──────────────────────────────────────────
def get_scatter_coords(px_df, pred_df, key):
    """Return (lon, lat, values) arrays for scatter plotting."""
    return px_df["lon"].values, px_df["lat"].values, pred_df[key].values


# ─── 8. Main comparison figure: 2 rows × 6 cols ───────────────────────────────
print("\nRendering comparison figure (2×6) ...")

DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
TEXT_CLR = "#e0e0e0"
ACC_CLR  = "#e94560"

POINT_SIZE = 0.4
ALPHA_SCATTER = 0.6

fig, axes = plt.subplots(
    2, 6, figsize=(28, 10),
    facecolor=DARK_BG,
    gridspec_kw={"hspace": 0.08, "wspace": 0.05},
)
fig.patch.set_facecolor(DARK_BG)

lons = px_df["lon"].values
lats = px_df["lat"].values

for col_idx, tgt in enumerate(TARGETS):
    rf_v    = rf_predictions[tgt]
    ridge_v = ridge_predictions[tgt]

    cmap = TARGET_CMAPS[tgt]

    # Shared colormap range for fair comparison
    vmin = min(np.nanpercentile(rf_v, 2), np.nanpercentile(ridge_v, 2))
    vmax = max(np.nanpercentile(rf_v, 98), np.nanpercentile(ridge_v, 98))

    for row_idx, (vals, row_label) in enumerate([
        (ridge_v, "Ridge (geo-spectral)"),
        (rf_v,    "Random Forest"),
    ]):
        ax = axes[row_idx, col_idx]
        ax.set_facecolor(PANEL_BG)

        sc = ax.scatter(
            lons, lats, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
            s=POINT_SIZE, alpha=ALPHA_SCATTER, linewidths=0, rasterized=True,
        )

        # Colorbar
        cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.04,
                            shrink=0.85, aspect=25)
        cbar.ax.tick_params(labelsize=6, colors=TEXT_CLR)
        cbar.outline.set_edgecolor("none")

        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_color("#555555")

        # Column header (top row only)
        if row_idx == 0:
            ax.set_title(TARGET_LABELS[tgt], color=TEXT_CLR,
                         fontsize=10, fontweight="bold", pad=6)

        # Row label (left column only)
        if col_idx == 0:
            ax.set_ylabel(row_label, color=TEXT_CLR,
                          fontsize=9, labelpad=8)

# Title
fig.suptitle(
    f"RF vs Ridge Pixel-Level Soil Predictions  |  {FARM} {YEAR}",
    color=TEXT_CLR, fontsize=14, fontweight="bold", y=1.01,
)

out_path = OUT_PLOTS / "rf_pixel_maps_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=DARK_BG, edgecolor="none")
plt.close(fig)
print(f"  Saved: {out_path.name}")

# ─── 9. Difference map: RF − Ridge ───────────────────────────────────────────
print("Rendering difference map (RF − Ridge) ...")

fig, axes = plt.subplots(
    1, 6, figsize=(28, 5),
    facecolor=DARK_BG,
    gridspec_kw={"wspace": 0.05},
)
fig.patch.set_facecolor(DARK_BG)

for col_idx, tgt in enumerate(TARGETS):
    diff = rf_predictions[tgt] - ridge_predictions[tgt]
    ax = axes[col_idx]
    ax.set_facecolor(PANEL_BG)

    # Symmetric colormap
    abs_max = np.nanpercentile(np.abs(diff), 98)
    sc = ax.scatter(
        lons, lats, c=diff, cmap="RdBu_r",
        vmin=-abs_max, vmax=abs_max,
        s=POINT_SIZE, alpha=ALPHA_SCATTER, linewidths=0, rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.04,
                        shrink=0.85, aspect=30)
    cbar.ax.tick_params(labelsize=6, colors=TEXT_CLR)
    cbar.outline.set_edgecolor("none")

    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color("#555555")

    ax.set_title(f"{TARGET_LABELS[tgt]}\n(RF − Ridge)",
                 color=TEXT_CLR, fontsize=9, fontweight="bold", pad=5)

fig.suptitle(
    f"RF minus Ridge Difference Maps  |  {FARM} {YEAR}",
    color=TEXT_CLR, fontsize=13, fontweight="bold", y=1.02,
)

out_path = OUT_PLOTS / "rf_vs_ridge_diff.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=DARK_BG, edgecolor="none")
plt.close(fig)
print(f"  Saved: {out_path.name}")

# ─── 10. Per-element detail maps (4-panel: RF / Ridge / Diff / Scatter) ───────
print("Rendering per-element detail maps ...")

for tgt in TARGETS:
    rf_v    = rf_predictions[tgt]
    ridge_v = ridge_predictions[tgt]
    diff    = rf_v - ridge_v
    cmap    = TARGET_CMAPS[tgt]
    label   = TARGET_LABELS[tgt]

    vmin_sh = min(np.nanpercentile(rf_v, 2), np.nanpercentile(ridge_v, 2))
    vmax_sh = max(np.nanpercentile(rf_v, 98), np.nanpercentile(ridge_v, 98))
    abs_max_diff = np.nanpercentile(np.abs(diff), 98)

    fig, axes = plt.subplots(
        1, 4, figsize=(22, 6),
        facecolor=DARK_BG,
        gridspec_kw={"wspace": 0.06},
    )
    fig.patch.set_facecolor(DARK_BG)

    panels = [
        (ridge_v, cmap,      vmin_sh,      vmax_sh,      f"Ridge\n{label}"),
        (rf_v,    cmap,      vmin_sh,      vmax_sh,      f"Random Forest\n{label}"),
        (diff,    "RdBu_r", -abs_max_diff, abs_max_diff, f"RF − Ridge\n{label}"),
    ]

    for ax_i, (vals, c, vlo, vhi, ttl) in enumerate(panels):
        ax = axes[ax_i]
        ax.set_facecolor(PANEL_BG)
        sc = ax.scatter(
            lons, lats, c=vals, cmap=c, vmin=vlo, vmax=vhi,
            s=0.5, alpha=0.65, linewidths=0, rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.045,
                            shrink=0.85, aspect=28)
        cbar.ax.tick_params(labelsize=7, colors=TEXT_CLR)
        cbar.outline.set_edgecolor("none")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.set_title(ttl, color=TEXT_CLR, fontsize=10,
                     fontweight="bold", pad=7)

    # 4th panel: scatter RF vs Ridge
    ax = axes[3]
    ax.set_facecolor(PANEL_BG)
    rho, _ = spearmanr(rf_v, ridge_v)

    # Hex-density scatter
    h = ax.hexbin(ridge_v, rf_v, gridsize=40, cmap="hot_r",
                  mincnt=1, linewidths=0.1, bins="log")
    fig.colorbar(h, ax=ax, label="log₁₀(count)",
                 fraction=0.045, pad=0.01, shrink=0.85,
                 aspect=28).ax.tick_params(labelsize=7, colors=TEXT_CLR)

    # 1:1 line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "w--", lw=1.0, alpha=0.6)
    ax.set_xlabel(f"Ridge  ({label})", color=TEXT_CLR, fontsize=8)
    ax.set_ylabel(f"RF  ({label})", color=TEXT_CLR, fontsize=8)
    ax.set_title(f"RF vs Ridge (pixel)\nρ = {rho:.3f}",
                 color=TEXT_CLR, fontsize=10, fontweight="bold", pad=7)
    ax.tick_params(colors=TEXT_CLR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#555555")

    fig.suptitle(
        f"{label}  |  {FARM} {YEAR}  |  n={len(rf_v):,} pixels",
        color=TEXT_CLR, fontsize=12, fontweight="bold", y=1.02,
    )

    out_path = OUT_PLOTS / f"rf_pixel_detail_{tgt}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

# ─── 11. Summary statistics table ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("PIXEL PREDICTION SUMMARY")
print("=" * 70)
print(f"  {'Element':<8} {'RF_mean':>9} {'RF_std':>8} "
      f"{'Ridge_mean':>11} {'Ridge_std':>10} "
      f"{'diff_mean':>10} {'ρ(RF,Ri)':>10}")
print("  " + "-" * 70)

for tgt in TARGETS:
    rf_v    = rf_predictions[tgt]
    ri_v    = ridge_predictions[tgt]
    diff    = rf_v - ri_v
    rho, _  = spearmanr(rf_v, ri_v)
    print(f"  {tgt:<8} {rf_v.mean():>9.3f} {rf_v.std():>8.3f} "
          f"{ri_v.mean():>11.3f} {ri_v.std():>10.3f} "
          f"{diff.mean():>10.3f} {rho:>10.3f}")

print()
print(f"Models:  {MODELS_DIR}")
print(f"Plots:   {OUT_PLOTS}")
print(f"Reports: {OUT_RF}")
print()
print("Done.")
