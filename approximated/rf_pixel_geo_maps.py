"""
rf_pixel_geo_maps.py
====================
Pixel-level soil chemistry maps using Random Forest predictors,
rendered in the same style as geo_approx_Агро_Парасат_2023_ph.png.

Key differences from pixel_geo_approx.py (Ridge):
  - MODEL: RandomForestRegressor (instead of Ridge)
  - FEATURES: competitive selection from 10 to RF_MAX_FEATURES (see config)
    — at each step adds the next ranked feature and keeps the model
      with best LOFO-CV Spearman ρ
  - OUTSIDE zone: RF predicts per-pixel using only features available
    in the TIFF (spectral indices computed from bands) + topo/climate
    fields joined from rf_dataset

Configuration block (RF_CONFIG) at the top of the script:
  - RF_ENABLED      : True/False  — enable/disable RF model
  - RF_MIN_FEATURES : int         — minimum number of features to test (default 10)
  - RF_MAX_FEATURES : int         — maximum number of features to test (default 50)
  - RF_N_ESTIMATORS : int         — number of trees (max 100, default 100)
  - RF_STEP         : int         — step size for feature sweep (default 5)
  - RF_RANDOM_STATE : int         — reproducibility seed

Run: python approximated/rf_pixel_geo_maps.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ═══════════════════════════════════════════════════════════════════════════════
# ██  RF CONFIGURATION  ████████████████████████████████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════
RF_CONFIG = {
    # ── Main switch ──────────────────────────────────────────────────────────
    "RF_ENABLED":       True,   # Set False to skip all RF training/prediction

    # ── Feature sweep ────────────────────────────────────────────────────────
    "RF_MIN_FEATURES":  10,     # Start of sweep (min 10 features)
    "RF_MAX_FEATURES":  50,     # End of sweep   (max 50 features)
    "RF_STEP":          5,      # Increment between candidates

    # ── Tree parameters ──────────────────────────────────────────────────────
    "RF_N_ESTIMATORS":  100,    # Number of trees (capped at 100 per user request)
    "RF_MAX_FEATURES":  50,     # Override: also used as forest max_features below
    "RF_TREE_MAX_FEAT": "sqrt", # max_features per split: "sqrt" | "log2" | int

    # ── CV settings ──────────────────────────────────────────────────────────
    "RF_CV_FOLDS":      "field",# "field" = LOFO-CV by field_name (spatial)
                                # "kfold5" = standard 5-fold (faster, less rigorous)
    "RF_RANDOM_STATE":  42,
}
# pull out commonly used values
RF_ENABLED      = RF_CONFIG["RF_ENABLED"]
RF_MIN_FEAT     = RF_CONFIG["RF_MIN_FEATURES"]
RF_MAX_FEAT     = 50              # max features to test in sweep
RF_STEP         = RF_CONFIG["RF_STEP"]
RF_N_EST        = min(RF_CONFIG["RF_N_ESTIMATORS"], 100)   # hard cap at 100
RF_TREE_MFEAT   = RF_CONFIG["RF_TREE_MAX_FEAT"]
RF_CV_FOLDS     = RF_CONFIG["RF_CV_FOLDS"]
RF_SEED         = RF_CONFIG["RF_RANDOM_STATE"]
# ═══════════════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.features import geometry_mask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import contextily as cx
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
FARM        = "Агро Парасат"
YEAR        = 2023
farm_slug   = FARM.replace(" ", "_")

DATA_PATH   = BASE / "data"    / "features" / "full_dataset.csv"
RF_DATA     = BASE / "data"    / "features" / "rf_dataset.csv"
MODELS_DIR  = BASE / "math_statistics" / "output" / "rf" / "rf_models"
TIFF_PATH   = BASE / "approximated" / "tiff" / "s2_2023_summer_mosaic_B4B8B3B5B11.tif"
PIXELS_CSV  = BASE / "approximated" / "tiff" / f"pixels_{farm_slug}_{YEAR}_real.csv"
OUT_DIR     = BASE / "math_statistics" / "output" / "plots"
OUT_RF_DIR  = BASE / "math_statistics" / "output" / "rf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_LABEL = "median-mosaic 2023 May-Sep"
SCALE_M     = 10
TILE_ZOOM   = 14
ALPHA       = 0.62
BLUR_SIGMA  = 3
GRID_N      = 800

WGS84    = "EPSG:4326"
TIFF_CRS = "EPSG:32641"
MERC     = "EPSG:3857"

TARGETS = ["ph", "k", "p", "hu", "s", "no3"]
LOG_TARGETS = {"p", "s", "no3"}
TARGET_LABELS = {"ph":"pH", "k":"K, mg/kg", "p":"P, mg/kg",
                 "hu":"Humus, %", "s":"S, mg/kg", "no3":"NO₃, mg/kg"}
TARGET_CMAPS  = {"ph":"RdYlGn_r", "k":"YlOrRd", "p":"YlGn",
                 "hu":"BrBG", "s":"PuBuGn", "no3":"OrRd"}

print("=" * 70)
print("rf_pixel_geo_maps.py  —  RF Pixel Maps (geo_approx style)")
print("=" * 70)
print(f"\nRF_CONFIG:")
for k, v in RF_CONFIG.items():
    print(f"  {k:<22} = {v}")
print(f"\n  Effective: n_estimators={RF_N_EST}  sweep={RF_MIN_FEAT}..{RF_MAX_FEAT} step={RF_STEP}")

if not RF_ENABLED:
    print("\n[RF_ENABLED=False] Nothing to do. Set RF_ENABLED=True in RF_CONFIG.")
    sys.exit(0)

# ─── 1. Load data ─────────────────────────────────────────────────────────────
print("\nLoading data ...")
df_full = pd.read_csv(DATA_PATH)
rf_df   = pd.read_csv(RF_DATA)

parasat_mask = (
    rf_df["farm"].str.contains("Parasat|Парасат", case=False, na=False) &
    (rf_df["year"] == YEAR)
)
farm_rf = rf_df[parasat_mask].copy()
print(f"  full_dataset: {df_full.shape}  |  rf_dataset Parasat 2023: {len(farm_rf)} rows")

# Field-level geometry for polygon overlays
farm_full = df_full[(df_full["farm"] == FARM) & (df_full["year"] == YEAR)].copy()
agg_geom = (
    farm_full.groupby("field_name")
    .agg(geometry_wkt=("geometry_wkt", "first"))
    .reset_index()
)
gdf = gpd.GeoDataFrame(
    agg_geom,
    geometry=agg_geom["geometry_wkt"].apply(shapely_wkt.loads),
    crs=WGS84,
)

# ─── 2. Load TIFF + pixel CSV ─────────────────────────────────────────────────
if not TIFF_PATH.exists():
    print(f"ERROR: TIFF not found: {TIFF_PATH}")
    sys.exit(1)

with rasterio.open(TIFF_PATH) as src:
    tiff_crs_actual = src.crs.to_string()
    tiff_transform  = src.transform
    tiff_shape      = (src.height, src.width)
    t_tiff_wgs = Transformer.from_crs(tiff_crs_actual, WGS84, always_xy=True)
    b4_full  = src.read(1).astype(float)
    b8_full  = src.read(2).astype(float)
    b3_full  = src.read(3).astype(float)
    b5_full  = src.read(4).astype(float)
    b11_full = src.read(5).astype(float)
    print(f"TIFF: {src.height}×{src.width} px  CRS={src.crs.to_epsg()}")

if not PIXELS_CSV.exists():
    print(f"ERROR: Pixel CSV not found: {PIXELS_CSV}")
    sys.exit(1)

pixels_df = pd.read_csv(PIXELS_CSV)
print(f"Inside pixels: {len(pixels_df):,}")

t_wgs_utm  = Transformer.from_crs(WGS84, TIFF_CRS, always_xy=True)
t_wgs_merc = Transformer.from_crs(WGS84, MERC, always_xy=True)

def to_merc(lo, la):
    return t_wgs_merc.transform(np.asarray(lo), np.asarray(la))

# ─── 3. Feature field-level join for INSIDE pixels ────────────────────────────
print("\nBuilding field-level feature lookup for inside pixels ...")

# Get all unique features across all 6 RF models
all_rf_feats = set()
for tgt in TARGETS:
    with open(MODELS_DIR / f"rf_{tgt}.pkl", "rb") as f:
        b = pickle.load(f)
    all_rf_feats.update(b["features"])
all_rf_feats = sorted(all_rf_feats)
print(f"  Unique features across all 6 models: {len(all_rf_feats)}")

# Compute per-field median from Parasat 2023 rf_dataset
field_feat_map = (
    farm_rf.groupby("field_name")[all_rf_feats]
    .median().reset_index()
)
farm_feat_med = farm_rf[all_rf_feats].median()

# Join to pixels
px_feat = pixels_df.merge(field_feat_map, on="field_name", how="left")
for col in all_rf_feats:
    px_feat[col] = px_feat[col].fillna(farm_feat_med[col])
print(f"  NaN after join: {px_feat[all_rf_feats].isna().sum().sum()}")

# ─── 4. Extract OUTSIDE pixels from TIFF ─────────────────────────────────────
print("\nExtracting outside pixels ...")
gdf_proj = gdf.to_crs(tiff_crs_actual)
field_geoms = list(gdf_proj.geometry)

outside_mask = geometry_mask(
    field_geoms, transform=tiff_transform, invert=False, out_shape=tiff_shape,
)
valid_outside = outside_mask & (b4_full > 0) & (b8_full > 0)
ro, co = np.where(valid_outside)
print(f"  {valid_outside.sum():,} valid outside pixels")

MAX_OUT = 300_000
if len(ro) > MAX_OUT:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(ro), MAX_OUT, replace=False)
    ro, co = ro[idx], co[idx]
    print(f"  Subsampled to {MAX_OUT:,}")

px_x_o, px_y_o = rasterio.transform.xy(tiff_transform, ro, co, offset="center")
px_x_o = np.array(px_x_o, dtype=float)
px_y_o = np.array(px_y_o, dtype=float)
lon_o, lat_o = t_tiff_wgs.transform(px_x_o, px_y_o)

eps = 1e-9
b4r  = b4_full[ro, co]  / 10000.0
b8r  = b8_full[ro, co]  / 10000.0
b3r  = b3_full[ro, co]  / 10000.0
b5r  = b5_full[ro, co]  / 10000.0
b11r = b11_full[ro, co] / 10000.0

ndvi_o  = np.clip((b8r - b4r)  / (b8r + b4r  + eps), -1.0,  1.0)
ndre_o  = np.clip((b8r - b5r)  / (b8r + b5r  + eps), -1.0,  1.0)
gndvi_o = np.clip((b8r - b3r)  / (b8r + b3r  + eps), -1.0,  1.0)
evi_o   = np.clip(2.5*(b8r-b4r)/(b8r + 6*b4r - 7.5*b3r + 1 + eps), -2.0, 2.0)
bsi_o   = np.clip(((b11r+b4r)-(b8r+b3r)) / ((b11r+b4r)+(b8r+b3r)+eps), -1.0, 1.0)

# Build outside_df with spectral indices
outside_df = pd.DataFrame({
    "lon": lon_o, "lat": lat_o, "utm_x": px_x_o, "utm_y": px_y_o,
    "ndvi": ndvi_o, "ndre": ndre_o, "gndvi": gndvi_o, "evi": evi_o, "bsi": bsi_o,
})

# Map pixel index names → possible rf_dataset column names (spring season priority)
PIXEL_TO_FEAT = {
    "ndvi":  ["s2_NDVI_spring",  "s2_NDVI_summer"],
    "ndre":  ["s2_NDRE_spring",  "s2_NDRE_summer"],
    "gndvi": ["s2_GNDVI_spring", "s2_GNDVI_summer"],
    "evi":   ["s2_EVI_spring",   "s2_EVI_summer"],
    "bsi":   ["s2_BSI_spring",   "s2_BSI_summer"],
}

# For outside pixels: fill all 144 features with farm medians, then
# override the 5 spectral indices with real per-pixel values
for col in all_rf_feats:
    outside_df[col] = farm_feat_med[col]   # field-constant fallback

for px_col, candidates in PIXEL_TO_FEAT.items():
    for feat_col in candidates:
        if feat_col in all_rf_feats:
            outside_df[feat_col] = outside_df[px_col].values
            break  # use the first matching one

print(f"  Outside pixels feature matrix: {len(outside_df)} × {len(all_rf_feats)}")

# ─── 5. Mercator coords ───────────────────────────────────────────────────────
mx_in, my_in = to_merc(pixels_df["lon"].values, pixels_df["lat"].values)
pixels_df["mx"] = mx_in
pixels_df["my"] = my_in

mx_o, my_o = to_merc(outside_df["lon"].values, outside_df["lat"].values)
outside_df["mx"] = mx_o
outside_df["my"] = my_o

total_px = len(pixels_df)
xmin, xmax = mx_in.min() - 80, mx_in.max() + 80
ymin, ymax = my_in.min() - 80, my_in.max() + 80

# ─── 6. COMPETITIVE FEATURE SELECTION via LOFO-CV ─────────────────────────────
print("\n" + "=" * 70)
print("  COMPETITIVE RF FEATURE SELECTION  (n_features sweep)")
print("=" * 70)
print(f"  sweep: {RF_MIN_FEAT} → {RF_MAX_FEAT}  step={RF_STEP}  "
      f"n_trees={RF_N_EST}  cv={RF_CV_FOLDS}")

def lofo_cv_rho(X, y, field_ids, n_est, seed):
    """Leave-One-Field-Out CV, returns Spearman rho on OOF predictions."""
    fields = np.unique(field_ids)
    oof_true, oof_pred = [], []
    for fld in fields:
        mask_te = field_ids == fld
        mask_tr = ~mask_te
        if mask_tr.sum() < 20 or mask_te.sum() == 0:
            continue
        rf = RandomForestRegressor(
            n_estimators=n_est, max_features=RF_TREE_MFEAT,
            min_samples_leaf=3, random_state=seed, n_jobs=-1,
        )
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[mask_tr])
        X_te = imp.transform(X[mask_te])
        rf.fit(X_tr, y[mask_tr])
        oof_pred.extend(rf.predict(X_te).tolist())
        oof_true.extend(y[mask_te].tolist())
    if len(oof_true) < 10:
        return np.nan
    rho, _ = spearmanr(oof_true, oof_pred)
    return rho

def kfold5_cv_rho(X, y, n_est, seed):
    """5-fold CV Spearman rho (faster, less rigorous)."""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof_true, oof_pred = [], []
    for tr_idx, te_idx in kf.split(X):
        rf = RandomForestRegressor(
            n_estimators=n_est, max_features=RF_TREE_MFEAT,
            min_samples_leaf=3, random_state=seed, n_jobs=-1,
        )
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[tr_idx])
        X_te = imp.transform(X[te_idx])
        rf.fit(X_tr, y[tr_idx])
        oof_pred.extend(rf.predict(X_te).tolist())
        oof_true.extend(y[te_idx].tolist())
    rho, _ = spearmanr(oof_true, oof_pred)
    return rho

# Store results per target
best_models    = {}   # tgt -> {rf, imputer, best_k, best_rho, features_k, log_transform}
selection_logs = {}   # tgt -> list of (k, rho) tuples for sweep curve

for tgt in TARGETS:
    print(f"\n  [{tgt.upper()}] ─────────────────────────────────────────")

    # Load pre-trained model bundle just for feature ranking order
    with open(MODELS_DIR / f"rf_{tgt}.pkl", "rb") as f:
        bundle = pickle.load(f)

    ranked_features = bundle["features"]     # already sorted by perm importance
    log_transform   = bundle["log_transform"]

    # Use global rf_dataset for training (all valid rows for this target)
    mask_col = f"mask_{tgt}"
    if mask_col in rf_df.columns:
        valid_mask = rf_df[mask_col].astype(bool)
    else:
        valid_mask = rf_df[tgt].notna()

    train_df = rf_df[valid_mask].copy()
    tgt_col  = f"log_{tgt}" if log_transform and f"log_{tgt}" in train_df.columns else tgt
    y_all    = train_df[tgt_col].values.astype(float)
    fids_all = train_df["field_name"].values

    sweep_range = list(range(RF_MIN_FEAT, min(RF_MAX_FEAT + 1, len(ranked_features) + 1), RF_STEP))
    # Always include the full 40-feature set as a candidate
    if len(ranked_features) not in sweep_range:
        sweep_range.append(len(ranked_features))
    sweep_range = sorted(set(sweep_range))

    print(f"    n_train={len(train_df)}  target={tgt_col}  sweep={sweep_range}")

    best_k   = sweep_range[0]
    best_rho = -np.inf
    log_rows = []

    for k in sweep_range:
        feats_k = ranked_features[:k]
        X_k = train_df[feats_k].values.astype(float)

        if RF_CV_FOLDS == "field":
            rho_cv = lofo_cv_rho(X_k, y_all, fids_all, RF_N_EST, RF_SEED)
        else:
            rho_cv = kfold5_cv_rho(X_k, y_all, RF_N_EST, RF_SEED)

        marker = " ← best" if rho_cv > best_rho else ""
        print(f"    k={k:2d}  ρ_cv={rho_cv:+.4f}{marker}")
        log_rows.append((k, rho_cv))

        if rho_cv > best_rho:
            best_rho = rho_cv
            best_k   = k

    selection_logs[tgt] = log_rows

    # Train FINAL model on all valid data using best_k features
    best_feats = ranked_features[:best_k]
    X_final = train_df[best_feats].values.astype(float)
    imp_final = SimpleImputer(strategy="median")
    X_final   = imp_final.fit_transform(X_final)
    rf_final  = RandomForestRegressor(
        n_estimators=RF_N_EST, max_features=RF_TREE_MFEAT,
        min_samples_leaf=3, oob_score=True, random_state=RF_SEED, n_jobs=-1,
    )
    rf_final.fit(X_final, y_all)
    oob_r2 = rf_final.oob_score_

    print(f"    → Best: k={best_k}  ρ_cv={best_rho:+.4f}  OOB_R²={oob_r2:.3f}")
    best_models[tgt] = {
        "rf": rf_final, "imputer": imp_final,
        "features": best_feats, "best_k": best_k,
        "best_rho": best_rho, "oob_r2": oob_r2,
        "log_transform": log_transform,
        "sweep": log_rows,
    }

# ─── 7. Feature selection curve plot ──────────────────────────────────────────
print("\nSaving feature selection sweep plots ...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="#1a1a2e")
fig.suptitle(f"RF Competitive Feature Selection Sweep  |  {FARM} {YEAR}",
             color="white", fontsize=13, fontweight="bold", y=1.01)

for ax_i, tgt in enumerate(TARGETS):
    ax = axes[ax_i // 3][ax_i % 3]
    ax.set_facecolor("#16213e")
    sweep = selection_logs[tgt]
    ks   = [s[0] for s in sweep]
    rhos = [s[1] for s in sweep]
    best_k   = best_models[tgt]["best_k"]
    best_rho = best_models[tgt]["best_rho"]

    ax.plot(ks, rhos, "o-", color="#4fc3f7", lw=2, ms=5)
    ax.axvline(best_k, color="#ff6b6b", lw=1.5, ls="--", alpha=0.8)
    ax.scatter([best_k], [best_rho], color="#ff6b6b", s=80, zorder=5)
    ax.set_title(f"{TARGET_LABELS[tgt]}  |  best k={best_k}  ρ={best_rho:+.3f}",
                 color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("n_features", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("ρ_cv (LOFO)", color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.set_xlim(min(ks) - 1, max(ks) + 1)

fig.tight_layout()
fig.savefig(OUT_DIR / "rf_feature_sweep.png", dpi=150, bbox_inches="tight",
            facecolor="#1a1a2e")
plt.close(fig)
print(f"  Saved: rf_feature_sweep.png")

# ─── 8. Pixel predictions for INSIDE pixels ──────────────────────────────────
print("\nPredicting chemistry for INSIDE pixels ...")
for tgt in TARGETS:
    m = best_models[tgt]
    X = px_feat[m["features"]].values.astype(float)
    X = m["imputer"].transform(X)
    y_hat = m["rf"].predict(X)
    if m["log_transform"]:
        y_hat = np.expm1(y_hat)
        y_hat = np.maximum(y_hat, 0.0)
    pixels_df[f"rf_{tgt}"] = y_hat
    print(f"  {tgt}: mean={y_hat.mean():.3f}  std={y_hat.std():.3f}  "
          f"[best_k={m['best_k']}  ρ_cv={m['best_rho']:+.3f}]")

# ─── 9. Pixel predictions for OUTSIDE pixels ─────────────────────────────────
print("\nPredicting chemistry for OUTSIDE pixels ...")
for tgt in TARGETS:
    m = best_models[tgt]
    X = outside_df[m["features"]].values.astype(float)
    X = m["imputer"].transform(X)
    y_hat = m["rf"].predict(X)
    if m["log_transform"]:
        y_hat = np.expm1(y_hat)
        y_hat = np.maximum(y_hat, 0.0)
    outside_df[f"rf_{tgt}"] = y_hat
    print(f"  {tgt}: mean={y_hat.mean():.3f}  std={y_hat.std():.3f}")

# ─── 10. Map rendering helpers (same style as pixel_geo_approx.py) ────────────
gdf_merc = gdf.to_crs(MERC)

def _draw_validate_polygons(ax, fontsize=5.5):
    for _, row in gdf_merc.iterrows():
        geom = row.geometry
        polys = [geom] if geom.geom_type == "Polygon" else \
                list(geom.geoms) if geom.geom_type == "MultiPolygon" else []
        for poly in polys:
            xp, yp = poly.exterior.xy
            ax.plot(xp, yp, color="black", linewidth=2.0, zorder=6)
            ax.plot(xp, yp, color="white", linewidth=0.7,
                    linestyle="--", alpha=0.55, zorder=7)
        ax.text(geom.centroid.x, geom.centroid.y, "Sampling Zone",
                ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="black",
                          edgecolor="white", linewidth=0.6, alpha=0.78),
                zorder=8)

def _draw_approximate_label(ax, fontsize=9):
    lx = xmin + (xmax - xmin) * 0.03
    ly = ymin + (ymax - ymin) * 0.03
    ax.text(lx, ly, "Approximate\n(model extrapolation)",
            ha="left", va="bottom",
            fontsize=fontsize, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111",
                      edgecolor="#999999", linewidth=1.0, alpha=0.88),
            zorder=9)

_gx = np.linspace(xmin, xmax, GRID_N)
_gy = np.linspace(ymin, ymax, GRID_N)
grid_x, grid_y = np.meshgrid(_gx, _gy)

def _make_smooth_grid(col_name, df_src):
    sub = df_src[["mx", "my", col_name]].dropna()
    if len(sub) < 4:
        return None
    if len(sub) > 80_000:
        idx = np.random.default_rng(0).choice(len(sub), 80_000, replace=False)
        sub = sub.iloc[idx]
    gz = griddata(
        points=(sub["mx"].values, sub["my"].values),
        values=sub[col_name].values,
        xi=(grid_x, grid_y), method="nearest",
    )
    nan_mask = np.isnan(gz)
    if nan_mask.all():
        return None
    gz_filled = np.where(nan_mask, np.nanmedian(gz), gz)
    gz_smooth = gaussian_filter(gz_filled, sigma=BLUR_SIGMA)
    gz_smooth[nan_mask] = np.nan
    return gz_smooth

# ─── 11. Render per-element maps (same style as geo_approx) ───────────────────
print("\nRendering RF chemistry heatmaps ...")

for tgt in TARGETS:
    m        = best_models[tgt]
    in_col   = f"rf_{tgt}"
    label    = TARGET_LABELS[tgt]
    cmap_    = TARGET_CMAPS[tgt]

    sub_in = pixels_df[["mx", "my", in_col]].dropna()
    if len(sub_in) == 0:
        print(f"  Skip {tgt}: no inside data")
        continue

    vals_in = sub_in[in_col]
    v0 = vals_in.quantile(0.02)
    v1 = vals_in.quantile(0.98)

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#111111")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery,
                   zoom=TILE_ZOOM, alpha=1.0)

    # Smooth outside surface
    gz = _make_smooth_grid(in_col, outside_df)
    if gz is not None:
        ax.imshow(gz, extent=[xmin, xmax, ymin, ymax],
                  origin="lower", aspect="auto",
                  cmap=cmap_, vmin=v0, vmax=v1,
                  alpha=ALPHA * 0.75, interpolation="nearest", zorder=2)
        _draw_approximate_label(ax)

    # Real pixels inside fields
    ax.scatter(sub_in["mx"], sub_in["my"],
               c=vals_in, cmap=cmap_, s=0.8, alpha=ALPHA,
               linewidths=0, vmin=v0, vmax=v1,
               zorder=3, rasterized=True)

    _draw_validate_polygons(ax)

    sm = ScalarMappable(cmap=cmap_, norm=Normalize(v0, v1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.70, pad=0.02, aspect=22)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)
    cb.outline.set_edgecolor("#333333")
    cb.set_label(label, color="white", fontsize=7, labelpad=6)

    ax.set_title(label, fontsize=12, fontweight="bold", color="white", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")

    foot = (f"  |  RF({RF_N_EST} trees, {m['best_k']} features)"
            f"  |  ρ_cv={m['best_rho']:+.3f}  OOB_R²={m['oob_r2']:.3f}"
            f"  |  competitive sweep {RF_MIN_FEAT}→{RF_MAX_FEAT}")
    fig.text(
        0.5, 0.01,
        f"{FARM}  |  {total_px:,} real Sentinel-2 pixels @ 10m  |  {IMAGE_LABEL}"
        + foot,
        ha="center", va="bottom", color="#666666", fontsize=7.5,
    )
    fig.tight_layout(pad=0.4, rect=[0, 0.03, 1, 1])

    fname = f"rf_geo_{farm_slug}_{YEAR}_{tgt}.png"
    fig.savefig(OUT_DIR / fname, dpi=200, bbox_inches="tight",
                facecolor="#0a0a0a")
    plt.close(fig)
    print(f"  Saved: {fname}")

# ─── 12. Summary figure (1×7: NDVI + 6 elements) ─────────────────────────────
print("\nRendering summary figure (1×7) ...")

# NDVI (raw, no model)
sub_ndvi = pixels_df[["mx", "my", "ndvi"]].dropna()
v0n = sub_ndvi["ndvi"].quantile(0.02)
v1n = sub_ndvi["ndvi"].quantile(0.98)
outside_df["ndvi_col"] = outside_df["ndvi"]

fig, axes = plt.subplots(1, 7, figsize=(42, 7))
fig.patch.set_facecolor("#0a0a0a")

panel_cols  = ["ndvi"]   + [f"rf_{t}" for t in TARGETS]
panel_labs  = ["NDVI (real S2)"] + [TARGET_LABELS[t] for t in TARGETS]
panel_cmaps = ["RdYlGn"] + [TARGET_CMAPS[t] for t in TARGETS]
inside_src  = ["ndvi"]   + [f"rf_{t}" for t in TARGETS]   # column in pixels_df
outside_src = ["ndvi"]   + [f"rf_{t}" for t in TARGETS]   # column in outside_df

for i, (vc_in, vc_out, lb, cm) in enumerate(
        zip(inside_src, outside_src, panel_labs, panel_cmaps)):
    ax = axes[i]
    ax.set_facecolor("#111111")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zoom=TILE_ZOOM, alpha=1.0)

    # Column name in pixels_df
    col_in_df = vc_in
    sub_in = pixels_df[["mx", "my", col_in_df]].dropna()
    if len(sub_in) == 0:
        ax.set_title(lb, color="#555", fontsize=8)
        continue

    vals_in = sub_in[col_in_df]
    v0, v1  = vals_in.quantile(0.02), vals_in.quantile(0.98)

    # Smooth outside
    out_col = vc_out
    if out_col in outside_df.columns:
        gz = _make_smooth_grid(out_col, outside_df)
        if gz is not None:
            ax.imshow(gz, extent=[xmin, xmax, ymin, ymax],
                      origin="lower", aspect="auto",
                      cmap=cm, vmin=v0, vmax=v1,
                      alpha=ALPHA * 0.70, interpolation="nearest", zorder=2)
            _draw_approximate_label(ax, fontsize=5.5)

    ax.scatter(sub_in["mx"], sub_in["my"],
               c=vals_in, cmap=cm, s=0.5, alpha=ALPHA,
               linewidths=0, vmin=v0, vmax=v1,
               zorder=3, rasterized=True)
    _draw_validate_polygons(ax, fontsize=4.5)

    sm = ScalarMappable(cmap=cm, norm=Normalize(v0, v1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.70, pad=0.02, aspect=22)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
    cb.outline.set_edgecolor("#333333")
    cb.set_label(lb, color="white", fontsize=5, labelpad=4)

    rho_note = ""
    if lb != "NDVI (real S2)" and TARGETS[max(i - 1, 0)] in best_models:
        tgt_k = TARGETS[i - 1]
        m = best_models[tgt_k]
        rho_note = f"\nρ_cv={m['best_rho']:+.3f}  k={m['best_k']}"
    ax.set_title(lb + rho_note, fontsize=7.5, fontweight="bold",
                 color="white", pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")

fig.suptitle(
    f"RF soil chemistry — {FARM}, {YEAR}  |  {total_px:,} real S2 pixels @ 10m  |"
    f"  RF({RF_N_EST} trees) · competitive selection {RF_MIN_FEAT}→{RF_MAX_FEAT} features",
    fontsize=10, color="white", y=1.008,
)
fig.tight_layout(pad=0.5)
fname_sum = f"rf_geo_{farm_slug}_{YEAR}_summary.png"
fig.savefig(OUT_DIR / fname_sum, dpi=150, bbox_inches="tight",
            facecolor="#0a0a0a")
plt.close(fig)
print(f"  Saved: {fname_sum}")

# ─── 13. Save selection summary CSV ───────────────────────────────────────────
rows = []
for tgt in TARGETS:
    m = best_models[tgt]
    rows.append({
        "element":    tgt,
        "best_k":     m["best_k"],
        "rho_cv":     round(m["best_rho"], 4),
        "oob_r2":     round(m["oob_r2"], 4),
        "n_trees":    RF_N_EST,
        "log_transform": m["log_transform"],
        "top_features": ", ".join(m["features"][:5]),
    })
sel_df = pd.DataFrame(rows)
sel_df.to_csv(OUT_RF_DIR / "rf_feature_selection_best.csv", index=False)
print(f"\nSaved: rf_feature_selection_best.csv")

# ─── 14. Final summary ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPETITIVE SELECTION RESULTS")
print("=" * 70)
print(f"  {'Element':<8} {'best_k':>6} {'ρ_cv':>8} {'OOB_R²':>8}  top feature")
print("  " + "-" * 60)
for tgt in TARGETS:
    m = best_models[tgt]
    print(f"  {tgt:<8} {m['best_k']:>6} {m['best_rho']:>8.4f} {m['oob_r2']:>8.4f}"
          f"  {m['features'][0]}")

print()
print(f"RF_CONFIG used:")
for k, v in RF_CONFIG.items():
    print(f"  {k} = {v}")
print(f"\nOutputs → {OUT_DIR}")
print("Done.")
