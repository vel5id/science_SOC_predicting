"""
rf_ensemble_maps.py
===================
Two ensemble approaches to combine global RF signal with local spatial detail:

1. STACKING: weighted blend of RF (global) + Ridge (local geo-spectral)
   - RF captures between-farm gradients via 40 multi-source features
   - Ridge captures within-field spatial trends via spectral + UTM coords
   - Optimal weight α found by LOFO-CV within the farm:
       y_blend = α·RF + (1-α)·Ridge

2. KRIGING RESIDUALS: RF prediction + spatial interpolation of RF errors
   - RF predicts field-level average → flat within field
   - Residuals (actual - RF_pred) at sampling points carry local detail
   - RBF interpolation of residuals → continuous surface
   - Final: y_krig = RF_pred + RBF_interpolated_residual(utm_x, utm_y)

Both are rendered in geo_approx.py style (satellite basemap + outside zone).

Run: python approximated/rf_ensemble_maps.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")
import pickle, time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import contextily as cx
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
DATA_PATH  = BASE / "data" / "features" / "full_dataset.csv"
RF_DATA    = BASE / "data" / "features" / "rf_dataset.csv"
MODELS_DIR = BASE / "math_statistics" / "output" / "rf" / "rf_models"
TIFF_PATH  = BASE / "approximated" / "tiff" / "s2_2023_summer_mosaic_B4B8B3B5B11.tif"
PIXELS_CSV = BASE / "approximated" / "tiff" / "pixels_Агро_Парасат_2023_real.csv"
OUT_DIR    = BASE / "math_statistics" / "output" / "plots"
OUT_RF     = BASE / "math_statistics" / "output" / "rf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FARM = "Агро Парасат"
YEAR = 2023
farm_slug = FARM.replace(" ", "_")
IMAGE_LABEL = "median-mosaic 2023 May-Sep"

TARGETS = ["ph", "k", "p", "hu", "s", "no3"]
LOG_TARGETS = {"p", "s", "no3"}
TARGET_LABELS = {"ph":"pH","k":"K, mg/kg","p":"P, mg/kg",
                 "hu":"Humus, %","s":"S, mg/kg","no3":"NO₃, mg/kg"}
TARGET_CMAPS = {"ph":"RdYlGn_r","k":"YlOrRd","p":"YlGn",
                "hu":"BrBG","s":"PuBuGn","no3":"OrRd"}

# Ridge config (same as pixel_geo_approx.py)
BEST_PREDICTOR = {
    "ph":  [("s2_NDRE_spring","ndre"),("s2_BSI_spring","bsi")],
    "k":   [("s2_BSI_spring","bsi"),("s2_NDRE_spring","ndre")],
    "p":   [("s2_GNDVI_spring","gndvi"),("s2_BSI_spring","bsi")],
    "hu":  [("s2_EVI_summer","evi"),("s2_NDRE_spring","ndre")],
    "s":   [("s2_GNDVI_autumn","gndvi"),("s2_BSI_spring","bsi")],
    "no3": [("s2_GNDVI_spring","gndvi"),("s2_EVI_summer","evi")],
}

TILE_ZOOM = 14
ALPHA_MAP = 0.62
BLUR_SIGMA = 3
GRID_N = 800

WGS84 = "EPSG:4326"
TIFF_CRS = "EPSG:32641"
MERC = "EPSG:3857"

print("=" * 70)
print("rf_ensemble_maps.py — Stacking + Kriging Residuals")
print("=" * 70)

# ─── Load data ────────────────────────────────────────────────────────────────
print("\nLoading data ...")
df_full = pd.read_csv(DATA_PATH)
rf_df   = pd.read_csv(RF_DATA)
pixels_df = pd.read_csv(PIXELS_CSV)

farm_full = df_full[(df_full["farm"] == FARM) & (df_full["year"] == YEAR)].copy()
parasat_rf = rf_df[rf_df["farm"].str.contains("Parasat|Парасат", case=False, na=False) &
                   (rf_df["year"] == YEAR)].copy()

print(f"  Farm points: {len(farm_full)}, Pixels: {len(pixels_df):,}")

# Field geometry
agg_geom = farm_full.groupby("field_name").agg(
    geometry_wkt=("geometry_wkt","first")).reset_index()
gdf = gpd.GeoDataFrame(agg_geom,
    geometry=agg_geom["geometry_wkt"].apply(shapely_wkt.loads), crs=WGS84)

# ─── TIFF + outside pixels ───────────────────────────────────────────────────
with rasterio.open(TIFF_PATH) as src:
    tiff_crs = src.crs.to_string()
    tiff_transform = src.transform
    tiff_shape = (src.height, src.width)
    t_tiff_wgs = Transformer.from_crs(tiff_crs, WGS84, always_xy=True)
    b4 = src.read(1).astype(float)
    b8 = src.read(2).astype(float)
    b3 = src.read(3).astype(float)
    b5 = src.read(4).astype(float)
    b11 = src.read(5).astype(float)

t_wgs_utm  = Transformer.from_crs(WGS84, TIFF_CRS, always_xy=True)
t_wgs_merc = Transformer.from_crs(WGS84, MERC, always_xy=True)

gdf_proj = gdf.to_crs(tiff_crs)
outside_mask = geometry_mask(list(gdf_proj.geometry), transform=tiff_transform,
                             invert=False, out_shape=tiff_shape)
valid_out = outside_mask & (b4 > 0) & (b8 > 0)
ro, co = np.where(valid_out)
MAX_OUT = 300_000
if len(ro) > MAX_OUT:
    idx = np.random.default_rng(42).choice(len(ro), MAX_OUT, replace=False)
    ro, co = ro[idx], co[idx]

px_x_o, px_y_o = rasterio.transform.xy(tiff_transform, ro, co, offset="center")
px_x_o, px_y_o = np.array(px_x_o), np.array(px_y_o)
lon_o, lat_o = t_tiff_wgs.transform(px_x_o, px_y_o)
eps = 1e-9
b4r, b8r, b3r = b4[ro,co]/10000, b8[ro,co]/10000, b3[ro,co]/10000
b5r, b11r = b5[ro,co]/10000, b11[ro,co]/10000
ndvi_o  = np.clip((b8r-b4r)/(b8r+b4r+eps), -1, 1)
ndre_o  = np.clip((b8r-b5r)/(b8r+b5r+eps), -1, 1)
gndvi_o = np.clip((b8r-b3r)/(b8r+b3r+eps), -1, 1)
evi_o   = np.clip(2.5*(b8r-b4r)/(b8r+6*b4r-7.5*b3r+1+eps), -2, 2)
bsi_o   = np.clip(((b11r+b4r)-(b8r+b3r))/((b11r+b4r)+(b8r+b3r)+eps), -1, 1)

outside_df = pd.DataFrame({"lon":lon_o,"lat":lat_o,"utm_x":px_x_o,"utm_y":px_y_o,
    "ndvi":ndvi_o,"ndre":ndre_o,"gndvi":gndvi_o,"evi":evi_o,"bsi":bsi_o})

# Mercator coords
def to_merc(lo, la):
    return t_wgs_merc.transform(np.asarray(lo), np.asarray(la))

mx_in, my_in = to_merc(pixels_df["lon"].values, pixels_df["lat"].values)
pixels_df["mx"] = mx_in; pixels_df["my"] = my_in
mx_o, my_o = to_merc(lon_o, lat_o)
outside_df["mx"] = mx_o; outside_df["my"] = my_o

xmin, xmax = mx_in.min()-80, mx_in.max()+80
ymin, ymax = my_in.min()-80, my_in.max()+80

# UTM coords for inside pixels
ux_px, uy_px = t_wgs_utm.transform(pixels_df["lon"].values, pixels_df["lat"].values)

# Training points UTM
ux_tr, uy_tr = t_wgs_utm.transform(farm_full["centroid_lon"].values,
                                     farm_full["centroid_lat"].values)
ux_min, ux_max = ux_tr.min(), ux_tr.max()
uy_min, uy_max = uy_tr.min(), uy_tr.max()

def norm_utm(ux, uy):
    return ((ux - ux_min)/(ux_max - ux_min + 1e-9),
            (uy - uy_min)/(uy_max - uy_min + 1e-9))

nx_tr, ny_tr = norm_utm(ux_tr, uy_tr)
nx_px, ny_px = norm_utm(ux_px, uy_px)
nx_o, ny_o   = norm_utm(px_x_o, px_y_o)

# RF features for inside pixels (field-level join)
all_rf_feats = set()
for tgt in TARGETS:
    with open(MODELS_DIR / f"rf_{tgt}.pkl", "rb") as f:
        b = pickle.load(f)
    all_rf_feats.update(b["features"])
all_rf_feats = sorted(all_rf_feats)

field_feat_map = parasat_rf.groupby("field_name")[all_rf_feats].median().reset_index()
farm_feat_med  = parasat_rf[all_rf_feats].median()
px_feat = pixels_df.merge(field_feat_map, on="field_name", how="left")
for col in all_rf_feats:
    px_feat[col] = px_feat[col].fillna(farm_feat_med[col])

# RF features for outside pixels (farm median + pixel spectral)
PIXEL_TO_FEAT = {"ndvi":["s2_NDVI_spring"],"ndre":["s2_NDRE_spring"],
    "gndvi":["s2_GNDVI_spring"],"evi":["s2_EVI_spring"],"bsi":["s2_BSI_spring"]}
for col in all_rf_feats:
    outside_df[col] = farm_feat_med[col]
for px_col, cands in PIXEL_TO_FEAT.items():
    for fc in cands:
        if fc in all_rf_feats:
            outside_df[fc] = outside_df[px_col].values; break

# ─── Geo helpers ──────────────────────────────────────────────────────────────
gdf_merc = gdf.to_crs(MERC)

def _draw_polygons(ax, fontsize=5.5):
    for _, row in gdf_merc.iterrows():
        geom = row.geometry
        polys = [geom] if geom.geom_type == "Polygon" else \
                list(geom.geoms) if geom.geom_type == "MultiPolygon" else []
        for poly in polys:
            xp, yp = poly.exterior.xy
            ax.plot(xp, yp, color="black", lw=2.0, zorder=6)
            ax.plot(xp, yp, color="white", lw=0.7, ls="--", alpha=0.55, zorder=7)
        ax.text(geom.centroid.x, geom.centroid.y, "Sampling Zone",
                ha="center", va="center", fontsize=fontsize, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="black",
                          edgecolor="white", lw=0.6, alpha=0.78), zorder=8)

def _draw_approx_label(ax, fontsize=9):
    ax.text(xmin+(xmax-xmin)*0.03, ymin+(ymax-ymin)*0.03,
            "Approximate\n(model extrapolation)",
            ha="left", va="bottom", fontsize=fontsize, fontweight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor="#111",
            edgecolor="#999", lw=1.0, alpha=0.88), zorder=9)

_gx = np.linspace(xmin, xmax, GRID_N)
_gy = np.linspace(ymin, ymax, GRID_N)
grid_x, grid_y = np.meshgrid(_gx, _gy)

def _smooth_grid(col, df_src):
    sub = df_src[["mx","my",col]].dropna()
    if len(sub) < 4: return None
    if len(sub) > 80_000:
        sub = sub.iloc[np.random.default_rng(0).choice(len(sub), 80000, replace=False)]
    gz = griddata((sub["mx"].values, sub["my"].values), sub[col].values,
                  (grid_x, grid_y), method="nearest")
    m = np.isnan(gz)
    if m.all(): return None
    gz[m] = np.nanmedian(gz)
    gz = gaussian_filter(gz, sigma=BLUR_SIGMA)
    gz[m] = np.nan
    return gz

# ─── MAIN LOOP: build 3 models per element ───────────────────────────────────
print("\n" + "=" * 70)
print("  BUILDING ENSEMBLE MODELS")
print("=" * 70)

results = {}  # tgt -> {ridge_pred, rf_pred, stack_pred, krig_pred, ...}

for tgt in TARGETS:
    print(f"\n── {tgt.upper()} ({TARGET_LABELS[tgt]}) ──────────────────────")

    y_train = farm_full[tgt].values
    valid   = ~np.isnan(y_train)

    # ── A) RIDGE prediction (same as pixel_geo_approx.py) ─────────────────
    csv_cols = [p[0] for p in BEST_PREDICTOR[tgt]]
    px_cols  = [p[1] for p in BEST_PREDICTOR[tgt]]

    spec_tr = np.column_stack([farm_full[c].values for c in csv_cols])
    X_ridge_tr = np.column_stack([spec_tr, nx_tr, ny_tr, nx_tr*ny_tr])
    mask_r = valid & np.all(~np.isnan(X_ridge_tr), axis=1)
    pipe = Pipeline([("scaler",StandardScaler()),("ridge",Ridge(alpha=10.0))])
    pipe.fit(X_ridge_tr[mask_r], y_train[mask_r])

    # Predict inside
    spec_px = np.column_stack([pixels_df[pc].values for pc in px_cols])
    X_ridge_px = np.column_stack([spec_px, nx_px, ny_px, nx_px*ny_px])
    ridge_px = pipe.predict(X_ridge_px)

    # Predict outside
    spec_out = np.column_stack([outside_df[pc].values for pc in px_cols])
    X_ridge_out = np.column_stack([spec_out, nx_o, ny_o, nx_o*ny_o])
    ridge_out = pipe.predict(X_ridge_out)

    ridge_tr_pred = pipe.predict(X_ridge_tr[mask_r])
    rho_ridge, _ = spearmanr(y_train[mask_r], ridge_tr_pred)
    print(f"  Ridge: ρ_train={rho_ridge:+.3f}")

    # ── B) RF prediction (global model) ────────────────────────────────────
    with open(MODELS_DIR / f"rf_{tgt}.pkl", "rb") as f:
        bundle = pickle.load(f)
    rf_model    = bundle["rf"]
    imputer     = bundle["imputer"]
    rf_features = bundle["features"]
    log_tf      = bundle["log_transform"]

    # Inside
    X_rf_px = imputer.transform(px_feat[rf_features].values)
    rf_px = rf_model.predict(X_rf_px)
    if log_tf: rf_px = np.maximum(np.expm1(rf_px), 0)

    # Outside
    X_rf_out = imputer.transform(outside_df[rf_features].values)
    rf_out = rf_model.predict(X_rf_out)
    if log_tf: rf_out = np.maximum(np.expm1(rf_out), 0)

    # RF at training points (for residuals)
    rf_features_train = parasat_rf.groupby("field_name")[rf_features].median()
    rf_at_train = []
    for _, row in farm_full.iterrows():
        fn = row["field_name"]
        if fn in rf_features_train.index:
            x_row = rf_features_train.loc[fn].values.reshape(1, -1)
        else:
            x_row = farm_feat_med[rf_features].values.reshape(1, -1)
        x_row = imputer.transform(x_row.astype(float))
        pred = rf_model.predict(x_row)[0]
        if log_tf: pred = max(np.expm1(pred), 0)
        rf_at_train.append(pred)
    rf_at_train = np.array(rf_at_train)

    rho_rf, _ = spearmanr(y_train[valid], rf_at_train[valid])
    print(f"  RF:    ρ_train={rho_rf:+.3f}")

    # ── C) STACKING: find optimal α ───────────────────────────────────────
    # LOFO-CV within farm to find best α
    fields_farm = farm_full["field_name"].unique()
    best_alpha = 0.5
    best_rho_stack = -np.inf

    for alpha in np.arange(0.0, 1.05, 0.05):
        oof_true, oof_blend = [], []
        for fld in fields_farm:
            te_mask = (farm_full["field_name"] == fld).values
            tr_mask = ~te_mask & mask_r

            if tr_mask.sum() < 5 or te_mask.sum() == 0:
                continue

            # Retrain Ridge on fold
            pipe_f = Pipeline([("scaler",StandardScaler()),("ridge",Ridge(alpha=10.0))])
            pipe_f.fit(X_ridge_tr[tr_mask], y_train[tr_mask])
            ridge_te = pipe_f.predict(X_ridge_tr[te_mask & mask_r])

            rf_te = rf_at_train[te_mask & mask_r]
            y_te  = y_train[te_mask & mask_r]

            blend = alpha * rf_te + (1 - alpha) * ridge_te
            oof_true.extend(y_te.tolist())
            oof_blend.extend(blend.tolist())

        if len(oof_true) > 5:
            rho_cv, _ = spearmanr(oof_true, oof_blend)
            if rho_cv > best_rho_stack:
                best_rho_stack = rho_cv
                best_alpha = alpha

    # Final stacking predictions
    stack_px  = best_alpha * rf_px  + (1 - best_alpha) * ridge_px
    stack_out = best_alpha * rf_out + (1 - best_alpha) * ridge_out
    stack_tr  = best_alpha * rf_at_train + (1 - best_alpha) * ridge_tr_pred if len(ridge_tr_pred)==len(rf_at_train) else stack_px[:len(y_train)]

    rho_stack_insample = spearmanr(y_train[mask_r],
                                    best_alpha*rf_at_train[mask_r]+(1-best_alpha)*ridge_tr_pred)[0]
    print(f"  Stack: α_RF={best_alpha:.2f}  ρ_cv={best_rho_stack:+.3f}  ρ_train={rho_stack_insample:+.3f}")

    # ── D) KRIGING RESIDUALS: RBF interpolation of RF errors ───────────────
    residuals = y_train - rf_at_train   # actual - RF_predicted
    valid_res = valid & ~np.isnan(residuals)

    # UTM coordinates of training points
    train_coords = np.column_stack([ux_tr[valid_res], uy_tr[valid_res]])
    res_values   = residuals[valid_res]

    # RBF interpolation (thin_plate_spline = smooth, no nugget noise)
    rbf = RBFInterpolator(train_coords, res_values, kernel="thin_plate_spline",
                           smoothing=1.0)

    # Interpolate residuals at inside pixel locations
    px_coords = np.column_stack([ux_px, uy_px])
    res_interp_px = rbf(px_coords)
    krig_px = rf_px + res_interp_px

    # Outside pixels
    out_coords = np.column_stack([px_x_o, px_y_o])
    res_interp_out = rbf(out_coords)
    krig_out = rf_out + res_interp_out

    krig_at_train = rf_at_train + rbf(np.column_stack([ux_tr, uy_tr]))
    rho_krig, _ = spearmanr(y_train[valid], krig_at_train[valid])
    print(f"  Krig:  ρ_train={rho_krig:+.3f}  (RF + RBF residuals)")

    # LOFO-CV for kriging
    oof_true_k, oof_krig = [], []
    for fld in fields_farm:
        te_mask = (farm_full["field_name"] == fld).values
        tr_mask = ~te_mask & valid_res
        if tr_mask.sum() < 5 or (te_mask & valid_res).sum() == 0:
            continue
        rbf_f = RBFInterpolator(
            np.column_stack([ux_tr[tr_mask], uy_tr[tr_mask]]),
            residuals[tr_mask], kernel="thin_plate_spline", smoothing=1.0)
        te_pts = np.column_stack([ux_tr[te_mask & valid], uy_tr[te_mask & valid]])
        res_f = rbf_f(te_pts)
        pred_f = rf_at_train[te_mask & valid] + res_f
        oof_true_k.extend(y_train[te_mask & valid].tolist())
        oof_krig.extend(pred_f.tolist())
    rho_krig_cv = spearmanr(oof_true_k, oof_krig)[0] if len(oof_true_k) > 5 else np.nan
    print(f"  Krig:  ρ_cv={rho_krig_cv:+.3f}  (LOFO-CV)")

    # Clip negative values for log targets
    if log_tf:
        krig_px  = np.maximum(krig_px, 0)
        krig_out = np.maximum(krig_out, 0)
        stack_px = np.maximum(stack_px, 0)
        stack_out= np.maximum(stack_out, 0)

    results[tgt] = {
        "ridge_px": ridge_px, "ridge_out": ridge_out,
        "rf_px": rf_px, "rf_out": rf_out,
        "stack_px": stack_px, "stack_out": stack_out,
        "krig_px": krig_px, "krig_out": krig_out,
        "alpha": best_alpha,
        "rho_ridge": rho_ridge, "rho_rf": rho_rf,
        "rho_stack_cv": best_rho_stack, "rho_krig_cv": rho_krig_cv,
        "rho_krig_train": rho_krig,
    }

# ─── SUMMARY TABLE ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("WITHIN-FARM COMPARISON (Агро Парасат, n=151)")
print("=" * 70)
print(f"  {'Element':<8} {'Ridge':>8} {'RF':>8} {'Stack':>10} {'Krig':>10}")
print(f"  {'':8} {'ρ_train':>8} {'ρ_train':>8} {'α  ρ_cv':>10} {'ρ_cv':>10}")
print("  " + "-" * 50)
for tgt in TARGETS:
    r = results[tgt]
    print(f"  {tgt:<8} {r['rho_ridge']:>+8.3f} {r['rho_rf']:>+8.3f} "
          f"{r['alpha']:.2f} {r['rho_stack_cv']:>+6.3f} {r['rho_krig_cv']:>+10.3f}")

# ─── RENDER MAPS ──────────────────────────────────────────────────────────────
print("\n\nRendering maps ...")

def render_single(ax, vals_in, vals_out, cmap_, v0, v1, title, footer=""):
    ax.set_facecolor("#111111")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zoom=TILE_ZOOM, alpha=1.0)
    # Outside smooth
    _df = pd.DataFrame({"mx":mx_o,"my":my_o,"v":vals_out})
    gz = _smooth_grid("v", _df)
    if gz is not None:
        ax.imshow(gz, extent=[xmin,xmax,ymin,ymax], origin="lower", aspect="auto",
                  cmap=cmap_, vmin=v0, vmax=v1, alpha=ALPHA_MAP*0.75,
                  interpolation="nearest", zorder=2)
    # Inside pixels
    ax.scatter(mx_in, my_in, c=vals_in, cmap=cmap_, s=0.8, alpha=ALPHA_MAP,
               linewidths=0, vmin=v0, vmax=v1, zorder=3, rasterized=True)
    _draw_polygons(ax, fontsize=4.5)
    sm = ScalarMappable(cmap=cmap_, norm=Normalize(v0,v1)); sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.70, pad=0.02, aspect=22)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
    cb.outline.set_edgecolor("#333333")
    ax.set_title(title, fontsize=8, fontweight="bold", color="white", pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("#222222")

# Per-element: 1×4 comparison (Ridge | RF | Stack | Kriging)
for tgt in TARGETS:
    r = results[tgt]
    label = TARGET_LABELS[tgt]
    cmap  = TARGET_CMAPS[tgt]

    all_vals = np.concatenate([r["ridge_px"], r["rf_px"], r["stack_px"], r["krig_px"]])
    v0, v1 = np.nanpercentile(all_vals, 2), np.nanpercentile(all_vals, 98)

    fig, axes = plt.subplots(1, 4, figsize=(36, 9))
    fig.patch.set_facecolor("#0a0a0a")

    panels = [
        (r["ridge_px"], r["ridge_out"],
         f"Ridge (geo-spectral)\n{label}\nρ_train={r['rho_ridge']:+.3f}"),
        (r["rf_px"], r["rf_out"],
         f"Random Forest (global)\n{label}\nρ_train={r['rho_rf']:+.3f}"),
        (r["stack_px"], r["stack_out"],
         f"Stacking (α_RF={r['alpha']:.2f})\n{label}\nρ_cv={r['rho_stack_cv']:+.3f}"),
        (r["krig_px"], r["krig_out"],
         f"RF + Kriging residuals\n{label}\nρ_cv={r['rho_krig_cv']:+.3f}"),
    ]

    for ax, (vin, vout, ttl) in zip(axes, panels):
        render_single(ax, vin, vout, cmap, v0, v1, ttl)

    fig.suptitle(
        f"{label}  |  {FARM} {YEAR}  |  {len(pixels_df):,} pixels @ 10m",
        color="white", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR / f"ensemble_{farm_slug}_{YEAR}_{tgt}.png",
                dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)
    print(f"  Saved: ensemble_{farm_slug}_{YEAR}_{tgt}.png")

# Summary 1×7 (NDVI + 6 krig)
print("\nRendering summary figure (1×7 Kriging) ...")
fig, axes = plt.subplots(1, 7, figsize=(42, 7))
fig.patch.set_facecolor("#0a0a0a")

# NDVI panel
ax = axes[0]
ax.set_facecolor("#111111")
ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zoom=TILE_ZOOM, alpha=1.0)
sub_ndvi = pixels_df[["mx","my","ndvi"]].dropna()
v0n, v1n = sub_ndvi["ndvi"].quantile(0.02), sub_ndvi["ndvi"].quantile(0.98)
ax.scatter(sub_ndvi["mx"], sub_ndvi["my"], c=sub_ndvi["ndvi"], cmap="RdYlGn",
           s=0.5, alpha=ALPHA_MAP, linewidths=0, vmin=v0n, vmax=v1n, zorder=3, rasterized=True)
_draw_polygons(ax, fontsize=4.5)
sm = ScalarMappable(cmap="RdYlGn", norm=Normalize(v0n,v1n)); sm.set_array([])
cb = plt.colorbar(sm, ax=ax, shrink=0.70, pad=0.02, aspect=22)
cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
cb.outline.set_edgecolor("#333333")
ax.set_title("NDVI (real S2)", fontsize=7.5, fontweight="bold", color="white", pad=4)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_edgecolor("#222222")

# Chemistry panels (Kriging)
for i, tgt in enumerate(TARGETS):
    ax = axes[i+1]
    r = results[tgt]
    label = TARGET_LABELS[tgt]
    cmap  = TARGET_CMAPS[tgt]
    v0 = np.nanpercentile(r["krig_px"], 2)
    v1 = np.nanpercentile(r["krig_px"], 98)
    render_single(ax, r["krig_px"], r["krig_out"], cmap, v0, v1,
                  f"{label}\nρ_cv={r['rho_krig_cv']:+.3f}")

fig.suptitle(
    f"RF + Kriging Residuals  |  {FARM}, {YEAR}  |  "
    f"{len(pixels_df):,} S2 pixels @ 10m",
    fontsize=10, color="white", y=1.008)
fig.tight_layout(pad=0.5)
fig.savefig(OUT_DIR / f"ensemble_{farm_slug}_{YEAR}_summary_krig.png",
            dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig)
print(f"  Saved: ensemble_{farm_slug}_{YEAR}_summary_krig.png")

# ─── Save metrics CSV ─────────────────────────────────────────────────────────
rows = []
for tgt in TARGETS:
    r = results[tgt]
    rows.append({"element":tgt,"alpha_rf":r["alpha"],
        "rho_ridge_train":round(r["rho_ridge"],4),
        "rho_rf_train":round(r["rho_rf"],4),
        "rho_stack_cv":round(r["rho_stack_cv"],4),
        "rho_krig_cv":round(r["rho_krig_cv"],4),
        "rho_krig_train":round(r["rho_krig_train"],4)})
pd.DataFrame(rows).to_csv(OUT_RF / "ensemble_metrics.csv", index=False)
print(f"\nSaved: ensemble_metrics.csv")

print("\n" + "=" * 70)
print("FINAL WITHIN-FARM COMPARISON")
print("=" * 70)
print(f"  {'Element':<8} {'Ridge':>8} {'RF':>8} {'Stack':>12} {'Kriging':>10}  BEST")
print(f"  {'':8} {'ρ_train':>8} {'ρ_train':>8} {'ρ_cv':>12} {'ρ_cv':>10}")
print("  " + "-" * 60)
for tgt in TARGETS:
    r = results[tgt]
    vals = {"Ridge":r["rho_ridge"],"RF":r["rho_rf"],
            "Stack":r["rho_stack_cv"],"Krig":r["rho_krig_cv"]}
    best = max(vals, key=vals.get)
    print(f"  {tgt:<8} {r['rho_ridge']:>+8.3f} {r['rho_rf']:>+8.3f} "
          f"(α={r['alpha']:.2f}) {r['rho_stack_cv']:>+6.3f} {r['rho_krig_cv']:>+10.3f}  ← {best}")

print(f"\nDone. Plots → {OUT_DIR}")
