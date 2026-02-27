"""
Geo-aware pixel approximation of soil chemistry.

Improvements over pixel_ndvi_real.py:
  1. Geo-aware model: chem ~ spectral_index + utm_x + utm_y + utm_x*utm_y
     (Ridge regression with spatial coordinates as features).
     This lets the model capture spatial trends (e.g. salinity gradient
     from West to East) that pure spectral regression misses.
  2. Smooth outside zone: instead of raw scatter, the Approximate area
     is rendered as a smooth interpolated surface (griddata cubic +
     gaussian_filter) — same technique as heatmap_run.py.
  3. Validate zone: real Sentinel-2 pixels inside field polygons,
     outlined in black with "Validate" label.

Run: python approximated/pixel_geo_approx.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.features import geometry_mask
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import contextily as cx
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from pathlib import Path

# ─── Parameters ──────────────────────────────────────────────────
YEAR      = 2023
FARM      = "Агро Парасат"
IMAGE_LABEL = "median-mosaic 2023 May-Sep"

SCALE_M   = 10
TILE_ZOOM = 14
ALPHA     = 0.62          # chemistry layer opacity (slightly higher = richer look)
BLUR_SIGMA = 3            # gaussian blur for outside zone (small = preserve spatial detail)
GRID_N    = 800           # interpolation grid resolution for outside zone (higher = sharper)

# Spectral predictors per element.
# Format: list of (csv_column, pixel_column) tuples.
#   csv_column  — column name in full_dataset.csv (field-level spectral features)
#   pixel_column — column name in pixels_df (real pixel CSV from pixel_ndvi_real.py)
#
# Multiple predictors are stacked into [spec1, spec2, ..., nx, ny, nx*ny].
# Opposite-sign correlations work correctly: Ridge finds β₁<0 for one predictor
# and β₂>0 for another — they are NOT summed naively, they are weighted by the model.
# Combining predictors improves quality when they are NOT highly collinear (|r|<0.85).
#
# Available pixel_columns: ndvi, ndre, gndvi, evi, bsi
# Available csv_columns: s2_{INDEX}_{season} where INDEX ∈ {NDVI,NDRE,GNDVI,EVI,BSI,SAVI}
#                        and season ∈ {spring, summer, late_summer, autumn}
BEST_PREDICTOR = {
    # pH: NDRE_spring ρ=-0.616 (dominant) + BSI_spring ρ=+0.44 (complementary, opposite sign)
    # BSI captures bare soil / salinity which correlates with pH independently of chlorophyll
    "ph":  [("s2_NDRE_spring", "ndre"),   # ρ = -0.616
            ("s2_BSI_spring",  "bsi")],   # ρ = +0.44 (opposite sign → complementary info)

    # K: BSI_spring ρ=-0.478 + NDRE_spring ρ=-0.37 (same-sign, different mechanism)
    "k":   [("s2_BSI_spring",  "bsi"),    # ρ = -0.478
            ("s2_NDRE_spring", "ndre")],  # ρ = -0.37

    # P: weak correlations — use two best non-collinear predictors
    "p":   [("s2_GNDVI_spring", "gndvi"), # ρ = +0.254
            ("s2_BSI_spring",   "bsi")],  # ρ = -0.22 (opposite sign)

    # Humus: EVI_summer ρ=+0.200 + NDRE_spring ρ=-0.18
    "hu":  [("s2_EVI_summer",   "evi"),   # ρ = +0.200
            ("s2_NDRE_spring",  "ndre")], # ρ = -0.18

    # S: GNDVI_autumn ρ=+0.323 + BSI_spring ρ=-0.28 (opposite sign)
    "s":   [("s2_GNDVI_autumn", "gndvi"), # ρ = +0.323
            ("s2_BSI_spring",   "bsi")],  # ρ = -0.28

    # NO3: GNDVI_spring ρ=-0.298 + EVI_summer ρ=+0.21 (opposite sign)
    "no3": [("s2_GNDVI_spring", "gndvi"), # ρ = -0.298
            ("s2_EVI_summer",   "evi")],  # ρ = +0.21 (opposite sign → additive info)
}

CHEM_LABELS = {
    "ph":  "pH",
    "k":   "K, mg/kg",
    "p":   "P, mg/kg",
    "hu":  "Humus, %",
    "s":   "S, mg/kg",
    "no3": "NO3, mg/kg",
}

CHEM_CMAPS = {
    "ph":  "RdYlGn_r",
    "k":   "YlOrRd",
    "p":   "YlGn",
    "hu":  "BrBG",
    "s":   "PuBuGn",
    "no3": "OrRd",
}

BASE       = Path(__file__).parent
DATA_PATH  = BASE.parent / "data" / "features" / "full_dataset.csv"
TIFF_PATH  = BASE / "tiff" / "s2_2023_summer_mosaic_B2B4B8B3B5B11.tif"
PIXELS_CSV = BASE / "tiff" / f"pixels_{FARM.replace(' ', '_')}_{YEAR}_real.csv"
OUT_DIR    = BASE.parent / "math_statistics" / "output" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: load cross-validated metrics from pixel_geo_cv.py output
# If present, ρ_cv replaces in-sample ρ in map footers for scientific honesty
CV_REPORT_CSV = OUT_DIR / "cv_oof_predictions.csv"
_cv_rho = {}   # col -> (rho_cv, ci_lo, ci_hi) if available
if CV_REPORT_CSV.exists():
    try:
        _oof = pd.read_csv(CV_REPORT_CSV)
        for _col in _oof["element"].unique():
            _sub = _oof[_oof["element"] == _col]
            from scipy.stats import spearmanr as _sr
            _r, _ = _sr(_sub["y_true"].values, _sub["y_pred"].values)
            _boot = []
            _rng  = np.random.default_rng(42)
            _n    = len(_sub)
            for _ in range(500):
                _idx = _rng.choice(_n, _n, replace=True)
                _rb, _ = _sr(_sub["y_true"].values[_idx], _sub["y_pred"].values[_idx])
                _boot.append(_rb)
            _boot = np.array(_boot)
            _cv_rho[_col] = (_r, float(np.percentile(_boot, 2.5)),
                             float(np.percentile(_boot, 97.5)))
        print(f"  CV metrics loaded: {list(_cv_rho.keys())}")
    except Exception as _e:
        print(f"  CV report not loaded: {_e}")
else:
    print(f"  No CV report found at {CV_REPORT_CSV.name} — run pixel_geo_cv.py first")

WGS84    = "EPSG:4326"
TIFF_CRS = "EPSG:32641"   # UTM Zone 41N
MERC     = "EPSG:3857"

t_tiff_wgs = Transformer.from_crs(TIFF_CRS, WGS84, always_xy=True)
t_wgs_merc = Transformer.from_crs(WGS84,    MERC,  always_xy=True)

def to_merc(lo, la):
    return t_wgs_merc.transform(np.asarray(lo), np.asarray(la))

# ─── 1. Load data ────────────────────────────────────────────────
print(f"Loading data ...")
df_full  = pd.read_csv(DATA_PATH)
farm_df  = df_full[(df_full["year"] == YEAR) & (df_full["farm"] == FARM)].copy()

# Collect all unique CSV columns across all elements (BEST_PREDICTOR is now a list of tuples)
pred_csv_cols = list({csv_col for preds in BEST_PREDICTOR.values() for csv_col, _ in preds})
chem_cols     = list(BEST_PREDICTOR.keys())

agg = (
    farm_df.groupby("field_name")
    .agg(
        geometry_wkt=("geometry_wkt", "first"),
        **{c: (c, "mean") for c in chem_cols + pred_csv_cols},
    )
    .reset_index()
)
print(f"  {len(agg)} fields loaded")

gdf = gpd.GeoDataFrame(
    agg,
    geometry=agg["geometry_wkt"].apply(shapely_wkt.loads),
    crs=WGS84,
)

# ─── 2. Check TIFF + pixel CSV ───────────────────────────────────
if not TIFF_PATH.exists():
    print(f"ERROR: TIFF not found at {TIFF_PATH}")
    print("  Run pixel_ndvi_real.py first to download the GeoTIFF.")
    sys.exit(1)

with rasterio.open(TIFF_PATH) as src:
    tiff_crs_actual = src.crs.to_string()
    tiff_transform  = src.transform
    tiff_shape      = (src.height, src.width)
    t_tiff_wgs = Transformer.from_crs(tiff_crs_actual, WGS84, always_xy=True)

    b2_full  = src.read(1).astype(float)
    b4_full  = src.read(2).astype(float)
    b8_full  = src.read(3).astype(float)
    b3_full  = src.read(4).astype(float)
    b5_full  = src.read(5).astype(float)
    b11_full = src.read(6).astype(float)

    print(f"TIFF: {src.height}x{src.width} px  CRS={src.crs.to_epsg()}")

if PIXELS_CSV.exists():
    print(f"Loading pixel CSV: {PIXELS_CSV.name} ...")
    pixels_df = pd.read_csv(PIXELS_CSV)
    print(f"  {len(pixels_df):,} real pixels inside fields")
else:
    print(f"ERROR: Pixel CSV not found at {PIXELS_CSV}")
    print("  Run pixel_ndvi_real.py first.")
    sys.exit(1)

# ─── 3. Geo-aware model: chem ~ spectral + utm_x + utm_y ─────────
print("\nTraining GEO-AWARE regression models ...")
print("  Features: [spectral_index, utm_x, utm_y, utm_x*utm_y] (Ridge)")

# Field-level training data: need centroid UTM coords from full dataset
# We use the grid-point centroids from farm_df (each row = one sampling point)
geo_models = {}

# Build training set from full dataset: spectral + spatial coords
# farm_df has centroid_lon/centroid_lat per grid point
# Convert to UTM for spatial features
t_wgs_utm = Transformer.from_crs(WGS84, TIFF_CRS, always_xy=True)

farm_train = df_full[df_full["farm"] == FARM].copy()
lons_tr = farm_train["centroid_lon"].values
lats_tr = farm_train["centroid_lat"].values
utm_x_tr, utm_y_tr = t_wgs_utm.transform(lons_tr, lats_tr)

# Normalize spatial coords to [0,1] range for numerical stability
utm_x_min, utm_x_max = utm_x_tr.min(), utm_x_tr.max()
utm_y_min, utm_y_max = utm_y_tr.min(), utm_y_tr.max()

def norm_utm(ux, uy):
    """Normalize UTM coords to [0,1] using training range."""
    return (
        (ux - utm_x_min) / (utm_x_max - utm_x_min + 1e-9),
        (uy - utm_y_min) / (utm_y_max - utm_y_min + 1e-9),
    )

nx_tr, ny_tr = norm_utm(utm_x_tr, utm_y_tr)

for col, preds in BEST_PREDICTOR.items():
    # preds is a list of (csv_col, px_col) tuples — may be 1 or more predictors
    csv_cols = [p[0] for p in preds]
    px_cols  = [p[1] for p in preds]

    needed   = csv_cols + [col, "centroid_lon", "centroid_lat"]
    sub = farm_train[needed].dropna()
    if len(sub) < 30:
        print(f"  Skip {col}: n={len(sub)}")
        continue

    lons_s = sub["centroid_lon"].values
    lats_s = sub["centroid_lat"].values
    ux_s, uy_s = t_wgs_utm.transform(lons_s, lats_s)
    nx_s, ny_s = norm_utm(ux_s, uy_s)

    spec_cols = [sub[c].values for c in csv_cols]

    # Feature matrix: [spec1, spec2, ..., x_norm, y_norm, x*y interaction]
    X = np.column_stack(spec_cols + [nx_s, ny_s, nx_s * ny_s])
    y = sub[col].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=10.0)),
    ])
    pipe.fit(X, y)

    y_pred_tr = pipe.predict(X)
    # For logging: report Spearman of first (dominant) predictor alone vs full model
    r_spec1, _ = spearmanr(spec_cols[0], y)
    r_full,  _ = spearmanr(y_pred_tr, y)
    preds_label = "+".join(f"{c.split('_')[2]}_{c.split('_')[3]}" for c in csv_cols)
    print(f"  {col:4s}: Spearman [{preds_label}] 1st-only={r_spec1:+.3f}  "
          f"multi-geo-model={r_full:+.3f}  n={len(sub)}")
    # Store px_cols list alongside pipe so prediction can use multiple pixel columns
    geo_models[col] = (pipe, px_cols, r_full)

# ─── 4. Approximate chemistry for INSIDE pixels (geo-aware) ──────
print("\nApproximating chemistry for INSIDE pixels (geo-aware) ...")

# Get UTM coords of real pixels
ux_px, uy_px = t_wgs_utm.transform(pixels_df["lon"].values, pixels_df["lat"].values)
nx_px, ny_px = norm_utm(ux_px, uy_px)

for col, (pipe, px_cols, r_full) in geo_models.items():
    spec_cols_px = [pixels_df[pc].values for pc in px_cols]
    X_px = np.column_stack(spec_cols_px + [nx_px, ny_px, nx_px * ny_px])
    y_hat = pipe.predict(X_px)
    pixels_df[f"geo_{col}"] = y_hat
    print(f"  {col:4s}: mean={y_hat.mean():.3f}  std={y_hat.std():.3f}")

mx_in, my_in = to_merc(pixels_df["lon"].values, pixels_df["lat"].values)
pixels_df["mx"] = mx_in
pixels_df["my"] = my_in

total_px  = len(pixels_df)
xmin, xmax = mx_in.min() - 80, mx_in.max() + 80
ymin, ymax = my_in.min() - 80, my_in.max() + 80
farm_slug = FARM.replace(" ", "_")

# ─── 5. Extract OUTSIDE pixels + geo-aware approximation ─────────
print("\nExtracting OUTSIDE pixels (full bbox minus field polygons) ...")

gdf_proj = gdf.to_crs(tiff_crs_actual)
field_geoms = list(gdf_proj.geometry)

outside_mask = geometry_mask(
    field_geoms,
    transform=tiff_transform,
    invert=True,    # True = pixels outside all polygons
    out_shape=tiff_shape,
)

valid_outside = outside_mask & (b4_full > 0) & (b8_full > 0)
ro, co = np.where(valid_outside)
print(f"  {valid_outside.sum():,} valid pixels outside fields")

# Subsample for performance
MAX_OUT = 300_000
if len(ro) > MAX_OUT:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(ro), MAX_OUT, replace=False)
    ro, co = ro[idx], co[idx]
    print(f"  Subsampled to {MAX_OUT:,} pixels")

px_x_o, px_y_o = rasterio.transform.xy(tiff_transform, ro, co, offset="center")
px_x_o = np.array(px_x_o, dtype=float)
px_y_o = np.array(px_y_o, dtype=float)
lon_o, lat_o = t_tiff_wgs.transform(px_x_o, px_y_o)

eps = 1e-9
b2r  = b2_full[ro, co]  / 10000.0
b4r  = b4_full[ro, co]  / 10000.0
b8r  = b8_full[ro, co]  / 10000.0
b3r  = b3_full[ro, co]  / 10000.0
b5r  = b5_full[ro, co]  / 10000.0
b11r = b11_full[ro, co] / 10000.0

ndvi_o  = np.clip((b8r - b4r)  / (b8r + b4r  + eps), -1.0,  1.0)
ndre_o  = np.clip((b8r - b5r)  / (b8r + b5r  + eps), -1.0,  1.0)
gndvi_o = np.clip((b8r - b3r)  / (b8r + b3r  + eps), -1.0,  1.0)
# EVI: standard formula uses Blue (B2) as C2 band
evi_o   = np.clip(2.5*(b8r-b4r) / (b8r + 6*b4r - 7.5*b2r + 1 + eps), -2.0, 2.0)
# BSI standard formula: ((SWIR1+Red)-(NIR+Blue)) / ((SWIR1+Red)+(NIR+Blue))
bsi_o   = np.clip(((b11r+b4r)-(b8r+b2r)) / ((b11r+b4r)+(b8r+b2r)+eps), -1.0, 1.0)

# Spatial features for outside pixels (same normalization as training)
nx_o, ny_o = norm_utm(px_x_o, px_y_o)

outside_df = pd.DataFrame({
    "lon": lon_o, "lat": lat_o,
    "utm_x": px_x_o, "utm_y": px_y_o,
    "ndvi": ndvi_o, "ndre": ndre_o,
    "gndvi": gndvi_o, "evi": evi_o, "bsi": bsi_o,
})

print("  Approximating chemistry for outside pixels (geo-aware) ...")
for col, (pipe, px_cols, _) in geo_models.items():
    spec_cols_o = [outside_df[pc].values for pc in px_cols]
    X_o = np.column_stack(spec_cols_o + [nx_o, ny_o, nx_o * ny_o])
    y_hat_o = pipe.predict(X_o)
    outside_df[f"geo_{col}"] = y_hat_o

mx_o, my_o = to_merc(outside_df["lon"].values, outside_df["lat"].values)
outside_df["mx"] = mx_o
outside_df["my"] = my_o
print(f"  Done: {len(outside_df):,} outside pixels approximated")

# ─── 6. Polygon helpers ──────────────────────────────────────────
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
        cen_x = geom.centroid.x
        cen_y = geom.centroid.y
        ax.text(cen_x, cen_y, "Sampling Zone",
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

# ─── 7. Smooth-surface rendering for outside zone ────────────────
#   We use griddata (cubic) + gaussian_filter to create a smooth
#   interpolated image for the Approximate area, then imshow it.
#   This is much smoother than raw scatter and reveals spatial gradients.

# Build interpolation grid in Mercator (same extent as map)
_gx = np.linspace(xmin, xmax, GRID_N)
_gy = np.linspace(ymin, ymax, GRID_N)
grid_x, grid_y = np.meshgrid(_gx, _gy)

def _make_smooth_grid(col_name, df_src):
    """Interpolate col_name from df_src onto the Mercator grid."""
    sub = df_src[["mx", "my", col_name]].dropna()
    if len(sub) < 4:
        return None
    # Subsample for griddata speed (keep up to 80k)
    if len(sub) > 80_000:
        idx = np.random.default_rng(0).choice(len(sub), 80_000, replace=False)
        sub = sub.iloc[idx]
    gz = griddata(
        points=(sub["mx"].values, sub["my"].values),
        values=sub[col_name].values,
        xi=(grid_x, grid_y),
        method="nearest",  # nearest = no inter-cell averaging, preserves field boundaries
    )
    nan_mask = np.isnan(gz)
    if nan_mask.all():
        return None
    gz_filled = np.where(nan_mask, np.nanmedian(gz), gz)
    # Small sigma only to suppress single-pixel noise, not to smooth spatial patterns
    gz_smooth = gaussian_filter(gz_filled, sigma=BLUR_SIGMA)
    gz_smooth[nan_mask] = np.nan
    return gz_smooth

# ─── 8. Render function ──────────────────────────────────────────
def render_map(col, title, cmap_, fname, vmin=None, vmax=None, footer_extra=""):
    in_col  = f"geo_{col}"
    out_col = f"geo_{col}"

    sub_in = pixels_df[["mx", "my", in_col]].dropna()
    if len(sub_in) == 0:
        print(f"  Skip {fname}: no inside data")
        return

    vals_in = sub_in[in_col]
    v0 = vmin if vmin is not None else vals_in.quantile(0.02)
    v1 = vmax if vmax is not None else vals_in.quantile(0.98)

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#111111")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Satellite basemap
    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery,
                   zoom=TILE_ZOOM, alpha=1.0)

    # ── Layer 1: Smooth interpolated surface for Approximate zone ──
    gz = _make_smooth_grid(out_col, outside_df)
    if gz is not None:
        ax.imshow(
            gz,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower", aspect="auto",
            cmap=cmap_, vmin=v0, vmax=v1,
            alpha=ALPHA * 0.75,
            interpolation="nearest",
            zorder=2,
        )
        _draw_approximate_label(ax)

    # ── Layer 2: Real pixels INSIDE field polygons (Validate) ──────
    ax.scatter(sub_in["mx"], sub_in["my"],
               c=vals_in, cmap=cmap_,
               s=0.8, alpha=ALPHA, linewidths=0,
               vmin=v0, vmax=v1,
               zorder=3, rasterized=True)

    # ── Layer 3: Field polygon outlines + Validate labels ──────────
    _draw_validate_polygons(ax)

    # Colorbar
    sm = ScalarMappable(cmap=cmap_, norm=Normalize(v0, v1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.70, pad=0.02, aspect=22)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)
    cb.outline.set_edgecolor("#333333")
    cb.set_label(title, color="white", fontsize=7, labelpad=6)

    ax.set_title(title, fontsize=12, fontweight="bold", color="white", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")

    fig.text(
        0.5, 0.01,
        f"{FARM}  |  {total_px:,} real Sentinel-2 pixels @ 10m  |  {IMAGE_LABEL}"
        f"  |  Ridge(spectral+UTM) · field-level trained, pixel-level applied{footer_extra}",
        ha="center", va="bottom", color="#666666", fontsize=7.5,
    )
    fig.tight_layout(pad=0.4, rect=[0, 0.03, 1, 1])

    out = OUT_DIR / fname
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)
    print(f"  Saved: {out.name}")

# ─── 9. Render NDVI (geo-aware not needed, use raw real pixels) ──
print("\nRendering NDVI heatmap ...")
sub_ndvi = pixels_df[["mx", "my", "ndvi"]].dropna()
if len(sub_ndvi) > 0:
    v0n, v1n = sub_ndvi["ndvi"].quantile(0.02), sub_ndvi["ndvi"].quantile(0.98)
    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#111111")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zoom=TILE_ZOOM, alpha=1.0)

    # Smooth NDVI for outside
    gz_ndvi = _make_smooth_grid("ndvi", outside_df)
    if gz_ndvi is not None:
        ax.imshow(gz_ndvi, extent=[xmin, xmax, ymin, ymax],
                  origin="lower", aspect="auto",
                  cmap="RdYlGn", vmin=v0n, vmax=v1n,
                  alpha=ALPHA * 0.75, interpolation="nearest", zorder=2)
        _draw_approximate_label(ax)

    ax.scatter(sub_ndvi["mx"], sub_ndvi["my"],
               c=sub_ndvi["ndvi"], cmap="RdYlGn",
               s=0.8, alpha=ALPHA, linewidths=0,
               vmin=v0n, vmax=v1n, zorder=3, rasterized=True)
    _draw_validate_polygons(ax)

    sm = ScalarMappable(cmap="RdYlGn", norm=Normalize(v0n, v1n))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.70, pad=0.02, aspect=22)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)
    cb.outline.set_edgecolor("#333333")
    cb.set_label("NDVI", color="white", fontsize=7, labelpad=6)

    ax.set_title("NDVI — real Sentinel-2 pixels @ 10m",
                 fontsize=12, fontweight="bold", color="white", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")
    fig.text(0.5, 0.01,
             f"{FARM}  |  {total_px:,} real S2 pixels @ 10m  |  {IMAGE_LABEL}  |  NDVI (no model applied)",
             ha="center", va="bottom", color="#666666", fontsize=7.5)
    fig.tight_layout(pad=0.4, rect=[0, 0.03, 1, 1])
    out = OUT_DIR / f"geo_approx_{farm_slug}_{YEAR}_NDVI.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)
    print(f"  Saved: {out.name}")

# ─── 10. Chemistry maps ───────────────────────────────────────────
print("\nRendering geo-aware chemistry heatmaps ...")
for col, label in CHEM_LABELS.items():
    if col not in geo_models:
        continue
    _, px_cols, r_full = geo_models[col]
    preds_short = "+".join(pc.upper() for pc in px_cols)
    # Use cross-validated ρ if available, otherwise fall back to in-sample
    if col in _cv_rho:
        rho_cv, ci_lo, ci_hi = _cv_rho[col]
        foot = (f"  |  predictors: {preds_short}"
                f"  |  \u03c1_cv={rho_cv:+.3f} [95%: {ci_lo:+.3f},{ci_hi:+.3f}]"
                f"  |  \u03c1_train={r_full:+.3f} (in-sample)")
    else:
        foot = (f"  |  predictors: {preds_short}"
                f"  |  \u03c1_train={r_full:+.3f} (in-sample, run pixel_geo_cv.py for \u03c1_cv)")
    render_map(
        col, label,
        CHEM_CMAPS[col],
        f"geo_approx_{farm_slug}_{YEAR}_{col}.png",
        footer_extra=foot,
    )

# ─── 11. Summary figure (1x7) ─────────────────────────────────────
print("\nRendering summary figure (1x7) ...")
cols_all   = ["ndvi"] + [f"geo_{c}" for c in CHEM_LABELS]
labels_all = ["NDVI (real S2)"] + list(CHEM_LABELS.values())
cmaps_all  = ["RdYlGn"] + [CHEM_CMAPS[c] for c in CHEM_LABELS]
# ndvi column is in pixels_df directly; outside ndvi in outside_df
src_col_in  = ["ndvi"] + [f"geo_{c}" for c in CHEM_LABELS]
src_col_out = ["ndvi"] + [f"geo_{c}" for c in CHEM_LABELS]

fig, axes = plt.subplots(1, 7, figsize=(42, 7))
fig.patch.set_facecolor("#0a0a0a")

for i, (vc_in, vc_out, lb, cm) in enumerate(
        zip(src_col_in, src_col_out, labels_all, cmaps_all)):
    ax = axes[i]
    ax.set_facecolor("#111111")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zoom=TILE_ZOOM, alpha=1.0)

    col_in_df = "ndvi" if vc_in == "ndvi" else vc_in
    sub_in = pixels_df[["mx", "my", col_in_df]].dropna()
    if len(sub_in) == 0:
        ax.set_title(lb, color="#555", fontsize=8)
        continue

    vals_in = sub_in[col_in_df]
    v0, v1 = vals_in.quantile(0.02), vals_in.quantile(0.98)

    # Smooth outside surface
    out_src_col = "ndvi" if vc_out == "ndvi" else vc_out
    if out_src_col in outside_df.columns:
        gz = _make_smooth_grid(out_src_col, outside_df)
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

    ax.set_title(lb, fontsize=8, fontweight="bold", color="white", pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")

_cv_note = "ρ_cv (LOFO-CV)" if _cv_rho else "ρ values are in-sample (run pixel_geo_cv.py for cross-validated ρ)"
fig.suptitle(
    f"Geo-aware soil chemistry — {FARM}, {YEAR}  |  "
    f"{total_px:,} real S2 pixels @ 10m  |  Ridge(spectral+UTM) · field-level trained, pixel-level applied  |  {_cv_note}",
    fontsize=10, color="white", y=1.008,
)
fig.tight_layout(pad=0.5)
out = OUT_DIR / f"geo_approx_{farm_slug}_{YEAR}_summary.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig)
print(f"  Saved: {out.name}")

print(f"\nDone! Output: {OUT_DIR}")
print(f"Files: geo_approx_{farm_slug}_{YEAR}_*.png")
