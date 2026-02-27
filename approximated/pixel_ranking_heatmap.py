"""
pixel_ranking_heatmap.py
========================
Для каждого химического элемента (pH, K, P, Hu, S, NO3):
  1. Считает Spearman rho между всеми признаками и целевым элементом
     (только спектральные / GLCM / topo / climate признаки — без перекрёстных
     химических таргетов).
  2. Строит ranking-таблицу (горизонтальный barplot) — лучший индекс ВВЕРХУ.
  3. Строит попиксельную тепловую карту с использованием лучшего предиктора
     (Ridge + UTM-coords, реальные Sentinel-2 пиксели из pixels_*.csv).

Выходные файлы (math_statistics/output/plots/):
  ranking_<element>.png        — ranking barplot
  heatmap_best_<element>.png   — тепловая карта с лучшим предиктором
  ranking_all_summary.png      — сводный 2×6 барплот всех элементов

Запуск: python approximated/pixel_ranking_heatmap.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import contextily as cx
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pathlib import Path

# ─── Parameters ──────────────────────────────────────────────────
YEAR        = 2023
FARM        = "Агро Парасат"
IMAGE_LABEL = "median-mosaic 2023 May-Sep"
TILE_ZOOM   = 14
ALPHA       = 0.65
BLUR_SIGMA  = 3
GRID_N      = 800
TOP_N       = 20      # сколько признаков показывать в ranking барплоте
MIN_N       = 30      # минимум строк для включения признака в ranking

WGS84    = "EPSG:4326"
TIFF_CRS = "EPSG:32641"
MERC     = "EPSG:3857"
EPS      = 1e-9

ALPHAS_CV = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]

BASE        = Path(__file__).parent
ENRICH_CSV  = BASE.parent / "data" / "features" / "enriched_dataset.csv"
DELTA_CSV   = BASE.parent / "data" / "features" / "delta_dataset.csv"
TIFF_PATH   = BASE / "tiff" / "s2_2023_summer_mosaic_B4B8B3B5B11.tif"
PIXELS_CSV  = BASE / "tiff" / f"pixels_{FARM.replace(' ', '_')}_{YEAR}_real.csv"
OUT_DIR     = BASE.parent / "math_statistics" / "output" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["ph", "k", "p", "hu", "s", "no3"]

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

# Признаки, которые НЕ должны попадать в ranking
# (другие химические анализы — они не являются дистанционными признаками)
NON_SPECTRAL = {
    "cu", "mo", "fe", "zn", "mg", "mn", "soc", "b", "ca", "na",
    "id", "year",
}
META_COLS = {"farm", "field_name", "centroid_lon", "centroid_lat",
             "geometry_wkt", "sample_id"}

# Для pixel-level предсказания: маппинг csv-индекса → колонка в pixel CSV
PIXEL_INDEX_MAP = {
    "s2_NDVI_spring":  "ndvi",
    "s2_NDVI_summer":  "ndvi",
    "s2_NDVI_autumn":  "ndvi",
    "s2_NDRE_spring":  "ndre",
    "s2_NDRE_summer":  "ndre",
    "s2_NDRE_autumn":  "ndre",
    "s2_GNDVI_spring": "gndvi",
    "s2_GNDVI_summer": "gndvi",
    "s2_GNDVI_autumn": "gndvi",
    "s2_EVI_spring":   "evi",
    "s2_EVI_summer":   "evi",
    "s2_EVI_autumn":   "evi",
    "s2_BSI_spring":   "bsi",
    "s2_BSI_summer":   "bsi",
    "s2_BSI_autumn":   "bsi",
    # spectral_ duplicates
    "spectral_NDVI_spring":  "ndvi",
    "spectral_NDVI_summer":  "ndvi",
    "spectral_NDRE_spring":  "ndre",
    "spectral_NDRE_summer":  "ndre",
    "spectral_GNDVI_spring": "gndvi",
    "spectral_GNDVI_summer": "gndvi",
    "spectral_GNDVI_autumn": "gndvi",
    "spectral_EVI_spring":   "evi",
    "spectral_EVI_summer":   "evi",
    "spectral_BSI_spring":   "bsi",
    "spectral_BSI_summer":   "bsi",
    "spectral_BSI_autumn":   "bsi",
    # l8 (no per-pixel equivalent — use ndvi as proxy)
    "l8_NDVI_spring":  "ndvi",
    "l8_NDVI_summer":  "ndvi",
    "l8_NDVI_autumn":  "ndvi",
    "l8_GNDVI_spring": "gndvi",
    "l8_GNDVI_summer": "gndvi",
    "l8_SAVI_spring":  "ndvi",
    "l8_SAVI_summer":  "ndvi",
    "l8_SAVI_late_summer": "ndvi",
    # temporal stats — use mean of ndvi as proxy
    "ts_s2_NDVI_mean":   "ndvi",
    "ts_s2_GNDVI_mean":  "gndvi",
    "ts_s2_NDRE_mean":   "ndre",
    "ts_s2_EVI_mean":    "evi",
    "ts_l8_NDVI_mean":   "ndvi",
    "ts_l8_GNDVI_mean":  "gndvi",
    "ts_l8_SAVI_mean":   "ndvi",
    "ts_s2_MSI_std":     "bsi",   # MSI ~ BSI (bare/moisture)
}

print("=" * 70)
print("pixel_ranking_heatmap.py  —  Per-element feature ranking + heatmaps")
print("=" * 70)

# ─── 1. Load data ─────────────────────────────────────────────────
print("\nLoading enriched_dataset.csv ...")
df_enr = pd.read_csv(ENRICH_CSV)
print(f"  Shape: {df_enr.shape}")

print("Loading delta_dataset.csv ...")
df_dlt = pd.read_csv(DELTA_CSV)
delta_only = [c for c in df_dlt.columns
              if (c.startswith("delta_") or c.startswith("range_"))
              and c not in df_enr.columns]
for c in delta_only:
    df_enr[c] = df_dlt[c]
print(f"  Added {len(delta_only)} delta/range cols  →  combined: {df_enr.shape}")

# Feature columns (spectral / topo / climate / GLCM only)
exclude_cols = set(TARGETS) | NON_SPECTRAL | META_COLS
feat_cols = [
    c for c in df_enr.select_dtypes(include=[np.number]).columns
    if c not in exclude_cols
]
print(f"  Usable feature columns: {len(feat_cols)}")

# ─── 2. Compute Spearman ranking per element ──────────────────────
print("\nComputing Spearman rho rankings ...")

rankings = {}   # tgt -> list of (feat, rho) sorted by |rho| desc

for tgt in TARGETS:
    sub = df_enr[[tgt] + feat_cols].dropna(subset=[tgt])
    y = sub[tgt].values
    rho_dict = {}
    for c in feat_cols:
        x = sub[c].values
        mask = ~np.isnan(x)
        if mask.sum() < MIN_N:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, _ = spearmanr(x[mask], y[mask])
        if not np.isnan(r):
            rho_dict[c] = r
    ranked = sorted(rho_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
    rankings[tgt] = ranked
    best_feat, best_r = ranked[0]
    print(f"  {tgt.upper():4s}: best={best_feat}  rho={best_r:+.3f}  "
          f"(total ranked: {len(ranked)})")

# ─── 3. Build farm training data ─────────────────────────────────
farm_df = df_enr[df_enr["farm"] == FARM].copy()
farm_2023 = farm_df[farm_df["year"] == YEAR].copy()
print(f"\nFarm '{FARM}': n={len(farm_df)}  fields_2023={farm_2023['field_name'].nunique()}")

# ─── 4. Load TIFF + Pixel CSV ─────────────────────────────────────
if not TIFF_PATH.exists():
    print(f"ERROR: TIFF not found: {TIFF_PATH}")
    sys.exit(1)
if not PIXELS_CSV.exists():
    print(f"ERROR: Pixel CSV not found: {PIXELS_CSV}")
    sys.exit(1)

with rasterio.open(TIFF_PATH) as src:
    tiff_crs_str  = src.crs.to_string()
    tiff_transform = src.transform
    tiff_shape    = (src.height, src.width)
    b4_full  = src.read(1).astype(float)
    b8_full  = src.read(2).astype(float)
    b3_full  = src.read(3).astype(float)
    b5_full  = src.read(4).astype(float)
    b11_full = src.read(5).astype(float)
    t_tiff_wgs = Transformer.from_crs(tiff_crs_str, WGS84, always_xy=True)
    print(f"\nTIFF: {src.height}x{src.width} px  CRS={src.crs.to_epsg()}")

pixels_df = pd.read_csv(PIXELS_CSV)
print(f"Pixel CSV: {len(pixels_df):,} real pixels  |  "
      f"{pixels_df['field_name'].nunique()} fields")

# ─── 5. Coordinate helpers ────────────────────────────────────────
t_wgs_utm  = Transformer.from_crs(WGS84, TIFF_CRS, always_xy=True)
t_wgs_merc = Transformer.from_crs(WGS84, MERC,    always_xy=True)

def to_merc(lo, la):
    return t_wgs_merc.transform(np.asarray(lo), np.asarray(la))

lons_farm = farm_df["centroid_lon"].values
lats_farm = farm_df["centroid_lat"].values
ux_farm, uy_farm = t_wgs_utm.transform(lons_farm, lats_farm)
UTM_X_MIN, UTM_X_MAX = ux_farm.min(), ux_farm.max()
UTM_Y_MIN, UTM_Y_MAX = uy_farm.min(), uy_farm.max()

def norm_utm(ux, uy):
    return (
        (ux - UTM_X_MIN) / (UTM_X_MAX - UTM_X_MIN + EPS),
        (uy - UTM_Y_MIN) / (UTM_Y_MAX - UTM_Y_MIN + EPS),
    )

# Pixel CSV mercator coords
mx_in, my_in = to_merc(pixels_df["lon"].values, pixels_df["lat"].values)
pixels_df["mx"] = mx_in
pixels_df["my"] = my_in
ux_px, uy_px = t_wgs_utm.transform(pixels_df["lon"].values, pixels_df["lat"].values)
nx_px, ny_px = norm_utm(ux_px, uy_px)

xmin = mx_in.min() - 80
xmax = mx_in.max() + 80
ymin = my_in.min() - 80
ymax = my_in.max() + 80
farm_slug = FARM.replace(" ", "_")
total_px = len(pixels_df)

# ─── 6. Outside pixels (compute once) ────────────────────────────
print("\nExtracting outside-field pixels ...")
agg_geom = (
    farm_2023.groupby("field_name")
    .agg(geometry_wkt=("geometry_wkt", "first"))
    .reset_index()
)
gdf_farm = gpd.GeoDataFrame(
    agg_geom,
    geometry=agg_geom["geometry_wkt"].apply(shapely_wkt.loads),
    crs=WGS84,
).to_crs(tiff_crs_str)

outside_mask = geometry_mask(
    list(gdf_farm.geometry),
    transform=tiff_transform,
    invert=False,
    out_shape=tiff_shape,
)
valid_out = outside_mask & (b4_full > 0) & (b8_full > 0)
ro, co = np.where(valid_out)
print(f"  Valid outside pixels: {valid_out.sum():,}")

MAX_OUT = 250_000
if len(ro) > MAX_OUT:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(ro), MAX_OUT, replace=False)
    ro, co = ro[idx], co[idx]

px_x_o, px_y_o = rasterio.transform.xy(tiff_transform, ro, co, offset="center")
px_x_o = np.array(px_x_o, dtype=float)
px_y_o = np.array(px_y_o, dtype=float)
lon_o, lat_o = t_tiff_wgs.transform(px_x_o, px_y_o)

b4r  = b4_full[ro, co]  / 10000.0
b8r  = b8_full[ro, co]  / 10000.0
b3r  = b3_full[ro, co]  / 10000.0
b5r  = b5_full[ro, co]  / 10000.0
b11r = b11_full[ro, co] / 10000.0

outside_spectral = {
    "ndvi":  np.clip((b8r - b4r)  / (b8r + b4r  + EPS), -1.0, 1.0),
    "ndre":  np.clip((b8r - b5r)  / (b8r + b5r  + EPS), -1.0, 1.0),
    "gndvi": np.clip((b8r - b3r)  / (b8r + b3r  + EPS), -1.0, 1.0),
    "evi":   np.clip(2.5*(b8r-b4r) / (b8r + 6*b4r - 7.5*b3r + 1 + EPS), -2.0, 2.0),
    "bsi":   np.clip(((b11r+b4r)-(b8r+b3r)) / ((b11r+b4r)+(b8r+b3r)+EPS), -1.0, 1.0),
}
nx_o, ny_o = norm_utm(px_x_o, px_y_o)
mx_o, my_o = to_merc(lon_o, lat_o)

# ─── 7. Helper: train + predict best-predictor Ridge model ────────

def _get_pixel_col(feat_name):
    """Map a csv feature name to its pixel-level column (or None if not available)."""
    if feat_name in PIXEL_INDEX_MAP:
        return PIXEL_INDEX_MAP[feat_name]
    # Fallback: try pattern matching
    for key in ("ndvi", "ndre", "gndvi", "evi", "bsi"):
        if key.lower() in feat_name.lower():
            return key
    return None


def train_and_predict(tgt, feat_name, rho_val):
    """Train Ridge(CV) on farm data using best spectral feature.
    Returns (y_hat_inside, y_hat_outside, rho_train, n_train, px_col_used).
    """
    # Try to map feat_name → pixel column
    px_col = _get_pixel_col(feat_name)

    # --- Build training data ---
    needed = [feat_name, tgt, "centroid_lon", "centroid_lat"]
    needed = [c for c in needed if c in farm_df.columns]
    sub = farm_df[needed].dropna()
    if len(sub) < 15:
        return None

    lons_s = sub["centroid_lon"].values
    lats_s = sub["centroid_lat"].values
    ux_s, uy_s = t_wgs_utm.transform(lons_s, lats_s)
    nx_s, ny_s = norm_utm(ux_s, uy_s)

    X_tr = np.column_stack([
        sub[feat_name].values, nx_s, ny_s, nx_s * ny_s
    ])
    y_tr = sub[tgt].values

    # Remove constant cols
    stds = X_tr.std(axis=0)
    keep = stds > 1e-6
    X_tr = X_tr[:, keep]
    if X_tr.shape[1] == 0:
        return None

    pipe = Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(alphas=ALPHAS_CV, cv=5, scoring="r2")),
    ])
    pipe.fit(X_tr, y_tr)
    pipe._keep_mask = keep

    y_pred_tr = pipe.predict(X_tr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho_tr, _ = spearmanr(y_pred_tr, y_tr)

    # --- Predict inside pixels ---
    if px_col and px_col in pixels_df.columns:
        X_in_raw = np.column_stack([
            pixels_df[px_col].values, nx_px, ny_px, nx_px * ny_px
        ])
    else:
        # Field-level feature: use farm median as constant → spatial pattern only
        med_val = farm_df[feat_name].median()
        X_in_raw = np.column_stack([
            np.full(len(pixels_df), med_val), nx_px, ny_px, nx_px * ny_px
        ])
        px_col = "(field-level, farm median)"

    X_in = X_in_raw[:, keep]
    y_hat_in = pipe.predict(X_in)

    # --- Predict outside pixels ---
    if px_col in outside_spectral:
        spec_o = outside_spectral[px_col]
    else:
        spec_o = np.full(len(lon_o), farm_df[feat_name].median())

    X_out_raw = np.column_stack([spec_o, nx_o, ny_o, nx_o * ny_o])
    X_out = X_out_raw[:, keep]
    y_hat_out = pipe.predict(X_out)

    return y_hat_in, y_hat_out, rho_tr, len(sub), px_col


# ─── 8. Smooth grid helper ────────────────────────────────────────
_gx = np.linspace(xmin, xmax, GRID_N)
_gy = np.linspace(ymin, ymax, GRID_N)
grid_x, grid_y = np.meshgrid(_gx, _gy)

def make_smooth_grid(mx_arr, my_arr, z_arr):
    sub = pd.DataFrame({"mx": mx_arr, "my": my_arr, "z": z_arr}).dropna()
    if len(sub) < 4:
        return None
    if len(sub) > 80_000:
        idx = np.random.default_rng(0).choice(len(sub), 80_000, replace=False)
        sub = sub.iloc[idx]
    gz = griddata(
        (sub["mx"].values, sub["my"].values), sub["z"].values,
        (grid_x, grid_y), method="nearest"
    )
    nan_m = np.isnan(gz)
    if nan_m.all():
        return None
    gz = np.where(nan_m, np.nanmedian(gz), gz)
    gz = gaussian_filter(gz, sigma=BLUR_SIGMA)
    gz[nan_m] = np.nan
    return gz

# ─── 9. Draw field borders ────────────────────────────────────────
gdf_merc = gpd.GeoDataFrame(
    agg_geom,
    geometry=agg_geom["geometry_wkt"].apply(shapely_wkt.loads),
    crs=WGS84,
).to_crs(MERC)

def draw_borders(ax):
    for _, row in gdf_merc.iterrows():
        geom = row.geometry
        polys = ([geom] if geom.geom_type == "Polygon"
                 else list(geom.geoms) if geom.geom_type == "MultiPolygon"
                 else [])
        for poly in polys:
            xp, yp = poly.exterior.xy
            ax.plot(xp, yp, color="black", linewidth=2.0, zorder=6)
            ax.plot(xp, yp, color="white", linewidth=0.7,
                    linestyle="--", alpha=0.55, zorder=7)

# ─── 10. Per-element ranking barplot + heatmap ───────────────────
print("\n" + "=" * 70)
print("Rendering per-element ranking barplots + heatmaps ...")
print("=" * 70)

best_summary = {}   # tgt -> (feat_name, rho, px_col, rho_train)

for tgt in TARGETS:
    label = CHEM_LABELS[tgt]
    cmap  = CHEM_CMAPS[tgt]
    ranked = rankings[tgt]

    print(f"\n── {tgt.upper()} ({label}) ──")

    # ── A. Ranking barplot ──────────────────────────────────────
    top = ranked[:TOP_N]   # top-N, best at index 0
    feats   = [f for f, _ in top]
    rhos    = [r for _, r in top]
    abs_rho = [abs(r) for r in rhos]
    colors  = ["#e05c4a" if r < 0 else "#4ab5e0" for r in rhos]

    fig_r, ax_r = plt.subplots(figsize=(10, 8))
    fig_r.patch.set_facecolor("#0d0d0d")
    ax_r.set_facecolor("#111111")

    # Horizontal bars — best at TOP (invert y-axis display)
    y_pos = np.arange(TOP_N)
    bars = ax_r.barh(
        y_pos, abs_rho,
        color=colors, edgecolor="#222", linewidth=0.4,
        height=0.72,
    )

    # Labels on bars
    for i, (feat, rho_v, bar) in enumerate(zip(feats, rhos, bars)):
        sign_str = f"rho={rho_v:+.3f}"
        ax_r.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            sign_str,
            va="center", ha="left", color="white", fontsize=7.5,
        )

    # Feature names on y-axis
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(feats, fontsize=7.5, color="white")
    ax_r.invert_yaxis()   # rank #1 at top

    ax_r.set_xlabel("|Spearman rho|", color="white", fontsize=9)
    ax_r.tick_params(axis="x", colors="white", labelsize=8)
    ax_r.tick_params(axis="y", colors="white")
    for spine in ax_r.spines.values():
        spine.set_edgecolor("#333333")
    ax_r.set_xlim(0, max(abs_rho) + 0.12)
    ax_r.axvline(0.3, color="#555", linewidth=0.8, linestyle=":")
    ax_r.axvline(0.5, color="#888", linewidth=0.8, linestyle=":")
    ax_r.axvline(0.7, color="#aaa", linewidth=1.0, linestyle=":")

    # Legend patch
    from matplotlib.patches import Patch
    legend_elem = [
        Patch(facecolor="#e05c4a", label="inverse (rho < 0)"),
        Patch(facecolor="#4ab5e0", label="direct  (rho > 0)"),
    ]
    ax_r.legend(handles=legend_elem, loc="lower right",
                facecolor="#1a1a1a", edgecolor="#444", labelcolor="white",
                fontsize=8)

    ax_r.set_title(
        f"Feature ranking for  {label}  |  top-{TOP_N} predictors  "
        f"(Spearman rho, n>={MIN_N}, all farms)\n"
        f"Best predictor: {feats[0]}  rho={rhos[0]:+.3f}",
        color="white", fontsize=9, fontweight="bold", pad=8,
    )
    fig_r.tight_layout(pad=0.6)
    out_rank = OUT_DIR / f"ranking_{tgt}.png"
    fig_r.savefig(out_rank, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig_r)
    print(f"  Saved: {out_rank.name}")

    # ── B. Heatmap with best predictor ─────────────────────────
    best_feat, best_rho = ranked[0]
    print(f"  Training model for {tgt} using '{best_feat}' (rho={best_rho:+.3f}) ...")

    result = train_and_predict(tgt, best_feat, best_rho)
    if result is None:
        print(f"  SKIP heatmap: not enough data for '{tgt}'")
        best_summary[tgt] = (best_feat, best_rho, None, None)
        continue

    y_hat_in, y_hat_out, rho_train, n_train, px_col_used = result
    best_summary[tgt] = (best_feat, best_rho, rho_train, px_col_used)
    print(f"  Model: n={n_train}  px_col={px_col_used}  rho_train={rho_train:+.3f}")
    print(f"  Inside:  mean={y_hat_in.mean():.3f}  std={y_hat_in.std():.3f}")

    # Color scale from inside pixels
    valid_in = y_hat_in[~np.isnan(y_hat_in)]
    if len(valid_in) == 0:
        continue
    vmin = np.percentile(valid_in, 2)
    vmax = np.percentile(valid_in, 98)

    fig_h, ax_h = plt.subplots(figsize=(11, 10))
    fig_h.patch.set_facecolor("#0a0a0a")
    ax_h.set_facecolor("#111111")
    ax_h.set_xlim(xmin, xmax)
    ax_h.set_ylim(ymin, ymax)

    cx.add_basemap(ax_h, source=cx.providers.Esri.WorldImagery,
                   zoom=TILE_ZOOM, alpha=1.0)

    # Outside smooth surface
    gz = make_smooth_grid(mx_o, my_o, y_hat_out)
    if gz is not None:
        ax_h.imshow(gz, extent=[xmin, xmax, ymin, ymax],
                    origin="lower", aspect="auto",
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    alpha=ALPHA * 0.75, interpolation="nearest", zorder=2)
        # Approximate label
        ax_h.text(
            xmin + (xmax - xmin) * 0.03,
            ymin + (ymax - ymin) * 0.03,
            "Approximate\n(model extrapolation)",
            ha="left", va="bottom", fontsize=8, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#111",
                      edgecolor="#999", linewidth=1.0, alpha=0.88),
            zorder=9,
        )

    # Inside pixels scatter
    ax_h.scatter(
        pixels_df["mx"], pixels_df["my"],
        c=y_hat_in, cmap=cmap,
        s=0.9, alpha=ALPHA, linewidths=0,
        vmin=vmin, vmax=vmax, zorder=3, rasterized=True,
    )

    draw_borders(ax_h)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin, vmax))
    sm.set_array([])
    cb = fig_h.colorbar(sm, ax=ax_h, shrink=0.72, pad=0.02, aspect=22)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)
    cb.outline.set_edgecolor("#333")
    cb.set_label(label, color="white", fontsize=8, labelpad=6)

    # Info box
    info_txt = (
        f"Best predictor: {best_feat}\n"
        f"Spearman rho (all farms) = {best_rho:+.3f}\n"
        f"rho_train (farm, in-sample) = {rho_train:+.3f}  n={n_train}\n"
        f"pixel proxy: {px_col_used}"
    )
    ax_h.text(
        0.99, 0.99, info_txt,
        transform=ax_h.transAxes, ha="right", va="top",
        color="white", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#1a1a1a",
                  edgecolor="#666", alpha=0.94),
        zorder=10,
    )

    ax_h.set_title(
        f"{label}  —  best-predictor heatmap\n"
        f"Predictor: {best_feat}  (rho={best_rho:+.3f})",
        fontsize=11, fontweight="bold", color="white", pad=8,
    )
    ax_h.set_xticks([]); ax_h.set_yticks([])
    for spine in ax_h.spines.values():
        spine.set_edgecolor("#222")

    fig_h.text(
        0.5, 0.005,
        f"{FARM}  |  {total_px:,} real S2 pixels @ 10m  |  {IMAGE_LABEL}  |  "
        f"Ridge + UTM-coords  |  RidgeCV alpha auto",
        ha="center", va="bottom", color="#666", fontsize=6.5,
    )
    fig_h.tight_layout(pad=0.4, rect=[0, 0.02, 1, 1])

    out_h = OUT_DIR / f"heatmap_best_{tgt}.png"
    fig_h.savefig(out_h, dpi=180, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig_h)
    print(f"  Saved: {out_h.name}")


# ─── 11. Summary: all rankings in one figure ─────────────────────
print("\nRendering summary ranking figure (all elements) ...")

n_tgt = len(TARGETS)
fig_s = plt.figure(figsize=(36, 28), facecolor="#0d0d0d")
gs = gridspec.GridSpec(
    2, 3, figure=fig_s,
    left=0.04, right=0.97, top=0.93, bottom=0.04,
    wspace=0.35, hspace=0.45,
)

for ti, tgt in enumerate(TARGETS):
    row, col_i = divmod(ti, 3)
    ax = fig_s.add_subplot(gs[row, col_i])
    ax.set_facecolor("#111111")

    label = CHEM_LABELS[tgt]
    ranked = rankings[tgt]
    top = ranked[:TOP_N]
    feats   = [f for f, _ in top]
    rhos    = [r for _, r in top]
    abs_rho = [abs(r) for r in rhos]
    colors  = ["#e05c4a" if r < 0 else "#4ab5e0" for r in rhos]

    y_pos = np.arange(len(top))
    ax.barh(y_pos, abs_rho, color=colors, edgecolor="#222",
            linewidth=0.3, height=0.72)

    # rho labels
    for i, (r_v, a_r) in enumerate(zip(rhos, abs_rho)):
        ax.text(a_r + 0.004, i, f"{r_v:+.3f}",
                va="center", ha="left", color="white", fontsize=6.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=6.5, color="white")
    ax.invert_yaxis()

    ax.set_xlabel("|Spearman rho|", color="white", fontsize=7.5)
    ax.tick_params(axis="x", colors="white", labelsize=7)
    ax.tick_params(axis="y", colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.set_xlim(0, max(abs_rho) + 0.13)
    ax.axvline(0.3, color="#555", lw=0.7, ls=":")
    ax.axvline(0.5, color="#888", lw=0.7, ls=":")

    # Highlight best bar
    ax.get_yticklabels()[0].set_color("#FFD700")
    ax.get_yticklabels()[0].set_fontweight("bold")

    best_f, best_r = ranked[0]
    info = best_summary.get(tgt, (best_f, best_r, None, None))
    rho_tr = info[2]
    rho_tr_str = f"  |  rho_train={rho_tr:+.3f}" if rho_tr is not None else ""
    ax.set_title(
        f"{label}  (best: rho={best_r:+.3f}{rho_tr_str})",
        color="#FFD700", fontsize=8.5, fontweight="bold", pad=5,
    )

fig_s.suptitle(
    f"Feature ranking for all soil elements  |  {FARM}, {YEAR}  |  "
    f"Spearman |rho|, all farms+years, top-{TOP_N} predictors\n"
    f"Gold = best predictor  |  Red = inverse corr  |  Blue = direct corr  |  "
    f"Vertical lines: 0.3 / 0.5 / 0.7",
    color="white", fontsize=11, fontweight="bold", y=0.975,
)

out_sum = OUT_DIR / "ranking_all_summary.png"
fig_s.savefig(out_sum, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
plt.close(fig_s)
print(f"  Saved: {out_sum.name}")

# ─── 12. Console summary ─────────────────────────────────────────
print()
print("=" * 70)
print("RANKING SUMMARY")
print("=" * 70)
print(f"{'Element':<8}  {'Best predictor':<48}  {'rho(all)':<10}  {'rho_train'}")
print("-" * 90)
for tgt in TARGETS:
    best_f, best_r, rho_tr, px_used = best_summary.get(tgt, ("-", 0, None, None))
    rho_tr_s = f"{rho_tr:+.3f}" if rho_tr is not None else "n/a"
    print(f"  {tgt.upper():<6}  {best_f:<48}  {best_r:+.3f}      {rho_tr_s}")

print()
print("Output files saved to:", OUT_DIR)
print(f"  ranking_<element>.png          x {len(TARGETS)} files")
print(f"  heatmap_best_<element>.png     x {len(TARGETS)} files")
print(f"  ranking_all_summary.png        (2x3 grid)")
print()
print("NOTE: rho (all farms) = Spearman across full dataset (n>=30 rows)")
print("      rho_train       = in-sample rho for Ridge model on farm only")
