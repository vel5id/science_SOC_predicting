"""
pixel_p_enriched.py — MODIFIED VERSION of pixel_geo_approx.py
==============================================================
Специализирован для P (фосфор, mg/kg) с использованием новых признаков
из enriched_dataset.csv и delta_dataset.csv.

ОТЛИЧИЯ от pixel_geo_approx.py:
  1. Единственный таргет: P (фосфор)
  2. Расширенный набор предикторов:
       - Spectral pixel-level: ndvi, ndre, gndvi, evi, bsi (из pixel CSV)
       - Field-level join: GLCM autumn/summer (top predictors |rho|>0.47)
         glcm_glcm_red_ent_autumn   rho=-0.525  ← лучший по |rho|
         glcm_glcm_red_asm_autumn   rho=+0.513
         glcm_glcm_nir_ent_autumn   rho=-0.512
         glcm_glcm_nir_asm_autumn   rho=+0.506
         glcm_glcm_red_idm_autumn   rho=+0.498
       - Field-level join: Topo (aspect_cos, slope)
         topo_aspect_cos  rho=+0.470
         topo_slope       rho=-0.458
       - Field-level join: Climate
         climate_GS_temp   rho=+0.476
         climate_GS_precip rho=+0.397
       - Delta/Range: range_s2_BSI rho=+0.227
       - Temporal:   ts_s2_MSI_std rho=+0.272
  3. 2-уровневая стратегия признаков:
       a) Pixel-level: вычисляются/берутся для каждого пикселя
       b) Field-level: джойн field_name → пикселям внутри поля присваивается
          среднее GLCM/topo/climate своего поля
  4. Ridge + StandardScaler (alpha=10), те же spatial coords
  5. Выдаёт ТОЛЬКО карты для P: детальную + сравнение моделей (old vs new)
  6. Добавлена панель сравнения: 4 карты рядом:
       - P (original: GNDVI_spring+BSI_spring, только spectral)
       - P (enriched: +GLCM+topo+climate+delta)
       - P (GLCM-only: без spectral, only field-level)
       - NDVI (reference)

ПОМЕТКА: MODIFIED — uses enriched+delta features for P
Запуск: python approximated/pixel_p_enriched.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

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
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path

# ─── Parameters ──────────────────────────────────────────────────
YEAR        = 2023
FARM        = "Агро Парасат"
IMAGE_LABEL = "median-mosaic 2023 May-Sep"
TILE_ZOOM   = 14
ALPHA       = 0.65
BLUR_SIGMA  = 3
GRID_N      = 800
RIDGE_ALPHA = 10.0

WGS84    = "EPSG:4326"
TIFF_CRS = "EPSG:32641"
MERC     = "EPSG:3857"
EPS      = 1e-9

BASE        = Path(__file__).parent
TIFF_PATH   = BASE / "tiff" / "s2_2023_summer_mosaic_B4B8B3B5B11.tif"
PIXELS_CSV  = BASE / "tiff" / f"pixels_{FARM.replace(' ', '_')}_{YEAR}_real.csv"
ENRICH_CSV  = BASE.parent / "data" / "features" / "enriched_dataset.csv"
DELTA_CSV   = BASE.parent / "data" / "features" / "delta_dataset.csv"
OUT_DIR     = BASE.parent / "math_statistics" / "output" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CMAP_P = "YlGn"

# ─── Feature definitions ──────────────────────────────────────────
# [A] Pixel-level spectral (доступны для каждого пикселя из TIFF/pixel CSV)
#     (csv_train_col, pixel_col)
PIXEL_SPECTRAL = [
    ("s2_GNDVI_spring",   "gndvi"),   # rho=+0.254 ← оригинальный лучший spectral
    ("s2_BSI_spring",     "bsi"),     # rho=-0.220 ← оригинальный второй
    ("s2_NDRE_spring",    "ndre"),    # rho=-0.170
    ("s2_EVI_summer",     "evi"),     # rho=+0.155
    ("s2_NDVI_spring",    "ndvi"),    # rho=+0.150
]

# [B] Field-level признаки (один скаляр на поле, join по field_name)
#     Эти признаки нельзя вычислить per-pixel → джойним к каждому пикселю его поля
#     (csv_col_in_enriched, короткое_имя)
FIELD_LEVEL_FEATS = [
    # GLCM autumn — top-5 по |rho| для P
    ("glcm_glcm_red_ent_autumn",  "glcm_red_ent_aut"),   # rho=-0.525 ★ ЛУЧШИЙ
    ("glcm_glcm_red_asm_autumn",  "glcm_red_asm_aut"),   # rho=+0.513
    ("glcm_glcm_nir_ent_autumn",  "glcm_nir_ent_aut"),   # rho=-0.512
    ("glcm_glcm_nir_asm_autumn",  "glcm_nir_asm_aut"),   # rho=+0.506
    ("glcm_glcm_red_idm_autumn",  "glcm_red_idm_aut"),   # rho=+0.498
    # GLCM summer
    ("glcm_glcm_red_ent_summer",  "glcm_red_ent_sum"),   # rho=-0.506
    ("glcm_glcm_red_idm_summer",  "glcm_red_idm_sum"),   # rho=+0.478
    # Topo (topo_slope excluded: constant value = 0 for all points in this farm)
    ("topo_aspect_cos",           "topo_aspect_cos"),     # rho=+0.470
    # Climate
    ("climate_GS_temp",           "climate_gs_temp"),     # rho=+0.476
    ("climate_GS_precip",         "climate_gs_precip"),   # rho=+0.397
    # Temporal (enriched)
    ("ts_s2_MSI_std",             "ts_msi_std"),          # rho=+0.272
    # Spectral PCA (поля-level, нет per-pixel аналога)
    ("spectral_PCA_5_spring",     "pca5_spring"),         # rho=+0.370 ★
]

# Delta features (field-level, из delta_dataset.csv)
DELTA_FEATS = [
    ("range_s2_BSI",              "range_bsi"),           # rho=+0.227
    ("delta_l8_NDVI_summer_to_late_summer", "dl8ndvi_sum2lsm"),  # rho=+0.198
]

print("=" * 65)
print("pixel_p_enriched.py  [MODIFIED — enriched+delta features for P]")
print("=" * 65)

# ─── 1. Load enriched + delta data ───────────────────────────────
print("\nLoading enriched_dataset.csv ...")
df_enr = pd.read_csv(ENRICH_CSV)
print(f"  Shape: {df_enr.shape}")

print("Loading delta_dataset.csv (delta columns only) ...")
df_dlt = pd.read_csv(DELTA_CSV)
delta_only = [c for c in df_dlt.columns if c.startswith("delta_") or c.startswith("range_")]
for c in delta_only:
    if c not in df_enr.columns:
        df_enr[c] = df_dlt[c]
print(f"  Added {len(delta_only)} delta/range columns → combined shape: {df_enr.shape}")

farm_train = df_enr[df_enr["farm"] == FARM].copy()
print(f"\n  Farm '{FARM}': {len(farm_train)} grid points, "
      f"{farm_train['field_name'].nunique()} fields, "
      f"years: {sorted(farm_train['year'].unique())}")

# ─── 2. Field-level aggregation for join ─────────────────────────
# Aggегируем field-level признаки по field_name (среднее per field × year)
all_fl_csv_cols = [c for c, _ in FIELD_LEVEL_FEATS + DELTA_FEATS if c in df_enr.columns]
all_spec_csv    = [c for c, _ in PIXEL_SPECTRAL if c in df_enr.columns]

print("\nAggregating field-level features ...")
agg_cols = ["geometry_wkt", "p"] + all_fl_csv_cols + all_spec_csv
agg_cols = list(dict.fromkeys(agg_cols))  # deduplicate

farm_2023 = farm_train[farm_train["year"] == YEAR].copy()
agg = (
    farm_2023.groupby("field_name")
    .agg(geometry_wkt=("geometry_wkt", "first"),
         **{c: (c, "mean") for c in ["p"] + all_fl_csv_cols + all_spec_csv if c in farm_2023.columns})
    .reset_index()
)
print(f"  {len(agg)} fields in {YEAR}")

gdf = gpd.GeoDataFrame(
    agg,
    geometry=agg["geometry_wkt"].apply(shapely_wkt.loads),
    crs=WGS84,
)

# ─── 3. TIFF ─────────────────────────────────────────────────────
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
    print(f"\nTIFF: {src.height}x{src.width} px  CRS={src.crs.to_epsg()}")

# ─── 4. Pixel CSV ────────────────────────────────────────────────
if not PIXELS_CSV.exists():
    print(f"ERROR: Pixel CSV not found: {PIXELS_CSV}")
    sys.exit(1)

pixels_df = pd.read_csv(PIXELS_CSV)
print(f"Loaded {len(pixels_df):,} real pixels  |  fields: {pixels_df['field_name'].nunique()}")

# ─── 5. Join field-level features to pixels ──────────────────────
# Строим словарь field_name → {csv_col: value} из agg
print("\nJoining field-level features to pixels ...")

# Для join берём из farm_train (все годы) поля которые матчатся по field_name
# Используем 2023 данные (agg уже сделан по 2023)
fl_lookup = agg.set_index("field_name")[
    [c for c in all_fl_csv_cols if c in agg.columns]
].to_dict(orient="index")

for csv_col, short_name in FIELD_LEVEL_FEATS + DELTA_FEATS:
    if csv_col not in agg.columns:
        print(f"  WARN: {csv_col} not in agg → skip")
        continue
    vals = pixels_df["field_name"].map(
        agg.set_index("field_name")[csv_col].to_dict()
    )
    pixels_df[short_name] = vals.values

# Проверяем покрытие
n_matched = pixels_df[[n for _, n in FIELD_LEVEL_FEATS + DELTA_FEATS
                        if n in pixels_df.columns]].notna().all(axis=1).sum()
print(f"  Pixels with all field-level features: {n_matched:,} / {len(pixels_df):,}")

# ─── 6. Coordinate system ────────────────────────────────────────
t_wgs_utm = Transformer.from_crs(WGS84, TIFF_CRS, always_xy=True)
t_wgs_merc = Transformer.from_crs(WGS84, MERC,    always_xy=True)

def to_merc(lo, la):
    return t_wgs_merc.transform(np.asarray(lo), np.asarray(la))

lons_tr = farm_train["centroid_lon"].values
lats_tr = farm_train["centroid_lat"].values
utm_x_tr, utm_y_tr = t_wgs_utm.transform(lons_tr, lats_tr)
utm_x_min, utm_x_max = utm_x_tr.min(), utm_x_tr.max()
utm_y_min, utm_y_max = utm_y_tr.min(), utm_y_tr.max()

def norm_utm(ux, uy):
    return (
        (ux - utm_x_min) / (utm_x_max - utm_x_min + EPS),
        (uy - utm_y_min) / (utm_y_max - utm_y_min + EPS),
    )

# ─── 7. Train models ─────────────────────────────────────────────
print("\nTraining models for P ...")

# ── Pixel-scale training data (field median from pixels_df) ───────
# IMPORTANT: pixels_df has spectral indices in TRUE pixel scale (e.g. evi 0–1).
# enriched_dataset.csv uses different normalization (e.g. evi 0–2).
# To avoid train/test distribution mismatch we build spectral train data
# by aggregating pixels_df to field level (median), then joining p from agg.
PX_COLS = [px for _, px in PIXEL_SPECTRAL]  # ['gndvi','bsi','ndre','evi','ndvi']
PIXEL_COL_MAP = {px: csv for csv, px in PIXEL_SPECTRAL}  # px→csv name

# Aggregate pixel CSV to field level (median spectral + centroid lon/lat)
px_field_agg = (
    pixels_df.groupby("field_name")[PX_COLS + ["lon", "lat"]]
    .median()
    .reset_index()
)
# Join ground-truth P from farm_2023 (agg already has p per field)
px_field_agg = px_field_agg.merge(
    agg[["field_name", "p"]].dropna(),
    on="field_name", how="inner"
)
# Also join field-level features from enriched agg
fl_all_short = [sn for _, sn in FIELD_LEVEL_FEATS + DELTA_FEATS]
for csv_col, short_name in FIELD_LEVEL_FEATS + DELTA_FEATS:
    if csv_col in agg.columns:
        px_field_agg[short_name] = px_field_agg["field_name"].map(
            agg.set_index("field_name")[csv_col].to_dict()
        )
print(f"  Pixel-scale field agg: {len(px_field_agg)} fields, "
      f"spectral cols: {PX_COLS}")

ALPHAS = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]


def train_ridge(px_col_names, fl_short_names, label):
    """Train Ridge model with RidgeCV alpha selection + constant-feature removal.

    px_col_names  : list of short pixel column names (e.g. ['gndvi','bsi'])
                    — these come from px_field_agg (pixel-scale values)
    fl_short_names: list of short field-level names (e.g. ['glcm_red_ent_aut'])
                    — these also come from px_field_agg (joined from enriched agg)
    Returns (pipe, rho_train, n_train) or None.
    """
    needed = px_col_names + fl_short_names + ["p", "lon", "lat"]
    needed = [c for c in needed if c in px_field_agg.columns]
    sub = px_field_agg[needed].dropna()
    if len(sub) < 10:
        print(f"  [{label}] Too few samples: n={len(sub)}")
        return None

    lons_s = sub["lon"].values
    lats_s = sub["lat"].values
    ux_s, uy_s = t_wgs_utm.transform(lons_s, lats_s)
    nx_s, ny_s = norm_utm(ux_s, uy_s)

    parts = []
    for c in px_col_names:
        if c in sub.columns:
            parts.append(sub[c].values.reshape(-1, 1))
    for c in fl_short_names:
        if c in sub.columns:
            parts.append(sub[c].values.reshape(-1, 1))
    parts += [nx_s.reshape(-1, 1), ny_s.reshape(-1, 1), (nx_s * ny_s).reshape(-1, 1)]
    X_raw = np.hstack(parts)
    y = sub["p"].values

    # ── Remove constant columns (std < 1e-6) ──────────────────────
    col_stds = np.nanstd(X_raw, axis=0)
    keep_mask = col_stds > 1e-6
    n_dropped = int((~keep_mask).sum())
    if n_dropped > 0:
        print(f"  [{label}] Dropped {n_dropped} constant feature(s)")
    X = X_raw[:, keep_mask]
    if X.shape[1] == 0:
        print(f"  [{label}] No non-constant features left — skipping")
        return None

    # ── RidgeCV for optimal alpha ──────────────────────────────────
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("ridge",   RidgeCV(alphas=ALPHAS, cv=5, scoring="r2")),
    ])
    pipe.fit(X, y)
    best_alpha = pipe.named_steps["ridge"].alpha_
    y_pred = pipe.predict(X)
    r_full, _ = spearmanr(y_pred, y)
    print(f"  [{label}] n={len(sub)}  n_features={X.shape[1]}  "
          f"alpha={best_alpha:.1f}  rho_train={r_full:+.3f}")

    pipe._keep_mask = keep_mask
    return pipe, r_full, len(sub)


# MODEL A — Original (spectral only: gndvi + bsi, pixel-scale)
result_A = train_ridge(["gndvi", "bsi"], [], "Original: GNDVI+BSI")

# MODEL B — Enriched spectral (все pixel-level spectral, pixel-scale)
result_B = train_ridge(PX_COLS, [], "Enriched spectral only")

# MODEL C — GLCM+topo+climate (field-level only, no pixel spectral)
# fl_short_names: short names available in px_field_agg (joined from agg)
fl_short = [sn for _, sn in FIELD_LEVEL_FEATS + DELTA_FEATS
            if sn in px_field_agg.columns]
result_C = train_ridge([], fl_short, "GLCM+Topo+Climate only")

# MODEL D — FULL: pixel spectral + field-level GLCM+topo+climate+delta
result_D = train_ridge(PX_COLS, fl_short, "FULL: spectral+GLCM+topo+climate+delta")

models = {
    "original":  (result_A, ["gndvi", "bsi"],  []),
    "spectral":  (result_B, PX_COLS,           []),
    "field":     (result_C, [],                fl_short),
    "full":      (result_D, PX_COLS,           fl_short),
}

# ─── 8. Pixel CSV coords + Mercator ──────────────────────────────
ux_px, uy_px = t_wgs_utm.transform(pixels_df["lon"].values, pixels_df["lat"].values)
nx_px, ny_px = norm_utm(ux_px, uy_px)
mx_in, my_in = to_merc(pixels_df["lon"].values, pixels_df["lat"].values)
pixels_df["mx"] = mx_in
pixels_df["my"] = my_in
total_px  = len(pixels_df)
xmin, xmax = mx_in.min() - 80, mx_in.max() + 80
ymin, ymax = my_in.min() - 80, my_in.max() + 80
farm_slug = FARM.replace(" ", "_")

# ─── 9. Predict inside pixels ────────────────────────────────────
print("\nPredicting P for inside pixels ...")

def _build_pred_X(df_src, px_col_names, fl_short_names, nx_arr, ny_arr):
    """Build raw feature matrix for prediction (same column order as training).

    px_col_names  : short pixel column names (e.g. ['gndvi','bsi'])
                    — must exist directly in df_src
    fl_short_names: short field-level column names (e.g. ['glcm_red_ent_aut'])
                    — joined to df_src earlier via field_name
    """
    parts = []
    for c in px_col_names:
        if c in df_src.columns:
            parts.append(df_src[c].values.reshape(-1, 1))
        else:
            parts.append(np.zeros((len(df_src), 1)))
    for c in fl_short_names:
        if c in df_src.columns:
            parts.append(df_src[c].values.reshape(-1, 1))
        else:
            parts.append(np.zeros((len(df_src), 1)))
    parts.append(nx_arr.reshape(-1, 1))
    parts.append(ny_arr.reshape(-1, 1))
    parts.append((nx_arr * ny_arr).reshape(-1, 1))
    return np.hstack(parts) if parts else None


def predict_inside(model_result, px_col_names, fl_short_names, col_out):
    if model_result is None:
        pixels_df[col_out] = np.nan
        return
    pipe, r_full, n = model_result

    X_raw = _build_pred_X(pixels_df, px_col_names, fl_short_names, nx_px, ny_px)
    if X_raw is None:
        pixels_df[col_out] = np.nan
        return

    keep_mask = getattr(pipe, "_keep_mask", None)
    X_px = X_raw[:, keep_mask] if keep_mask is not None else X_raw

    y_hat = pipe.predict(X_px)
    pixels_df[col_out] = y_hat
    print(f"  {col_out}: mean={y_hat.mean():.2f}  std={y_hat.std():.2f}")

predict_inside(result_A, ["gndvi", "bsi"], [],       "p_original")
predict_inside(result_B, PX_COLS,          [],       "p_spectral")
predict_inside(result_C, [],               fl_short, "p_field")
predict_inside(result_D, PX_COLS,          fl_short, "p_full")

# ─── 10. Outside pixels ──────────────────────────────────────────
print("\nExtracting outside pixels ...")

gdf_proj = gdf.to_crs(tiff_crs_actual)
field_geoms = list(gdf_proj.geometry)
outside_mask = geometry_mask(field_geoms, transform=tiff_transform,
                             invert=False, out_shape=tiff_shape)
valid_outside = outside_mask & (b4_full > 0) & (b8_full > 0)
ro, co = np.where(valid_outside)
print(f"  {valid_outside.sum():,} valid outside pixels")

MAX_OUT = 300_000
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

ndvi_o  = np.clip((b8r - b4r)  / (b8r + b4r  + EPS), -1.0, 1.0)
ndre_o  = np.clip((b8r - b5r)  / (b8r + b5r  + EPS), -1.0, 1.0)
gndvi_o = np.clip((b8r - b3r)  / (b8r + b3r  + EPS), -1.0, 1.0)
evi_o   = np.clip(2.5*(b8r-b4r) / (b8r + 6*b4r - 7.5*b3r + 1 + EPS), -2.0, 2.0)
bsi_o   = np.clip(((b11r+b4r)-(b8r+b3r)) / ((b11r+b4r)+(b8r+b3r)+EPS), -1.0, 1.0)

nx_o, ny_o = norm_utm(px_x_o, px_y_o)

# Для outside пикселей поля-level признаки неизвестны → заполняем медианой обучения
outside_df = pd.DataFrame({
    "lon": lon_o, "lat": lat_o,
    "utm_x": px_x_o, "utm_y": px_y_o,
    "ndvi": ndvi_o, "ndre": ndre_o,
    "gndvi": gndvi_o, "evi": evi_o, "bsi": bsi_o,
})

# Добавляем field-level cols как медиану farm_train (best approximation for outside)
for csv_col, short_name in FIELD_LEVEL_FEATS + DELTA_FEATS:
    if csv_col in farm_train.columns:
        med = farm_train[csv_col].median()
        outside_df[short_name] = med
    else:
        outside_df[short_name] = 0.0

mx_o, my_o = to_merc(outside_df["lon"].values, outside_df["lat"].values)
outside_df["mx"] = mx_o
outside_df["my"] = my_o

print("Predicting P for outside pixels ...")

def predict_outside(model_result, px_col_names, fl_short_names, col_out):
    if model_result is None:
        outside_df[col_out] = np.nan
        return
    pipe, *_ = model_result

    X_raw = _build_pred_X(outside_df, px_col_names, fl_short_names, nx_o, ny_o)
    if X_raw is None:
        outside_df[col_out] = np.nan
        return

    keep_mask = getattr(pipe, "_keep_mask", None)
    X_o = X_raw[:, keep_mask] if keep_mask is not None else X_raw

    outside_df[col_out] = pipe.predict(X_o)

predict_outside(result_A, ["gndvi", "bsi"], [],       "p_original")
predict_outside(result_B, PX_COLS,          [],       "p_spectral")
predict_outside(result_C, [],               fl_short, "p_field")
predict_outside(result_D, PX_COLS,          fl_short, "p_full")

# ─── 11. Map helpers ─────────────────────────────────────────────
gdf_merc = gdf.to_crs(MERC)

def _draw_field_borders(ax, fontsize=5.5):
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

def _draw_approx_label(ax, text="Approximate\n(model extrapolation)", fontsize=8):
    lx = xmin + (xmax - xmin) * 0.03
    ly = ymin + (ymax - ymin) * 0.03
    ax.text(lx, ly, text, ha="left", va="bottom",
            fontsize=fontsize, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111",
                      edgecolor="#999999", linewidth=1.0, alpha=0.88),
            zorder=9)

_gx = np.linspace(xmin, xmax, GRID_N)
_gy = np.linspace(ymin, ymax, GRID_N)
grid_x, grid_y = np.meshgrid(_gx, _gy)

def _make_grid(col_name, df_src):
    sub = df_src[["mx", "my", col_name]].dropna()
    if len(sub) < 4:
        return None
    if len(sub) > 80_000:
        idx = np.random.default_rng(0).choice(len(sub), 80_000, replace=False)
        sub = sub.iloc[idx]
    gz = griddata((sub["mx"].values, sub["my"].values), sub[col_name].values,
                  (grid_x, grid_y), method="nearest")
    nan_mask = np.isnan(gz)
    if nan_mask.all():
        return None
    gz_filled = np.where(nan_mask, np.nanmedian(gz), gz)
    gz_smooth = gaussian_filter(gz_filled, sigma=BLUR_SIGMA)
    gz_smooth[nan_mask] = np.nan
    return gz_smooth

def _render_panel(ax, col_in, col_out, title, subtitle="", r_val=None, basemap=True):
    """Render one map panel on ax."""
    sub_in = pixels_df[["mx", "my", col_in]].dropna()
    if len(sub_in) == 0:
        ax.set_title(f"{title}\n(no data)", color="white", fontsize=9)
        return

    vals_in = sub_in[col_in]
    v0 = vals_in.quantile(0.02)
    v1 = vals_in.quantile(0.98)
    # Share color scale across panels for comparability
    return v0, v1


# ─── 12. Compute global vmin/vmax for P (shared across all models) ─
# Use full model predictions as reference
p_vals_all = pd.concat([
    pixels_df["p_original"].dropna(),
    pixels_df["p_full"].dropna(),
])
VMIN_P = p_vals_all.quantile(0.02)
VMAX_P = p_vals_all.quantile(0.98)
print(f"\nShared P color scale: [{VMIN_P:.1f}, {VMAX_P:.1f}] mg/kg")

# ─── 13. Main figure: 4-panel comparison ─────────────────────────
print("\nRendering 4-panel comparison figure ...")

panel_defs = [
    # (in_col,       out_col,        title,                         model_result, n_feats_desc)
    ("p_original", "p_original",
     "P — Original model\n(GNDVI+BSI+UTM)",
     result_A, "spectral only\n(pixel-level)"),

    ("p_spectral", "p_spectral",
     "P — All spectral\n(GNDVI+BSI+NDRE+EVI+NDVI+UTM)",
     result_B, "5 spectral indices\n(pixel-level)"),

    ("p_field",   "p_field",
     "P — GLCM+Topo+Climate\n(field-level only)",
     result_C, "GLCM+topo+climate\n(field-level join)"),

    ("p_full",    "p_full",
     "P — FULL model\n(spectral + GLCM + topo + climate + delta)",
     result_D, "spectral + GLCM\n+ topo + climate\n+ delta"),
]

fig = plt.figure(figsize=(40, 11), facecolor="#0a0a0a")
gs_main = gridspec.GridSpec(1, 4, figure=fig, left=0.02, right=0.98,
                            top=0.88, bottom=0.06, wspace=0.04)

for panel_i, (in_col, out_col, title, model_result, feats_desc) in enumerate(panel_defs):
    ax = fig.add_subplot(gs_main[0, panel_i])
    ax.set_facecolor("#111111")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery,
                   zoom=TILE_ZOOM, alpha=1.0)

    # Outside smooth surface
    gz = _make_grid(out_col, outside_df)
    if gz is not None:
        ax.imshow(gz, extent=[xmin, xmax, ymin, ymax],
                  origin="lower", aspect="auto",
                  cmap=CMAP_P, vmin=VMIN_P, vmax=VMAX_P,
                  alpha=ALPHA * 0.75, interpolation="nearest", zorder=2)
        _draw_approx_label(ax, fontsize=7)

    # Inside pixels
    sub_in = pixels_df[["mx", "my", in_col]].dropna()
    if len(sub_in) > 0:
        ax.scatter(sub_in["mx"], sub_in["my"],
                   c=sub_in[in_col], cmap=CMAP_P,
                   s=0.9, alpha=ALPHA, linewidths=0,
                   vmin=VMIN_P, vmax=VMAX_P, zorder=3, rasterized=True)

    _draw_field_borders(ax, fontsize=5)

    # Colorbar
    sm = ScalarMappable(cmap=CMAP_P, norm=Normalize(VMIN_P, VMAX_P))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.68, pad=0.015, aspect=24)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)
    cb.outline.set_edgecolor("#333333")
    cb.set_label("P, mg/kg", color="white", fontsize=7, labelpad=5)

    # rho annotation
    rho_str = ""
    if model_result is not None:
        _, r_val, n_tr = model_result
        rho_str = f"rho_train={r_val:+.3f}  n={n_tr}"

    ax.set_title(title, fontsize=9, fontweight="bold", color="white", pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")

    # Feature label (bottom)
    ax.text(0.5, -0.01, feats_desc, transform=ax.transAxes,
            ha="center", va="top", color="#aaaaaa", fontsize=7,
            style="italic")

    # rho box (top-right)
    if rho_str:
        ax.text(0.98, 0.98, rho_str, transform=ax.transAxes,
                ha="right", va="top", color="white", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                          edgecolor="#555", alpha=0.92), zorder=10)

fig.suptitle(
    f"P (phosphorus, mg/kg) — Model comparison  |  {FARM}, {YEAR}  |  "
    f"{total_px:,} real S2 pixels @ 10m  |  "
    f"[MODIFIED: enriched+delta features]  |  Ridge alpha={RIDGE_ALPHA}  |  "
    f"field-level trained, pixel-level applied",
    fontsize=10, color="white", y=0.97,
)

out_compare = OUT_DIR / f"p_enriched_{farm_slug}_{YEAR}_comparison.png"
fig.savefig(out_compare, dpi=160, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig)
print(f"  Saved: {out_compare.name}")

# ─── 14. Detailed map — FULL model ───────────────────────────────
print("\nRendering detailed P (FULL model) map ...")

fig2, ax2 = plt.subplots(figsize=(12, 10))
fig2.patch.set_facecolor("#0a0a0a")
ax2.set_facecolor("#111111")
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)

cx.add_basemap(ax2, source=cx.providers.Esri.WorldImagery, zoom=TILE_ZOOM, alpha=1.0)

gz_full = _make_grid("p_full", outside_df)
if gz_full is not None:
    ax2.imshow(gz_full, extent=[xmin, xmax, ymin, ymax],
               origin="lower", aspect="auto",
               cmap=CMAP_P, vmin=VMIN_P, vmax=VMAX_P,
               alpha=ALPHA * 0.75, interpolation="nearest", zorder=2)
    _draw_approx_label(ax2, fontsize=9)

sub_full = pixels_df[["mx", "my", "p_full"]].dropna()
ax2.scatter(sub_full["mx"], sub_full["my"],
            c=sub_full["p_full"], cmap=CMAP_P,
            s=1.0, alpha=ALPHA, linewidths=0,
            vmin=VMIN_P, vmax=VMAX_P, zorder=3, rasterized=True)

_draw_field_borders(ax2, fontsize=6)

sm2 = ScalarMappable(cmap=CMAP_P, norm=Normalize(VMIN_P, VMAX_P))
sm2.set_array([])
cb2 = fig2.colorbar(sm2, ax=ax2, shrink=0.72, pad=0.02, aspect=22)
cb2.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)
cb2.outline.set_edgecolor("#333333")
cb2.set_label("P, mg/kg", color="white", fontsize=8, labelpad=6)

rho_str_full = ""
if result_D is not None:
    _, r_D, n_D = result_D
    rho_str_full = f"rho_train = {r_D:+.3f}  (in-sample, n={n_D})"

ax2.set_title("P (phosphorus, mg/kg)  —  FULL enriched model\n"
              "spectral (GNDVI+BSI+NDRE+EVI+NDVI) + GLCM-autumn/summer + topo + climate + delta",
              fontsize=11, fontweight="bold", color="white", pad=8)
ax2.set_xticks([]); ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_edgecolor("#222222")

if rho_str_full:
    ax2.text(0.99, 0.99, rho_str_full, transform=ax2.transAxes,
             ha="right", va="top", color="white", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#1a1a1a",
                       edgecolor="#666", alpha=0.93), zorder=10)

feat_summary = (
    "Predictors:  GNDVI_spring | BSI_spring | NDRE_spring | EVI_summer | NDVI_spring  "
    "(pixel-level)\n"
    "  +  glcm_red_ent_aut (rho=-0.525) | glcm_red_asm_aut | glcm_nir_ent_aut | "
    "topo_aspect_cos | topo_slope | climate_GS_temp | pca5_spring | range_BSI  "
    "(field-level join)"
)
fig2.text(0.5, 0.01, f"{FARM}  |  {total_px:,} real S2 pixels @ 10m  |  {IMAGE_LABEL}\n"
          f"{feat_summary}",
          ha="center", va="bottom", color="#777777", fontsize=6.5)
fig2.tight_layout(pad=0.4, rect=[0, 0.05, 1, 1])

out_detail = OUT_DIR / f"p_enriched_{farm_slug}_{YEAR}_full_detail.png"
fig2.savefig(out_detail, dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig2)
print(f"  Saved: {out_detail.name}")

# ─── 15. Correlation scatter: top predictors for P ───────────────
print("\nRendering top predictor scatter for P ...")

top_pred_pairs = [
    ("glcm_glcm_red_ent_autumn", "GLCM Red Entropy autumn", -0.525),
    ("topo_aspect_cos",          "Topo aspect_cos",         +0.470),
    ("climate_GS_temp",          "Climate GS temperature",  +0.476),
    ("spectral_PCA_5_spring",    "Spectral PCA-5 spring",   +0.370),
    ("s2_GNDVI_spring",          "S2 GNDVI spring",         +0.254),
    ("ts_s2_MSI_std",            "TS S2 MSI std",           +0.272),
]

fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
fig3.patch.set_facecolor("#0a0a0a")
axes3 = axes3.ravel()

# merge farm_train with needed cols
for i, (feat_col, feat_label, rho_ref) in enumerate(top_pred_pairs):
    ax = axes3[i]
    ax.set_facecolor("#111111")

    src_df = farm_train[[feat_col, "p"]].dropna() if feat_col in farm_train.columns else pd.DataFrame()
    if len(src_df) < 5:
        ax.set_title(feat_label, color="gray")
        continue

    x_v = src_df[feat_col].values
    y_v = src_df["p"].values

    # Skip constant features (would produce undefined Spearman / ill-conditioned polyfit)
    if x_v.std() < 1e-9:
        ax.set_title(f"{feat_label}\n(constant — skip)", color="gray", fontsize=8)
        continue

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_act, _ = spearmanr(x_v, y_v)

    sc = ax.scatter(x_v, y_v, c=y_v, cmap=CMAP_P, s=22, alpha=0.70,
                    linewidths=0.2, edgecolors="#333", zorder=3)
    z = np.polyfit(x_v, y_v, 1)
    xr = np.linspace(x_v.min(), x_v.max(), 100)
    ax.plot(xr, np.polyval(z, xr), color="#e05c4a", linewidth=1.8, zorder=4)

    ax.set_xlabel(feat_label, color="white", fontsize=8)
    ax.set_ylabel("P, mg/kg", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.text(0.05, 0.95, f"rho = {r_act:+.3f}\nn = {len(src_df)}",
            transform=ax.transAxes, ha="left", va="top", color="white",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                      edgecolor="#555", alpha=0.9))

    cb3 = fig3.colorbar(sc, ax=ax, pad=0.02, shrink=0.80)
    cb3.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
    cb3.outline.set_edgecolor("#333")

fig3.suptitle(
    f"Top predictors for P (phosphorus)  |  {FARM} all years  |  Spearman rho  "
    f"[MODIFIED: enriched+delta features]",
    color="white", fontsize=11, fontweight="bold", y=1.01,
)
fig3.tight_layout(pad=0.6)
out_scatter = OUT_DIR / f"p_enriched_{farm_slug}_top_predictors.png"
fig3.savefig(out_scatter, dpi=180, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig3)
print(f"  Saved: {out_scatter.name}")

# ─── 16. Final summary ───────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY — P models (rho_train, in-sample)")
print("=" * 65)
model_labels = [
    ("Original (GNDVI+BSI+UTM)",            result_A),
    ("Enriched spectral (5 indices+UTM)",    result_B),
    ("Field-level (GLCM+topo+climate+UTM)",  result_C),
    ("FULL (spectral+GLCM+topo+clim+delta)", result_D),
]
for label, res in model_labels:
    if res is None:
        print(f"  {label:<45} SKIPPED")
    else:
        _, r, n = res
        print(f"  {label:<45} rho_train={r:+.3f}  n={n}")

print()
print("Output files:")
print(f"  {out_compare.name}")
print(f"  {out_detail.name}")
print(f"  {out_scatter.name}")
print()
print("NOTE: rho_train is IN-SAMPLE. For cross-validated rho → run pixel_geo_cv.py")
print("      (or extend it to include field-level features)")
print("NOTE: outside-zone field-level features (GLCM/topo/climate) = farm median")
print("      → outside zone shows spectral+spatial gradient only, not texture variation")
