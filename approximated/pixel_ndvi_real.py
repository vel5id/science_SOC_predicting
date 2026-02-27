"""
Реальная попиксельная NDVI карта через GEE + локальная обработка GeoTIFF.

Процесс:
  1. GEE: строим median mosaic из всех чистых снимков Sentinel-2 за сезон
     (один снимок не покрывает весь bbox — mosaic решает это)
  2. Скачиваем GeoTIFF (B2, B4, B8, B3, B5, B11) через getDownloadURL -> approximated/tiff/
  3. Открываем TIFF локально через rasterio
  4. Для каждого поля хозяйства маскируем растр по WKT-полигону
  5. На каждом пикселе вычисляем: NDVI, NDRE, GNDVI, EVI, BSI
  6. Присваиваем: pixel_id, field_name, lon, lat, utm_x, utm_y + индексы
  7. Обучаем HuberRegressor: chem ~ spectral_index (на полном датасете 1215 точек)
  8. Аппроксимируем pH, K, P, Hu, S, NO3 для каждого пикселя
  9. Строим попиксельные тепловые карты на Esri WorldImagery подложке
  10. Сохраняем PNG -> math_statistics/output/plots/
      Сохраняем CSV пикселей -> approximated/tiff/

Запуск: python approximated/pixel_ndvi_real.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import requests
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
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
import ee
from pathlib import Path

# ─── Параметры ───────────────────────────────────────────────────
YEAR        = 2023
FARM        = "Агро Парасат"

# Медианный мозаика: все чистые снимки за вегетационный сезон
DATE_START  = "2023-05-01"
DATE_END    = "2023-09-30"
CLOUD_MAX   = 20          # макс. % облачности на снимок
IMAGE_LABEL = "median-mosaic 2023 May-Sep"

SCALE_M     = 10          # нативное разрешение S2 (метров)
TILE_ZOOM   = 14          # zoom Esri тайлов (14 = ~5м/пиксель)
ALPHA       = 0.52        # прозрачность химического слоя поверх спутника

# Лучший спектральный предиктор для каждого элемента (Spearman ρ на 1215 точках)
BEST_PREDICTOR = {
    "ph":  ("s2_NDRE_spring",  "ndre"),   # ρ = -0.616
    "k":   ("s2_BSI_spring",   "bsi"),    # ρ = -0.478
    "p":   ("s2_GNDVI_spring", "gndvi"),  # ρ = +0.254
    "hu":  ("s2_EVI_summer",   "evi"),    # ρ = +0.200
    "s":   ("s2_GNDVI_autumn", "gndvi"),  # ρ = +0.323
    "no3": ("s2_GNDVI_spring", "gndvi"),  # ρ = -0.298
}

CHEM_LABELS = {
    "ph":  "pH",
    "k":   "K, мг/кг",
    "p":   "P, мг/кг",
    "hu":  "Гумус, %",
    "s":   "S, мг/кг",
    "no3": "NO₃, мг/кг",
}

CHEM_CMAPS = {
    "ph":  "RdYlGn_r",
    "k":   "YlOrRd",
    "p":   "YlGn",
    "hu":  "BrBG",
    "s":   "PuBuGn",
    "no3": "OrRd",
}

BASE        = Path(__file__).parent
DATA_PATH   = BASE.parent / "data" / "features" / "full_dataset.csv"
TIFF_DIR    = BASE / "tiff"
OUT_DIR     = BASE.parent / "math_statistics" / "output" / "plots"
TIFF_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIFF_PATH   = TIFF_DIR / "s2_2023_summer_mosaic_B2B4B8B3B5B11.tif"
PIXELS_CSV  = TIFF_DIR / f"pixels_{FARM.replace(' ', '_')}_{YEAR}_real.csv"

WGS84       = "EPSG:4326"
TIFF_CRS    = "EPSG:32641"   # UTM Zone 41N (Kazakhstan)
MERC        = "EPSG:3857"

t_tiff_wgs = Transformer.from_crs(TIFF_CRS, WGS84,  always_xy=True)
t_wgs_merc = Transformer.from_crs(WGS84,    MERC,   always_xy=True)

def to_merc(lo, la):
    return t_wgs_merc.transform(np.asarray(lo), np.asarray(la))

# ─── 1. GEE авторизация ──────────────────────────────────────────
print("Initializing GEE ...")
ee.Initialize()
print("  OK")

# ─── 2. Загрузка полигонов полей ─────────────────────────────────
print(f"Loading {FARM} field polygons ...")
df_full = pd.read_csv(DATA_PATH)
farm_df = df_full[(df_full["year"] == YEAR) & (df_full["farm"] == FARM)].copy()

# Все колонки нужные для регрессионных предикторов
pred_csv_cols = list({v[0] for v in BEST_PREDICTOR.values()})
chem_cols     = list(BEST_PREDICTOR.keys())

agg = (
    farm_df.groupby("field_name")
    .agg(
        geometry_wkt=("geometry_wkt", "first"),
        **{c: (c, "mean") for c in chem_cols + pred_csv_cols},
    )
    .reset_index()
)
print(f"  {len(agg)} fields")

gdf = gpd.GeoDataFrame(
    agg,
    geometry=agg["geometry_wkt"].apply(shapely_wkt.loads),
    crs=WGS84,
)
bounds = gdf.total_bounds  # [minx, miny, maxx, maxy] in WGS84

# ─── 3. Скачать GeoTIFF (если ещё нет) ───────────────────────────
if TIFF_PATH.exists():
    sz = TIFF_PATH.stat().st_size / 1024 / 1024
    print(f"TIFF already exists: {TIFF_PATH.name} ({sz:.1f} MB) — skipping download")
else:
    print(f"Building GEE mosaic and downloading GeoTIFF ...")
    print(f"  Period  : {DATE_START} to {DATE_END}  (cloud < {CLOUD_MAX}%)")
    print(f"  Bands   : B2, B4, B8, B3, B5, B11")
    print(f"  Scale   : {SCALE_M}m")

    ee_bbox = ee.Geometry.Rectangle(
        [bounds[0], bounds[1], bounds[2], bounds[3]]
    )

    # SCL cloud mask: exclude shadow(3), cloud-med(8), cloud-high(9), cirrus(10)
    def mask_clouds(img):
        scl = img.select("SCL")
        mask = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))
        return img.updateMask(mask)

    mosaic = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ee_bbox)
        .filterDate(DATE_START, DATE_END)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_MAX))
        .map(mask_clouds)
        .select(["B2", "B4", "B8", "B3", "B5", "B11"])
        .median()
    )

    url = mosaic.getDownloadURL({
        "scale":  SCALE_M,
        "region": ee_bbox,
        "format": "GEO_TIFF",
        "bands":  ["B2", "B4", "B8", "B3", "B5", "B11"],
        "crs":    TIFF_CRS,
    })
    print(f"  Download URL obtained, fetching ...")

    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(TIFF_PATH, "wb") as f:
        for chunk in resp.iter_content(1024 * 512):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded/1024/1024:.1f} MB / {total/1024/1024:.1f} MB  "
                      f"({downloaded/total*100:.0f}%)", end="", flush=True)
    print()
    sz = TIFF_PATH.stat().st_size / 1024 / 1024
    print(f"  Saved: {TIFF_PATH.name}  ({sz:.1f} MB)")

# ─── 4. Проверка TIFF ────────────────────────────────────────────
with rasterio.open(TIFF_PATH) as src:
    tiff_crs_actual = src.crs.to_string()
    b1_check = src.read(1)
    valid_all = (b1_check > 0).sum()
    print(f"TIFF: {src.count} bands  {src.height}×{src.width} px  "
          f"CRS={src.crs.to_epsg()}  "
          f"valid={valid_all:,}/{b1_check.size:,} ({valid_all/b1_check.size*100:.0f}%)")
    if tiff_crs_actual != TIFF_CRS:
        print(f"  Note: actual CRS={tiff_crs_actual}, adjusting transformer")
    t_tiff_wgs = Transformer.from_crs(tiff_crs_actual, WGS84, always_xy=True)

# ─── 5. Попиксельное вычисление индексов по каждому полю ─────────
if PIXELS_CSV.exists():
    print(f"\nPixel CSV exists: {PIXELS_CSV.name} — loading ...")
    pixels_df = pd.read_csv(PIXELS_CSV)
    print(f"  {len(pixels_df):,} pixels loaded")
else:
    print(f"\nExtracting pixels from TIFF field by field ...")
    gdf_proj = gdf.to_crs(tiff_crs_actual)

    pixel_rows = []
    with rasterio.open(TIFF_PATH) as src:
        for _, field in gdf_proj.iterrows():
            fname = field["field_name"]
            geom  = field["geometry"]

            try:
                out_img, out_tf = rio_mask(src, [geom], crop=True, nodata=0)
            except Exception as e:
                print(f"  Skip {fname}: {e}")
                continue

            # Бенды: B2=0, B4=1, B8=2, B3=3, B5=4, B11=5
            b2  = out_img[0].astype(float)
            b4  = out_img[1].astype(float)
            b8  = out_img[2].astype(float)
            b3  = out_img[3].astype(float)
            b5  = out_img[4].astype(float)
            b11 = out_img[5].astype(float)

            valid = (b4 > 0) & (b8 > 0)
            ri, ci = np.where(valid)

            if len(ri) == 0:
                print(f"  {fname}: no valid pixels")
                continue

            # Координаты центров пикселей -> WGS84
            px_x, px_y = rasterio.transform.xy(out_tf, ri, ci, offset="center")
            px_x = np.array(px_x, dtype=float)
            px_y = np.array(px_y, dtype=float)
            lon, lat = t_tiff_wgs.transform(px_x, px_y)

            # Рефлектанс (S2 L2A: DN / 10000)
            b2r  = b2[ri, ci]  / 10000.0
            b4r  = b4[ri, ci]  / 10000.0
            b8r  = b8[ri, ci]  / 10000.0
            b3r  = b3[ri, ci]  / 10000.0
            b5r  = b5[ri, ci]  / 10000.0
            b11r = b11[ri, ci] / 10000.0

            # Спектральные индексы для каждого пикселя
            eps = 1e-9
            ndvi  = np.clip((b8r - b4r)  / (b8r + b4r  + eps), -1.0,  1.0)
            ndre  = np.clip((b8r - b5r)  / (b8r + b5r  + eps), -1.0,  1.0)
            gndvi = np.clip((b8r - b3r)  / (b8r + b3r  + eps), -1.0,  1.0)
            evi   = np.clip(2.5 * (b8r - b4r) /
                            (b8r + 6*b4r - 7.5*b3r + 1 + eps), -2.0, 2.0)
            # BSI: standard formula uses Blue (B2), not Green (B3)
            bsi   = np.clip(((b11r + b4r) - (b8r + b2r)) /
                            ((b11r + b4r) + (b8r + b2r) + eps), -1.0, 1.0)

            n = len(ri)
            for j in range(n):
                pixel_rows.append({
                    "pixel_id":   f"{fname}_{ri[j]}_{ci[j]}",
                    "field_name": fname,
                    "lon":        lon[j],
                    "lat":        lat[j],
                    "utm_x":      px_x[j],
                    "utm_y":      px_y[j],
                    "B4":         b4[ri[j], ci[j]],
                    "B8":         b8[ri[j], ci[j]],
                    "ndvi":       ndvi[j],
                    "ndre":       ndre[j],
                    "gndvi":      gndvi[j],
                    "evi":        evi[j],
                    "bsi":        bsi[j],
                })
            print(f"  {fname:12s}: {n:5,} px  "
                  f"NDVI={ndvi.mean():.3f}  NDRE={ndre.mean():.3f}  "
                  f"BSI={bsi.mean():.3f}")

    pixels_df = pd.DataFrame(pixel_rows)
    pixels_df.to_csv(PIXELS_CSV, index=False, float_format="%.6f")
    print(f"\n  Total: {len(pixels_df):,} pixels saved -> {PIXELS_CSV.name}")

# ─── 6. Обучение регрессий chem ~ spectral_index ─────────────────
print("\nTraining regression models on full dataset (n=1,215) ...")
models = {}

for col, (csv_pred, px_col) in BEST_PREDICTOR.items():
    sub = df_full[[csv_pred, col]].dropna()
    if len(sub) < 30:
        print(f"  Skip {col}: n={len(sub)}")
        continue

    X = sub[[csv_pred]].values
    y = sub[col].values
    sX = StandardScaler().fit(X)
    sy = StandardScaler().fit(y.reshape(-1, 1))
    reg = HuberRegressor(epsilon=1.5, max_iter=500)
    reg.fit(sX.transform(X), sy.transform(y.reshape(-1, 1)).ravel())

    r, pval = spearmanr(X.ravel(), y)
    print(f"  {col:4s} ~ {px_col:6s}: Spearman ρ={r:+.3f}  p={pval:.4f}  n={len(sub)}")
    models[col] = (sX, sy, reg, r, px_col)

# ─── 7. Аппроксимация химии для каждого пикселя ──────────────────
print("\nApproximating chemistry per pixel ...")
for col, (sX, sy, reg, r, px_col) in models.items():
    X_px = pixels_df[px_col].values.reshape(-1, 1)
    y_hat = sy.inverse_transform(
        reg.predict(sX.transform(X_px)).reshape(-1, 1)
    ).ravel()
    pixels_df[f"approx_{col}"] = y_hat
    print(f"  {col:4s}: mean={y_hat.mean():.3f}  std={y_hat.std():.3f}")

# Mercator для рендера
mx, my = to_merc(pixels_df["lon"].values, pixels_df["lat"].values)
pixels_df["mx"] = mx
pixels_df["my"] = my

total_px  = len(pixels_df)
xmin, xmax = mx.min() - 80,  mx.max() + 80
ymin, ymax = my.min() - 80,  my.max() + 80
farm_slug = FARM.replace(" ", "_")

# ─── 8a. Загрузка внешних пикселей (вне полигонов) для Approximate ──
print("\nExtracting APPROXIMATE pixels (outside field polygons) from full TIFF bbox ...")

# Маска: True там где ВНЕ всех полигонов (в координатах TIFF_CRS)
gdf_proj_tiff = gdf.to_crs(tiff_crs_actual)
field_geoms_tiff = list(gdf_proj_tiff.geometry)

with rasterio.open(TIFF_PATH) as src:
    tiff_transform = src.transform
    tiff_shape = (src.height, src.width)
    tiff_bounds = src.bounds

    # Маска: True где ВНЕ всех полигонов (inside_field=False -> outside=True)
    inside_mask = ~geometry_mask(
        field_geoms_tiff,
        transform=tiff_transform,
        invert=False,   # False = pixels INSIDE polygons are masked (=True means NOT inside)
        out_shape=tiff_shape,
    )
    # Recompute: geometry_mask returns True for pixels OUTSIDE geometries by default
    # So geometry_mask(..., invert=False) -> True = outside field polygons
    outside_mask = geometry_mask(
        field_geoms_tiff,
        transform=tiff_transform,
        invert=False,
        out_shape=tiff_shape,
    )

    b2_full  = src.read(1).astype(float)
    b4_full  = src.read(2).astype(float)
    b8_full  = src.read(3).astype(float)
    b3_full  = src.read(4).astype(float)
    b5_full  = src.read(5).astype(float)
    b11_full = src.read(6).astype(float)

# Valid pixels outside polygons (data present + outside field boundaries)
valid_outside = outside_mask & (b4_full > 0) & (b8_full > 0)
ro, co = np.where(valid_outside)

print(f"  {valid_outside.sum():,} valid pixels outside field polygons")

if len(ro) > 0:
    # Subsample if too many (keep up to 300k for performance)
    MAX_OUTSIDE = 300_000
    if len(ro) > MAX_OUTSIDE:
        idx = np.random.choice(len(ro), MAX_OUTSIDE, replace=False)
        ro, co = ro[idx], co[idx]
        print(f"  Subsampled to {MAX_OUTSIDE:,} pixels")

    px_x_out, px_y_out = rasterio.transform.xy(tiff_transform, ro, co, offset="center")
    px_x_out = np.array(px_x_out, dtype=float)
    px_y_out = np.array(px_y_out, dtype=float)
    lon_out, lat_out = t_tiff_wgs.transform(px_x_out, px_y_out)

    eps = 1e-9
    b2r_o  = b2_full[ro, co]  / 10000.0
    b4r_o  = b4_full[ro, co]  / 10000.0
    b8r_o  = b8_full[ro, co]  / 10000.0
    b3r_o  = b3_full[ro, co]  / 10000.0
    b5r_o  = b5_full[ro, co]  / 10000.0
    b11r_o = b11_full[ro, co] / 10000.0

    ndvi_o  = np.clip((b8r_o - b4r_o)  / (b8r_o + b4r_o  + eps), -1.0,  1.0)
    ndre_o  = np.clip((b8r_o - b5r_o)  / (b8r_o + b5r_o  + eps), -1.0,  1.0)
    gndvi_o = np.clip((b8r_o - b3r_o)  / (b8r_o + b3r_o  + eps), -1.0,  1.0)
    evi_o   = np.clip(2.5 * (b8r_o - b4r_o) /
                      (b8r_o + 6*b4r_o - 7.5*b3r_o + 1 + eps), -2.0, 2.0)
    # BSI: standard formula uses Blue (B2)
    bsi_o   = np.clip(((b11r_o + b4r_o) - (b8r_o + b2r_o)) /
                      ((b11r_o + b4r_o) + (b8r_o + b2r_o) + eps), -1.0, 1.0)

    outside_df = pd.DataFrame({
        "lon": lon_out, "lat": lat_out,
        "ndvi": ndvi_o, "ndre": ndre_o,
        "gndvi": gndvi_o, "evi": evi_o, "bsi": bsi_o,
    })

    # Approximate chemistry for outside pixels
    for col, (sX, sy, reg, r, px_col) in models.items():
        X_out = outside_df[px_col].values.reshape(-1, 1)
        y_hat_o = sy.inverse_transform(
            reg.predict(sX.transform(X_out)).reshape(-1, 1)
        ).ravel()
        outside_df[f"approx_{col}"] = y_hat_o

    mx_out, my_out = to_merc(outside_df["lon"].values, outside_df["lat"].values)
    outside_df["mx"] = mx_out
    outside_df["my"] = my_out
    print(f"  Outside pixels approximated: {len(outside_df):,}")
else:
    outside_df = pd.DataFrame()
    print("  No outside pixels found")

# ─── 8b. Полигоны полей в Mercator для "Validate" аннотаций ──────
gdf_merc = gdf.to_crs(MERC)

def _draw_validate_polygons(ax):
    """Draw black polygon outlines + 'Validate' labels for each field."""
    for _, row in gdf_merc.iterrows():
        geom = row.geometry
        # Draw polygon boundary
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            polys = []

        for poly in polys:
            x_coords, y_coords = poly.exterior.xy
            ax.plot(x_coords, y_coords,
                    color="black", linewidth=1.8, zorder=6, solid_capstyle="round")
            ax.plot(x_coords, y_coords,
                    color="white", linewidth=0.6, zorder=7,
                    linestyle="--", alpha=0.6, solid_capstyle="round")

        # "Validate" label at centroid
        cen_x = geom.centroid.x
        cen_y = geom.centroid.y
        ax.text(cen_x, cen_y, "Validate",
                ha="center", va="center",
                fontsize=5.5, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="black",
                          edgecolor="white", linewidth=0.6, alpha=0.75),
                zorder=8)

def _draw_approximate_label(ax):
    """Draw 'Approximate' label in the surrounding (non-field) area."""
    # Place in lower-left corner of the map extent
    lx = xmin + (xmax - xmin) * 0.04
    ly = ymin + (ymax - ymin) * 0.04
    ax.text(lx, ly, "Approximate",
            ha="left", va="bottom",
            fontsize=9, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                      edgecolor="#888888", linewidth=1.0, alpha=0.85),
            zorder=9)

# ─── 8c. Функция рендера: Approximate (фон) + Validate (полигоны) ──
def render_map(val_col, title, cmap_, fname, vmin=None, vmax=None, footer_extra=""):
    """
    Render a dual-zone chemistry map:
      - Approximate zone: model-predicted pixels OUTSIDE field polygons (background)
      - Validate zone:    real Sentinel-2 pixels INSIDE field polygons + black outline
    """
    sub_in = pixels_df[["mx", "my", val_col]].dropna()
    if len(sub_in) == 0:
        print(f"  Skip {fname}: no inside data")
        return

    vals_in = sub_in[val_col]
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

    # ── Layer 1: Approximate pixels OUTSIDE field polygons ──────────
    if len(outside_df) > 0 and val_col in outside_df.columns:
        sub_out = outside_df[["mx", "my", val_col]].dropna()
        vals_out = sub_out[val_col]
        ax.scatter(sub_out["mx"], sub_out["my"],
                   c=vals_out, cmap=cmap_,
                   s=0.5, alpha=ALPHA * 0.85, linewidths=0,
                   vmin=v0, vmax=v1,
                   zorder=2, rasterized=True)
        _draw_approximate_label(ax)

    # ── Layer 2: Real (Validate) pixels INSIDE field polygons ───────
    ax.scatter(sub_in["mx"], sub_in["my"],
               c=vals_in, cmap=cmap_,
               s=0.6, alpha=ALPHA, linewidths=0,
               vmin=v0, vmax=v1,
               zorder=3, rasterized=True)

    # ── Layer 3: Polygon outlines + Validate labels ──────────────────
    _draw_validate_polygons(ax)

    # Colorbar
    sm = ScalarMappable(cmap=cmap_, norm=Normalize(v0, v1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.70, pad=0.02, aspect=22)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)
    cb.outline.set_edgecolor("#333333")

    ax.set_title(title, fontsize=12, fontweight="bold", color="white", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")

    fig.text(
        0.5, 0.01,
        f"{FARM}  |  {total_px:,} real Sentinel-2 pixels @ 10m  "
        f"|  {IMAGE_LABEL}  |  Esri WorldImagery{footer_extra}",
        ha="center", va="bottom", color="#666666", fontsize=7.5,
    )
    fig.tight_layout(pad=0.4, rect=[0, 0.03, 1, 1])

    out = OUT_DIR / fname
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)
    print(f"  Saved: {out.name}")

# ─── 9. NDVI карта ───────────────────────────────────────────────
print("\nRendering NDVI heatmap ...")
render_map(
    "ndvi",
    "NDVI — реальные пиксели 10м (Sentinel-2 median mosaic)",
    "RdYlGn",
    f"real_pixel_{farm_slug}_{YEAR}_NDVI.png",
    vmin=-0.05, vmax=0.75,
)

# ─── 10. Карты химических элементов ──────────────────────────────
print("\nRendering chemistry heatmaps ...")
for col, label in CHEM_LABELS.items():
    if col not in models:
        continue
    _, _, _, r, px_col = models[col]
    render_map(
        f"approx_{col}",
        label,
        CHEM_CMAPS[col],
        f"real_pixel_{farm_slug}_{YEAR}_{col}_approx.png",
        footer_extra=f"  |  proxy: {px_col.upper()} (Spearman ρ={r:+.3f})",
    )

# ─── 11. Сводный plot (NDVI + 6 элементов) ───────────────────────
print("\nRendering summary figure (1×7) ...")
cols_all   = ["ndvi"] + [f"approx_{c}" for c in CHEM_LABELS]
labels_all = ["NDVI (Sentinel-2)"] + list(CHEM_LABELS.values())
cmaps_all  = ["RdYlGn"] + [CHEM_CMAPS[c] for c in CHEM_LABELS]

fig, axes = plt.subplots(1, 7, figsize=(40, 7))
fig.patch.set_facecolor("#0a0a0a")

for i, (vc, lb, cm) in enumerate(zip(cols_all, labels_all, cmaps_all)):
    ax = axes[i]
    ax.set_facecolor("#111111")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery,
                   zoom=TILE_ZOOM, alpha=1.0)

    sub_in = pixels_df[["mx", "my", vc]].dropna()
    if len(sub_in) == 0:
        ax.set_title(lb, color="#555", fontsize=8)
        continue

    vals_in = sub_in[vc]
    v0, v1 = vals_in.quantile(0.02), vals_in.quantile(0.98)

    # Outside / Approximate layer
    if len(outside_df) > 0 and vc in outside_df.columns:
        sub_out = outside_df[["mx", "my", vc]].dropna()
        ax.scatter(sub_out["mx"], sub_out["my"],
                   c=sub_out[vc], cmap=cm,
                   s=0.4, alpha=ALPHA * 0.8, linewidths=0,
                   vmin=v0, vmax=v1,
                   zorder=2, rasterized=True)
        _draw_approximate_label(ax)

    # Inside / Validate layer
    ax.scatter(sub_in["mx"], sub_in["my"],
               c=vals_in, cmap=cm,
               s=0.5, alpha=ALPHA, linewidths=0,
               vmin=v0, vmax=v1,
               zorder=3, rasterized=True)

    # Polygon outlines
    _draw_validate_polygons(ax)

    sm = ScalarMappable(cmap=cm, norm=Normalize(v0, v1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.70, pad=0.02, aspect=22)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
    cb.outline.set_edgecolor("#333333")

    ax.set_title(lb, fontsize=8, fontweight="bold", color="white", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")

fig.suptitle(
    f"Real Sentinel-2 pixels 10m  —  {FARM}, {YEAR}  |  "
    f"{total_px:,} px  |  {IMAGE_LABEL}",
    fontsize=10, color="white", y=1.008,
)
fig.tight_layout(pad=0.5)
out = OUT_DIR / f"real_pixel_{farm_slug}_{YEAR}_summary.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig)
print(f"  Saved: {out.name}")

print(f"\nDone!")
print(f"  Pixels CSV  : {PIXELS_CSV}")
print(f"  TIFF        : {TIFF_PATH}")
print(f"  Output plots: {OUT_DIR}")
print(f"\nPixel sample:")
print(pixels_df[["pixel_id", "field_name", "lon", "lat", "ndvi",
                  "ndre", "bsi", "approx_ph", "approx_k"]].head(5).to_string(index=False))
