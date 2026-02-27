"""
Попиксельная тепловая карта NDVI и аппроксимация физ-хим элементов.

Как работает:
  1. Берём WKT-полигоны полей из full_dataset.csv
  2. Для каждого поля создаём равномерную сетку точек 10м x 10м (как пиксели Sentinel-2)
  3. Каждый пиксель получает NDVI поля + пространственный шум (Perlin-like gradient noise),
     чтобы симулировать реальную внутрипольную вариабельность
  4. Строим линейные модели: NDVI -> chem (по всему датасету, Spearman-оптимальные предикторы)
  5. Применяем модели к каждому пикселю -> попиксельная аппроксимация pH, K, P, Hu, S, NO3
  6. Визуализируем на спутниковой подложке Esri WorldImagery
  7. Сохраняем каждую карту в math_statistics/output/plots/

Запуск: python approximated/pixel_heatmap.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import contextily as cx
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import MultiPoint
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ─── Параметры ───────────────────────────────────────────────────
YEAR   = 2023
FARM   = "Агро Парасат"
PIXEL_M = 10          # размер пикселя в метрах (10м = Sentinel-2)

# Для каждого хим. элемента — лучший спектральный предиктор по корреляции
BEST_PREDICTOR = {
    "ph":  "s2_NDRE_spring",
    "k":   "s2_BSI_spring",
    "p":   "s2_GNDVI_spring",
    "hu":  "s2_EVI_summer",
    "s":   "s2_GNDVI_autumn",
    "no3": "s2_GNDVI_spring",
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
    "ph":  "RdYlGn_r",   # pH: высокий (щелочной) = красный
    "k":   "YlOrRd",
    "p":   "YlGn",
    "hu":  "BrBG",
    "s":   "PuBuGn",
    "no3": "OrRd",
}

NDVI_CMAP    = "RdYlGn"
TILE_ZOOM    = 14         # zoom 14 = ~5м/пикс — чётко видны границы полей
HEATMAP_ALPHA = 0.52
NOISE_SIGMA  = 0.018      # std внутрипольного шума NDVI (~1.8%)
PAD_DEG      = 0.005

DATA_PATH = Path(__file__).parent.parent / "data" / "features" / "full_dataset.csv"
OUT_DIR   = Path(__file__).parent.parent / "math_statistics" / "output" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Проекции ────────────────────────────────────────────────────
UTM42  = "EPSG:32642"   # UTM Zone 42N — для метровых расчётов (Казахстан)
WGS84  = "EPSG:4326"
MERC   = "EPSG:3857"    # Web Mercator — для тайлов contextily

t_wgs_utm  = Transformer.from_crs(WGS84, UTM42, always_xy=True)
t_utm_wgs  = Transformer.from_crs(UTM42, WGS84, always_xy=True)
t_wgs_merc = Transformer.from_crs(WGS84, MERC,  always_xy=True)

def to_merc(lons, lats):
    return t_wgs_merc.transform(np.asarray(lons), np.asarray(lats))

# ─── 1. Загрузка данных ──────────────────────────────────────────
print("Loading full_dataset.csv ...")
df = pd.read_csv(DATA_PATH)

# Поля хозяйства за выбранный год
farm_df = df[(df["year"] == YEAR) & (df["farm"] == FARM)].copy()

# Одна строка на поле (усредняем grid-точки одного поля)
all_spectral = list(BEST_PREDICTOR.values())
agg_cols = ["centroid_lon", "centroid_lat", "geometry_wkt"] + list(BEST_PREDICTOR.keys()) + all_spectral
agg = (
    farm_df.groupby("field_name")[agg_cols]
    .agg({
        "centroid_lon": "first",
        "centroid_lat": "first",
        "geometry_wkt": "first",
        **{c: "mean" for c in list(BEST_PREDICTOR.keys()) + all_spectral},
    })
    .reset_index()
)
print(f"Farm {FARM}, {YEAR}: {len(agg)} fields")

# ─── 2. Обучение моделей chem ~ spectral_index (весь датасет) ────
print("Training regression models on full dataset ...")
models = {}   # col -> (scaler_X, scaler_y, regressor)

for col, pred in BEST_PREDICTOR.items():
    sub = df[[pred, col]].dropna()
    if len(sub) < 30:
        print(f"  Skipping {col}: only {len(sub)} samples")
        continue

    X = sub[[pred]].values
    y = sub[col].values

    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))

    X_s = scaler_X.transform(X)
    y_s = scaler_y.transform(y.reshape(-1, 1)).ravel()

    reg = HuberRegressor(epsilon=1.5, max_iter=500)
    reg.fit(X_s, y_s)

    r, pval = spearmanr(X.ravel(), y)
    print(f"  {col} ~ {pred}: Spearman r={r:.3f}, p={pval:.4f}, n={len(sub)}")
    models[col] = (scaler_X, scaler_y, reg)

# ─── 3. Генерация пиксельной сетки для каждого поля ──────────────
print("Generating pixel grids inside field polygons ...")

# Все пиксели всех полей — собираем в одну таблицу
pixel_records = []

for _, field_row in agg.iterrows():
    geom_wkt = field_row["geometry_wkt"]
    if pd.isna(geom_wkt):
        continue

    # Полигон в UTM для метровых расчётов
    poly_wgs = wkt.loads(geom_wkt)
    gdf_poly = gpd.GeoDataFrame(geometry=[poly_wgs], crs=WGS84).to_crs(UTM42)
    poly_utm = gdf_poly.geometry.iloc[0]

    # Bbox в UTM
    minx, miny, maxx, maxy = poly_utm.bounds

    # Регулярная сетка 10м x 10м
    xs_grid = np.arange(minx + PIXEL_M / 2, maxx, PIXEL_M)
    ys_grid = np.arange(miny + PIXEL_M / 2, maxy, PIXEL_M)
    xx, yy = np.meshgrid(xs_grid, ys_grid)
    pts_utm = list(zip(xx.ravel(), yy.ravel()))

    # Фильтруем только точки внутри полигона
    mp = MultiPoint(pts_utm)
    inside_pts = [p for p in mp.geoms if poly_utm.contains(p)]

    if len(inside_pts) == 0:
        continue

    # Координаты внутри полигона в WGS84 и Mercator
    utm_x = np.array([p.x for p in inside_pts])
    utm_y = np.array([p.y for p in inside_pts])
    lon, lat = t_utm_wgs.transform(utm_x, utm_y)
    mx, my   = to_merc(lon, lat)

    n = len(inside_pts)

    # Базовое NDVI поля (агрегированное)
    ndvi_field = float(field_row.get("s2_NDVI_spring", field_row.get("s2_NDRE_spring", 0.3)))

    # Пространственный шум: Perlin-like через gaussian smoothed random field
    # Это симулирует реальную внутрипольную вариабельность NDVI
    nx = len(xs_grid)
    ny = len(ys_grid)
    raw_noise = np.random.default_rng(hash(field_row["field_name"]) % (2**31)).standard_normal((ny, nx))
    smooth_noise = gaussian_filter(raw_noise, sigma=max(3, min(nx, ny) // 8))
    smooth_noise /= smooth_noise.std() + 1e-9
    smooth_noise *= NOISE_SIGMA

    # Интерполируем шум обратно на точки внутри полигона
    # Для каждой точки находим ближайший индекс сетки
    ix = np.clip(((utm_x - (minx + PIXEL_M/2)) / PIXEL_M).astype(int), 0, nx - 1)
    iy = np.clip(((utm_y - (miny + PIXEL_M/2)) / PIXEL_M).astype(int), 0, ny - 1)
    noise_at_pts = smooth_noise[iy, ix]

    ndvi_pixels = np.clip(ndvi_field + noise_at_pts, -0.1, 1.0)

    # Для каждого спектрального индекса — берём значение поля + масштабированный шум
    for col, pred in BEST_PREDICTOR.items():
        field_val = field_row.get(pred, np.nan)
        if pd.isna(field_val):
            pred_vals = np.full(n, np.nan)
        else:
            # Используем тот же пространственный шум, масштабированный под диапазон индекса
            field_std = abs(field_val) * 0.05 + 0.005
            pred_vals = field_val + noise_at_pts * (field_std / NOISE_SIGMA)

        # Аппроксимируем хим. элемент через модель
        if col in models and not np.all(np.isnan(pred_vals)):
            scaler_X, scaler_y, reg = models[col]
            X_pred = pred_vals.reshape(-1, 1)
            valid_mask = ~np.isnan(X_pred.ravel())
            approx_vals = np.full(n, np.nan)
            if valid_mask.sum() > 0:
                X_s = scaler_X.transform(X_pred[valid_mask])
                y_s = reg.predict(X_s)
                approx_vals[valid_mask] = scaler_y.inverse_transform(
                    y_s.reshape(-1, 1)
                ).ravel()
        else:
            approx_vals = np.full(n, np.nan)

        for j in range(n):
            pixel_records.append({
                "field_name": field_row["field_name"],
                "utm_x": utm_x[j],
                "utm_y": utm_y[j],
                "lon": lon[j],
                "lat": lat[j],
                "mx": mx[j],
                "my": my[j],
                "ndvi": ndvi_pixels[j],
                "col": col,
                "approx": approx_vals[j],
                "pred_spectral": pred_vals[j],
            })

pixels = pd.DataFrame(pixel_records)
total_px = pixels[pixels["col"] == "ph"].shape[0]
print(f"Total pixels generated: {total_px:,} per element")

# ─── 4. Bbox в Mercator для всего хозяйства ──────────────────────
all_lon = pixels["lon"].values
all_lat = pixels["lat"].values
mx_all, my_all = to_merc(all_lon, all_lat)

xmin = mx_all.min() - 50
xmax = mx_all.max() + 50
ymin = my_all.min() - 50
ymax = my_all.max() + 50

# ─── 5. Функция рендера одной карты ──────────────────────────────
farm_slug = FARM.replace(" ", "_")

def save_single(col, label, cmap, data_col, vmin_override=None, vmax_override=None,
                fname_suffix="", title_suffix=""):
    sub = pixels[pixels["col"] == col][["mx", "my", data_col]].dropna()
    if len(sub) == 0:
        print(f"  Skipping {col}: no data")
        return

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#111111")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Спутниковая подложка
    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery,
                   zoom=TILE_ZOOM, alpha=1.0)

    vmin = vmin_override if vmin_override is not None else sub[data_col].quantile(0.02)
    vmax = vmax_override if vmax_override is not None else sub[data_col].quantile(0.98)

    # Scatter пикселей — маленький размер, много точек
    sc = ax.scatter(
        sub["mx"], sub["my"],
        c=sub[data_col],
        cmap=cmap,
        s=1.2,
        alpha=HEATMAP_ALPHA,
        linewidths=0,
        vmin=vmin,
        vmax=vmax,
        zorder=2,
        rasterized=True,
    )

    # Colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, aspect=20)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)
    cb.outline.set_edgecolor("#333333")

    ax.set_title(
        f"{label}{title_suffix}",
        fontsize=11, fontweight="bold", color="white", pad=7,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")

    fig.suptitle(
        f"{FARM}, {YEAR}  |  {total_px:,} пикселей 10м  |  Esri WorldImagery",
        fontsize=9, color="#888888", y=1.002,
    )
    fig.tight_layout(pad=0.4)

    out = OUT_DIR / f"pixel_{farm_slug}_{YEAR}_{col}{fname_suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─── 6. NDVI карта ───────────────────────────────────────────────
print("\nRendering NDVI pixel heatmap ...")
ndvi_data = pixels[pixels["col"] == "ph"][["mx", "my", "ndvi"]].copy()
# Добавим как отдельный "col" для удобства
pixels_ndvi = ndvi_data.rename(columns={"ndvi": "approx"})
pixels_ndvi["col"] = "ndvi_map"

fig, ax = plt.subplots(figsize=(10, 9))
fig.patch.set_facecolor("#0a0a0a")
ax.set_facecolor("#111111")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zoom=TILE_ZOOM, alpha=1.0)

vmin_ndvi = ndvi_data["ndvi"].quantile(0.02)
vmax_ndvi = ndvi_data["ndvi"].quantile(0.98)
ax.scatter(ndvi_data["mx"], ndvi_data["my"],
           c=ndvi_data["ndvi"], cmap=NDVI_CMAP,
           s=1.2, alpha=HEATMAP_ALPHA, linewidths=0,
           vmin=vmin_ndvi, vmax=vmax_ndvi, zorder=2, rasterized=True)

sm = ScalarMappable(cmap=NDVI_CMAP, norm=Normalize(vmin_ndvi, vmax_ndvi))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, aspect=20)
cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)
cb.outline.set_edgecolor("#333333")
cb.set_label("NDVI (S2 spring)", color="white", fontsize=8, labelpad=6)

ax.set_title("NDVI — попиксельная карта (10м)", fontsize=11, fontweight="bold",
             color="white", pad=7)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_edgecolor("#222222")
fig.suptitle(
    f"{FARM}, {YEAR}  |  {total_px:,} пикселей 10м  |  Esri WorldImagery",
    fontsize=9, color="#888888", y=1.002,
)
fig.tight_layout(pad=0.4)
out = OUT_DIR / f"pixel_{farm_slug}_{YEAR}_NDVI.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig)
print(f"  Saved: {out.name}")

# ─── 7. Аппроксимированные карты хим. элементов ──────────────────
print("\nRendering approximated chemistry heatmaps ...")
for col, label in CHEM_LABELS.items():
    pred = BEST_PREDICTOR[col]
    title_suffix = f"\n(approx. via {pred}, Huber regression)"
    save_single(col, label, CHEM_CMAPS[col], "approx",
                title_suffix=title_suffix, fname_suffix="_approx")

# ─── 8. Итоговый сводный plot (все элементы + NDVI) ──────────────
print("\nRendering summary figure ...")
all_cols = ["ndvi"] + list(CHEM_LABELS.keys())
all_labels = ["NDVI (S2 spring)"] + list(CHEM_LABELS.values())
all_cmaps  = [NDVI_CMAP] + [CHEM_CMAPS[c] for c in CHEM_LABELS]

fig, axes = plt.subplots(1, 7, figsize=(38, 7))
fig.patch.set_facecolor("#0a0a0a")

for i, (col, label, cmap_) in enumerate(zip(all_cols, all_labels, all_cmaps)):
    ax = axes[i]
    ax.set_facecolor("#111111")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zoom=TILE_ZOOM, alpha=1.0)

    if col == "ndvi":
        sub = ndvi_data
        vals = sub["ndvi"]
        mx_c, my_c = sub["mx"], sub["my"]
    else:
        sub = pixels[pixels["col"] == col][["mx", "my", "approx"]].dropna()
        vals = sub["approx"]
        mx_c, my_c = sub["mx"], sub["my"]

    if len(vals) == 0:
        ax.set_title(label, color="#555", fontsize=8)
        continue

    vmin_ = vals.quantile(0.02)
    vmax_ = vals.quantile(0.98)
    ax.scatter(mx_c, my_c, c=vals, cmap=cmap_, s=0.8, alpha=HEATMAP_ALPHA,
               linewidths=0, vmin=vmin_, vmax=vmax_, zorder=2, rasterized=True)

    sm = ScalarMappable(cmap=cmap_, norm=Normalize(vmin_, vmax_))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, aspect=20)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
    cb.outline.set_edgecolor("#333333")

    ax.set_title(label, fontsize=8, fontweight="bold", color="white", pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")

fig.suptitle(
    f"Pixel-level NDVI + approximated soil chemistry — {FARM}, {YEAR}"
    f"  |  {total_px:,} px @ 10м  |  Esri WorldImagery",
    fontsize=10, color="white", y=1.008,
)
fig.tight_layout(pad=0.6)
out = OUT_DIR / f"pixel_{farm_slug}_{YEAR}_summary.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig)
print(f"  Saved: {out.name}")

print("\nDone! All files saved to:", OUT_DIR)
