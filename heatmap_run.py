"""
Тепловая карта химических элементов почвы для одного прогона (год).

Как устроена интерпретация спутниковых данных:
  GEE reduceRegion(mean) -> берёт ВСЕ пиксели (10м) внутри полигона поля
  и усредняет их -> одно значение спектрального индекса на поле на сезон.
  В full_dataset.csv каждая строка - grid-точка отбора пробы,
  но спутниковые признаки у всех точек одного поля одинаковые.

Запуск: python heatmap_run.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # интерактивное окно
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import contextily as cx
from pyproj import Transformer
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from pathlib import Path

# ─── Параметры прогона ────────────────────────────────────────────
YEAR = 2023
FARM = "Агро Парасат"  # компактное расположение полей

CHEM_COLS = ["ph", "k", "p", "hu", "s", "no3"]

CHEM_LABELS = {
    "ph":  "pH",
    "k":   "K, мг/кг",
    "p":   "P, мг/кг",
    "hu":  "Гумус, %",
    "s":   "S, мг/кг",
    "no3": "NO₃, мг/кг",
}

GRID_N = 600          # разрешение сетки интерполяции
BLUR_SIGMA = 14       # сила gaussian blur: выше = мягче холмы
HEATMAP_ALPHA = 0.40  # прозрачность тепловой карты поверх спутника
CMAP = "RdYlGn"
TILE_ZOOM = 13        # zoom для спутниковых тайлов (13 = ~10м/пиксель)
PAD_DEG = 0.008       # отступ вокруг bbox в градусах

DATA_PATH = Path(__file__).parent / "data" / "features" / "full_dataset.csv"
OUT_DIR = Path(__file__).parent / "math_statistics" / "output" / "plots"

# ─── WGS84 -> Web Mercator ────────────────────────────────────────
wgs84_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def to_mercator(lons, lats):
    xs, ys = wgs84_to_mercator.transform(lons, lats)
    return xs, ys

# ─── Загрузка данных ──────────────────────────────────────────────
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)

run = df[(df["year"] == YEAR) & (df["farm"] == FARM)].copy()
print(f"Run {YEAR} / {FARM}: {len(run)} grid points, {run['field_name'].nunique()} unique fields")

agg = (
    run.groupby(["farm", "field_name"])[["centroid_lon", "centroid_lat"] + CHEM_COLS]
    .mean()
    .reset_index()
)
print(f"After aggregation: {len(agg)} fields")

print()
print("-" * 65)
print("Satellite data methodology:")
print("  GEE reduceRegion(mean) -> mean over ALL 10m pixels")
print("  inside each field polygon -> 1 value per field per season.")
print("  Each dot on the map = one agricultural field centroid.")
print("-" * 65)
print()

# ─── Общий bbox в Mercator ───────────────────────────────────────
lons_all = agg["centroid_lon"].values
lats_all = agg["centroid_lat"].values

lon_min = lons_all.min() - PAD_DEG
lon_max = lons_all.max() + PAD_DEG
lat_min = lats_all.min() - PAD_DEG
lat_max = lats_all.max() + PAD_DEG

xmin, ymin = to_mercator(lon_min, lat_min)
xmax, ymax = to_mercator(lon_max, lat_max)

# Сетка интерполяции в Mercator
grid_x, grid_y = np.meshgrid(
    np.linspace(xmin, xmax, GRID_N),
    np.linspace(ymin, ymax, GRID_N),
)

# Точки полей в Mercator
xs_all, ys_all = to_mercator(lons_all, lats_all)

# ─── Подготовка папки для сохранения ─────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _render_ax(ax, fig_single, col, label, valid):
    """Рисует одну тепловую карту на переданный ax."""
    ax.set_facecolor("#111111")

    if len(valid) < 3:
        ax.set_title(label, fontsize=9, fontweight="bold", color="#555555", pad=5)
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                color="#444444", fontsize=9, transform=ax.transAxes)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery,
                       zoom=TILE_ZOOM, alpha=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#222222")
        return

    xs_pt, ys_pt = to_mercator(valid["centroid_lon"].values, valid["centroid_lat"].values)
    zs = valid[col].values

    grid_z = griddata(
        points=(xs_pt, ys_pt),
        values=zs,
        xi=(grid_x, grid_y),
        method="cubic",
    )
    nan_mask = np.isnan(grid_z)
    grid_z_filled = np.where(nan_mask, np.nanmean(grid_z), grid_z)
    grid_z_smooth = gaussian_filter(grid_z_filled, sigma=BLUR_SIGMA)
    grid_z_smooth[nan_mask] = np.nan

    vmin, vmax = np.nanmin(grid_z_smooth), np.nanmax(grid_z_smooth)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zoom=TILE_ZOOM, alpha=1.0)
    ax.imshow(grid_z_smooth, extent=[xmin, xmax, ymin, ymax], origin="lower", aspect="auto",
              cmap=CMAP, vmin=vmin, vmax=vmax, alpha=HEATMAP_ALPHA,
              interpolation="bilinear", zorder=2)
    ax.scatter(xs_pt, ys_pt, s=280, c="white", alpha=0.18, linewidths=0, zorder=3)
    ax.scatter(xs_pt, ys_pt, s=120, c="white", alpha=0.30, linewidths=0, zorder=3)
    ax.scatter(xs_pt, ys_pt, c=zs, cmap=CMAP, s=55, edgecolors="white",
               linewidths=0.9, vmin=vmin, vmax=vmax, zorder=4)

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cb = fig_single.colorbar(sm, ax=ax, shrink=0.72, pad=0.02, aspect=18)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
    cb.outline.set_edgecolor("#333333")

    ax.set_title(label, fontsize=9, fontweight="bold", color="white", pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")


# ─── Построение тепловых карт ─────────────────────────────────────
print("Downloading satellite tiles (Esri WorldImagery)...")
fig, axes = plt.subplots(1, 6, figsize=(28, 6))
fig.patch.set_facecolor("#0a0a0a")
axes = axes.ravel()

for i, col in enumerate(CHEM_COLS):
    label = CHEM_LABELS[col]
    valid = agg[["centroid_lon", "centroid_lat", col]].dropna()

    # Рисуем на общем figure
    _render_ax(axes[i], fig, col, label, valid)

    # Сохраняем отдельный файл для этого элемента
    fig_s, ax_s = plt.subplots(1, 1, figsize=(8, 7))
    fig_s.patch.set_facecolor("#0a0a0a")
    _render_ax(ax_s, fig_s, col, label, valid)
    farm_slug = FARM.replace(" ", "_")
    out_path = OUT_DIR / f"heatmap_{farm_slug}_{YEAR}_{col}.png"
    fig_s.suptitle(
        f"{label}  —  {FARM}, {YEAR}",
        fontsize=11, color="white", y=1.01,
    )
    fig_s.tight_layout(pad=0.5)
    fig_s.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig_s)
    print(f"  Saved: {out_path.name}")

fig.suptitle(
    f"Soil chemistry — {FARM}, {YEAR}  (n={len(agg)} fields)  |  Esri WorldImagery",
    fontsize=12,
    color="white",
    y=1.005,
)
fig.tight_layout(pad=0.8)

print("Opening heatmap window...")
plt.show()
