"""
generate_ph_map_basemap.py
---------------------------
Generate pH prediction map (fig12) with satellite/terrain basemap overlay.

Uses contextily for tile-based basemap and real RF OOF predictions
for the test farm "Агро Парасат" (Farm-LOFO scenario).

Ground truth measurements are shown as scatter points on the left panel;
RF predicted values are shown as a semi-transparent interpolated surface
over the satellite basemap on the right panel.

Output:
  articles/article2_prediction/figures/fig12_prediction_map_pH.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from pathlib import Path
import contextily as ctx

# ─── Config ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
RF_OOF   = ROOT / "ML" / "results" / "rf" / "ph_oof_predictions.csv"
FIG_DIR  = ROOT / "articles" / "article2_prediction" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TEST_FARM = "Агро Парасат"
ALPHA_OVERLAY = 0.65       # transparency for prediction surface
GRID_RES     = 200         # interpolation grid resolution
PAD_FRAC     = 0.15        # padding around data extent (fraction)
CMAP         = "RdYlGn"    # diverging: red=acidic, green=neutral
PH_VMIN      = 5.5
PH_VMAX      = 8.0

# Basemap provider — Esri World Imagery (satellite) or Terrain
BASEMAP = ctx.providers.Esri.WorldImagery
# Alternative: ctx.providers.Stadia.StamenTerrain

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def main():
    print("[INFO] Loading RF OOF predictions...")
    df = pd.read_csv(RF_OOF, usecols=["farm", "centroid_lon", "centroid_lat", "ph", "oof_pred"],
                     low_memory=False)
    sub = df[df["farm"] == TEST_FARM].copy().reset_index(drop=True)
    print(f"[INFO] Test farm '{TEST_FARM}': {len(sub)} points")

    lon = sub["centroid_lon"].values
    lat = sub["centroid_lat"].values
    gt  = sub["ph"].values
    pred = sub["oof_pred"].values

    # ── Compute extent with padding ───────────────────────────────────────────
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    dlon = (lon_max - lon_min) * PAD_FRAC
    dlat = (lat_max - lat_min) * PAD_FRAC
    extent = [lon_min - dlon, lon_max + dlon, lat_min - dlat, lat_max + dlat]

    # ── Grid for interpolation ────────────────────────────────────────────────
    grid_lon = np.linspace(extent[0], extent[1], GRID_RES)
    grid_lat = np.linspace(extent[2], extent[3], GRID_RES)
    grid_LON, grid_LAT = np.meshgrid(grid_lon, grid_lat)

    # Interpolate ground truth and predictions onto grid
    gt_grid   = griddata((lon, lat), gt,   (grid_LON, grid_LAT), method="cubic")
    pred_grid = griddata((lon, lat), pred, (grid_LON, grid_LAT), method="cubic")

    norm = Normalize(vmin=PH_VMIN, vmax=PH_VMAX)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Panel 1: Ground Truth ---
    ax1.set_xlim(extent[0], extent[1])
    ax1.set_ylim(extent[2], extent[3])

    # Add basemap
    try:
        ctx.add_basemap(ax1, crs="EPSG:4326", source=BASEMAP, zoom=14, attribution="")
    except Exception as e:
        print(f"[WARN] Basemap download failed for panel 1: {e}")

    # Interpolated GT surface with transparency
    im1 = ax1.imshow(gt_grid, extent=[extent[0], extent[1], extent[2], extent[3]],
                     origin="lower", cmap=CMAP, norm=norm, alpha=ALPHA_OVERLAY,
                     aspect="auto", zorder=2)
    # Scatter GT points
    sc1 = ax1.scatter(lon, lat, c=gt, cmap=CMAP, norm=norm, s=30,
                      edgecolors="black", linewidths=0.5, zorder=3)

    ax1.set_title("Ground Truth pH (KCl)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="pH (KCl)")

    # --- Panel 2: RF Predicted ---
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])

    try:
        ctx.add_basemap(ax2, crs="EPSG:4326", source=BASEMAP, zoom=14, attribution="")
    except Exception as e:
        print(f"[WARN] Basemap download failed for panel 2: {e}")

    # Interpolated prediction surface with transparency
    im2 = ax2.imshow(pred_grid, extent=[extent[0], extent[1], extent[2], extent[3]],
                     origin="lower", cmap=CMAP, norm=norm, alpha=ALPHA_OVERLAY,
                     aspect="auto", zorder=2)
    # Scatter predicted points
    sc2 = ax2.scatter(lon, lat, c=pred, cmap=CMAP, norm=norm, s=30,
                      edgecolors="black", linewidths=0.5, zorder=3)

    ax2.set_title("RF Predicted pH (Farm-LOFO)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="pH (KCl)")

    fig.suptitle(f"Spatial Prediction Map: pH (KCl) — Test Farm \"{TEST_FARM}\"\n"
                 f"Farm-LOFO scenario (model trained on 19 other farms, N={len(sub)} test points)",
                 fontsize=13, y=1.02)

    plt.tight_layout()
    out = FIG_DIR / "fig12_prediction_map_pH.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[DONE] Saved: {out}")


if __name__ == "__main__":
    main()
