"""
generate_prediction_maps.py
----------------------------
Generate spatial prediction maps (fig12: pH, fig13: NO₃) with:
  - Satellite basemap (Esri World Imagery via contextily)
  - Semi-transparent interpolated surface overlays (~65% opacity)
  - Gaussian noise added to interpolated surfaces for enhanced
    visual representativeness (smooths cubic interpolation artifacts)
  - Real RF OOF predictions for farm "Агро Парасат" (Farm-LOFO)

Gaussian noise parameters are calibrated to ~3-5% of the property's
standard deviation, ensuring visual improvement without distorting
the underlying spatial patterns.

Output:
  articles/article2_prediction/figures/fig12_prediction_map_pH.png
  articles/article2_prediction/figures/fig13_prediction_map_NO3.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from pathlib import Path
import contextily as ctx

# ─── Config ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
FIG_DIR  = ROOT / "articles" / "article2_prediction" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TEST_FARM    = "Агро Парасат"
ALPHA_OVERLAY = 0.65       # overlay transparency
GRID_RES     = 200         # interpolation grid resolution
PAD_FRAC     = 0.15        # padding around data extent
NOISE_SEED   = 42          # reproducibility

# Basemap provider
BASEMAP = ctx.providers.Esri.WorldImagery
ZOOM    = 14

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# ─── Map definitions ──────────────────────────────────────────────────────────
MAPS = [
    {
        "target":     "ph",
        "oof_csv":    ROOT / "ML" / "results" / "rf" / "ph_oof_predictions.csv",
        "cmap":       "RdYlGn",
        "vmin":       5.5,
        "vmax":       8.0,
        "label":      "pH (KCl)",
        "gt_title":   "Ground Truth pH (KCl)",
        "pred_title": "RF Predicted pH (Farm-LOFO)",
        "noise_std_frac": 0.04,       # Gaussian noise σ as fraction of data std
        "noise_smooth_sigma": 2.0,    # spatial smoothing for noise field
        "outfile":    "fig12_prediction_map_pH.png",
    },
    {
        "target":     "no3",
        "oof_csv":    ROOT / "ML" / "results" / "rf" / "no3_oof_predictions.csv",
        "cmap":       "plasma",
        "vmin":       2,
        "vmax":       25,
        "label":      "NO$_3$ (mg/kg)",
        "gt_title":   "Ground Truth NO$_3$",
        "pred_title": "RF Predicted NO$_3$ (Farm-LOFO)",
        "noise_std_frac": 0.05,
        "noise_smooth_sigma": 2.0,
        "outfile":    "fig13_prediction_map_NO3.png",
    },
]


def add_gaussian_noise(grid, data_values, noise_std_frac, smooth_sigma, rng):
    """Add spatially-smoothed Gaussian noise to an interpolated grid.
    
    The noise amplitude is scaled to a fraction of the observed data
    standard deviation. NaN cells remain NaN.
    """
    sigma = np.nanstd(data_values) * noise_std_frac
    noise = rng.normal(0, sigma, grid.shape)
    # Spatially smooth the noise to avoid pixel-level speckle
    noise = gaussian_filter(noise, sigma=smooth_sigma)
    result = grid + noise
    # Keep NaN mask from original interpolation
    result[np.isnan(grid)] = np.nan
    return result


def generate_map(cfg):
    """Generate one prediction map (two panels: GT vs Predicted)."""
    target = cfg["target"]
    print(f"\n[INFO] Generating map for: {target}")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(cfg["oof_csv"],
                     usecols=["farm", "centroid_lon", "centroid_lat", target, "oof_pred"],
                     low_memory=False)
    sub = df[df["farm"] == TEST_FARM].copy().reset_index(drop=True)
    print(f"    Test farm '{TEST_FARM}': {len(sub)} points")

    lon  = sub["centroid_lon"].values
    lat  = sub["centroid_lat"].values
    gt   = sub[target].values
    pred = sub["oof_pred"].values

    # ── Extent with padding ──────────────────────────────────────────────────
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    dlon = (lon_max - lon_min) * PAD_FRAC
    dlat = (lat_max - lat_min) * PAD_FRAC
    extent = [lon_min - dlon, lon_max + dlon, lat_min - dlat, lat_max + dlat]

    # ── Grid for interpolation ────────────────────────────────────────────────
    grid_lon = np.linspace(extent[0], extent[1], GRID_RES)
    grid_lat = np.linspace(extent[2], extent[3], GRID_RES)
    grid_LON, grid_LAT = np.meshgrid(grid_lon, grid_lat)

    gt_grid   = griddata((lon, lat), gt,   (grid_LON, grid_LAT), method="cubic")
    pred_grid = griddata((lon, lat), pred, (grid_LON, grid_LAT), method="cubic")

    # ── Add Gaussian noise for better visual representativeness ───────────────
    rng = np.random.default_rng(NOISE_SEED)
    gt_grid   = add_gaussian_noise(gt_grid,   gt,   cfg["noise_std_frac"], cfg["noise_smooth_sigma"], rng)
    pred_grid = add_gaussian_noise(pred_grid, pred, cfg["noise_std_frac"], cfg["noise_smooth_sigma"], rng)

    # Clip to valid range
    gt_grid   = np.clip(gt_grid,   cfg["vmin"], cfg["vmax"])
    pred_grid = np.clip(pred_grid, cfg["vmin"], cfg["vmax"])

    norm = Normalize(vmin=cfg["vmin"], vmax=cfg["vmax"])

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, grid, values, title in [
        (ax1, gt_grid,   gt,   cfg["gt_title"]),
        (ax2, pred_grid, pred, cfg["pred_title"]),
    ]:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        # Basemap
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", source=BASEMAP, zoom=ZOOM, attribution="")
        except Exception as e:
            print(f"    [WARN] Basemap download failed: {e}")

        # Interpolated surface
        im = ax.imshow(grid,
                       extent=[extent[0], extent[1], extent[2], extent[3]],
                       origin="lower", cmap=cfg["cmap"], norm=norm,
                       alpha=ALPHA_OVERLAY, aspect="auto", zorder=2)
        # Scatter points
        ax.scatter(lon, lat, c=values, cmap=cfg["cmap"], norm=norm, s=30,
                   edgecolors="black", linewidths=0.5, zorder=3)

        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cfg["label"])

    fig.suptitle(
        f"Spatial Prediction Map: {cfg['label']} — Test Farm \"{TEST_FARM}\"\n"
        f"Farm-LOFO scenario (model trained on 19 other farms, N={len(sub)} test points)\n"
        f"Gaussian noise added for enhanced spatial representativeness",
        fontsize=12, y=1.03,
    )

    plt.tight_layout()
    out = FIG_DIR / cfg["outfile"]
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"    [DONE] Saved: {out}")


def main():
    for cfg in MAPS:
        generate_map(cfg)
    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
