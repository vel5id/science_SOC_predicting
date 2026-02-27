"""generate_four_panel_map.py
----------------------------
Generate a 4-panel spatial prediction comparison figure for pH:
  (a) Ground Truth (lab measurements, interpolated)
  (b) RF Field-LOFO (81 folds) OOF predictions
  (c) RF Farm-LOFO (20 folds, per-fold MDI + GridSearchCV) OOF predictions
  (d) CNN Field-LOFO (81 folds) OOF predictions

All panels share:
  - Satellite basemap (Esri World Imagery via contextily)
  - Same colour scale (pH 5.5—8.0)
  - Interpolated surface from point predictions
  - Scatter points with measured/predicted values
  - Test farm: "Агро Парасат" (lon ≈ 65.3–65.4, lat ≈ 53.3–53.4)

Output:
  articles/article2_prediction/figures/fig_four_panel_comparison.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from pathlib import Path
import contextily as ctx
import warnings
warnings.filterwarnings("ignore")

# ─── Config ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
FIG_DIR  = ROOT / "articles" / "article2_prediction" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TEST_FARM     = "Агро Парасат"
TARGET        = "ph"
ALPHA_OVERLAY = 0.65
GRID_RES      = 200
PAD_FRAC      = 0.10
NOISE_SEED    = 42
NOISE_STD     = 0.04      # fraction of data std
SMOOTH_SIGMA  = 2.0

CMAP  = "RdYlGn"
VMIN  = 5.5
VMAX  = 8.0
LABEL = "pH (KCl)"

BASEMAP = ctx.providers.Esri.WorldImagery
ZOOM    = 14

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ─── Data sources ─────────────────────────────────────────────────────────────
# RF Field-LOFO: ML/results/rf/ph_oof_predictions.csv  (81 folds)
# RF Farm-LOFO:  ML/results/rf_gridsearch_lofo/ph_oof_predictions.csv (20 folds)
# CNN Field-LOFO: ML/results/cnn/ph_oof_predictions.csv (81 folds)

PANELS = [
    {
        "title": "(a) Ground Truth\n(lab measurements)",
        "csv":   ROOT / "ML" / "results" / "rf" / "ph_oof_predictions.csv",
        "pred_col": None,  # use ground truth only
    },
    {
        "title": "(b) RF — Field-LOFO\n(81 folds, ρ = 0.798)",
        "csv":   ROOT / "ML" / "results" / "rf" / "ph_oof_predictions.csv",
        "pred_col": "oof_pred",
    },
    {
        "title": "(c) RF — Farm-LOFO\n(20 folds, ρ = 0.403)",
        "csv":   ROOT / "ML" / "results" / "rf_gridsearch_lofo" / "ph_oof_predictions.csv",
        "pred_col": "oof_pred",
    },
    {
        "title": "(d) CNN — Field-LOFO\n(81 folds, ρ = 0.699)",
        "csv":   ROOT / "ML" / "results" / "cnn" / "ph_oof_predictions.csv",
        "pred_col": "oof_pred",
    },
]


def add_gaussian_noise(grid, data_values, rng):
    sigma = np.nanstd(data_values) * NOISE_STD
    noise = rng.normal(0, sigma, grid.shape)
    noise = gaussian_filter(noise, sigma=SMOOTH_SIGMA)
    result = grid + noise
    result[np.isnan(grid)] = np.nan
    return result


def load_farm_data(csv_path, pred_col=None):
    """Load data for TEST_FARM, return lon, lat, gt, pred arrays."""
    cols_to_load = ["farm", "centroid_lon", "centroid_lat", TARGET]
    if pred_col:
        cols_to_load.append(pred_col)
    
    df = pd.read_csv(csv_path, usecols=cols_to_load, low_memory=False)
    sub = df[df["farm"] == TEST_FARM].copy().reset_index(drop=True)
    
    lon = sub["centroid_lon"].values
    lat = sub["centroid_lat"].values
    gt  = sub[TARGET].values
    pred = sub[pred_col].values if pred_col else gt
    
    return lon, lat, gt, pred


def make_grid(lon, lat, values, rng):
    """Interpolate values onto a regular grid with Gaussian noise."""
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    dlon = (lon_max - lon_min) * PAD_FRAC
    dlat = (lat_max - lat_min) * PAD_FRAC
    extent = [lon_min - dlon, lon_max + dlon, lat_min - dlat, lat_max + dlat]
    
    grid_lon = np.linspace(extent[0], extent[1], GRID_RES)
    grid_lat = np.linspace(extent[2], extent[3], GRID_RES)
    grid_LON, grid_LAT = np.meshgrid(grid_lon, grid_lat)
    
    grid = griddata((lon, lat), values, (grid_LON, grid_LAT), method="cubic")
    grid = add_gaussian_noise(grid, values, rng)
    grid = np.clip(grid, VMIN, VMAX)
    
    return grid, extent


def render_panel(ax, lon, lat, values, grid, extent, title, norm, is_placeholder=False):
    """Render one panel: basemap + interpolated surface + scatter."""
    if is_placeholder:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", source=BASEMAP, zoom=ZOOM, attribution="")
        except Exception:
            pass
        # Dark overlay
        ax.imshow(np.zeros((10, 10, 4)),
                  extent=[extent[0], extent[1], extent[2], extent[3]],
                  origin="lower", alpha=0.6, aspect="auto", zorder=2)
        # N/A text
        cx = (extent[0] + extent[1]) / 2
        cy = (extent[2] + extent[3]) / 2
        ax.text(cx, cy, "Per-point\npredictions\nnot saved",
                fontsize=12, fontweight="bold", color="white",
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="red",
                          edgecolor="white", alpha=0.8, linewidth=2))
        ax.text(cx, cy - (extent[3] - extent[2]) * 0.30,
                "R² < 0 for 5/6 properties\n(ResNet-18, 20 folds)",
                fontsize=8, color="white", ha="center", va="center", zorder=5,
                fontstyle="italic")
        ax.set_title(title, fontweight="bold", color="red")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        return
    
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    
    # Basemap
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=BASEMAP, zoom=ZOOM, attribution="")
    except Exception as e:
        print(f"    [WARN] Basemap failed: {e}")
    
    # Interpolated surface
    im = ax.imshow(grid,
                   extent=[extent[0], extent[1], extent[2], extent[3]],
                   origin="lower", cmap=CMAP, norm=norm,
                   alpha=ALPHA_OVERLAY, aspect="auto", zorder=2)
    
    # Scatter points
    ax.scatter(lon, lat, c=values, cmap=CMAP, norm=norm, s=20,
               edgecolors="black", linewidths=0.4, zorder=3)
    
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    return im


def main():
    print("[INFO] Generating 4-panel comparison figure for pH...")
    
    norm = Normalize(vmin=VMIN, vmax=VMAX)
    rng = np.random.default_rng(NOISE_SEED)
    
    # Load all available data first
    data = {}
    ref_extent = None
    
    for i, panel in enumerate(PANELS):
        if panel["csv"] is None:
            data[i] = None
            continue
        
        lon, lat, gt, pred = load_farm_data(panel["csv"], panel["pred_col"])
        values = pred if panel["pred_col"] else gt
        grid, extent = make_grid(lon, lat, values, rng)
        data[i] = {"lon": lon, "lat": lat, "values": values, "grid": grid, "extent": extent}
        
        if ref_extent is None:
            ref_extent = extent
        
        print(f"    Panel {i}: {panel['title'].split(chr(10))[0]} — {len(lon)} points, "
              f"values [{values.min():.2f}, {values.max():.2f}]")
    
    # Create figure: 1 row × 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    
    last_im = None
    for i, (ax, panel) in enumerate(zip(axes, PANELS)):
        if data[i] is None:
            # Placeholder panel
            render_panel(ax, None, None, None, None, ref_extent,
                         panel["title"], norm, is_placeholder=True)
        else:
            d = data[i]
            im = render_panel(ax, d["lon"], d["lat"], d["values"],
                              d["grid"], d["extent"], panel["title"], norm)
            if im is not None:
                last_im = im
        
        # Only show y-label on first panel
        if i > 0:
            ax.set_ylabel("")
    
    # Single shared colorbar
    cbar = fig.colorbar(last_im, ax=axes.tolist(), fraction=0.012, pad=0.02,
                        label=LABEL, shrink=0.85)
    
    fig.suptitle(
        f"Spatial prediction maps for pH (KCl) — test farm \"{TEST_FARM}\" "
        f"(N = {len(data[0]['lon'])} points)\n"
        f"Basemap: Esri World Imagery | Interpolation: cubic + Gaussian noise "
        f"(σ ≈ {NOISE_STD*100:.0f}% of std)",
        fontsize=11, y=1.02
    )
    
    plt.tight_layout()
    
    out = FIG_DIR / "fig_four_panel_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"\n[DONE] Saved: {out}")


if __name__ == "__main__":
    main()
