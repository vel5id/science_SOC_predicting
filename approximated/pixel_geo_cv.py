"""
Spatial cross-validation + bootstrap confidence intervals
для Geo-aware Ridge-регрессии (spectral + UTM coords).

Что делает скрипт:
  1. Spatial Leave-One-Field-Out CV (LOFO):
     - Каждый раз одно поле исключается из обучения (test), остальные — train
     - Исключение по field_name гарантирует независимость тестовой выборки
     - Нет spatial leakage: точки одного поля всегда в одном split
  2. Метрики на test (cross-validated, честные):
     - Spearman ρ_cv, RMSE_cv, MAE_cv, R²_cv
  3. Bootstrap CI (n=500) для ρ_cv:
     - Resample OOF predictions с заменой → распределение ρ → CI 95%
  4. Сравнение: in-sample ρ vs cross-validated ρ (показывает оптимизм модели)
  5. Сохраняет:
     - CSV с полными OOF предсказаниями: cv_oof_predictions.csv
     - PNG сводный рисунок: cv_summary.png
     - PNG scatter OOF: cv_scatter_{element}.png
     - Текстовый отчёт: cv_report.txt

Запуск: python approximated/pixel_geo_cv.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pyproj import Transformer
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

# ─── Parameters ──────────────────────────────────────────────────
YEAR   = 2023
FARM   = "Агро Парасат"
N_BOOT = 500       # bootstrap iterations for CI
ALPHA_RIDGE = 10.0 # Ridge regularization (same as pixel_geo_approx.py)
RANDOM_SEED = 42

BASE      = Path(__file__).parent
DATA_PATH = BASE.parent / "data" / "features" / "full_dataset.csv"
OUT_DIR   = BASE.parent / "math_statistics" / "output" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CV_CSV    = OUT_DIR / "cv_oof_predictions.csv"
CV_REPORT = OUT_DIR / "cv_report.txt"

WGS84    = "EPSG:4326"
TIFF_CRS = "EPSG:32641"

# Spectral predictors per element — MUST match pixel_geo_approx.py exactly.
# Format: list of (csv_column, label_str) tuples.
#   csv_column  — column name in full_dataset.csv
#   label_str   — short label for plots/reports
#
# Multiple predictors are combined as [spec1, spec2, ..., nx, ny, nx*ny].
# Opposite-sign correlations do NOT cancel: Ridge assigns β₁<0 and β₂>0 independently.
BEST_PREDICTOR = {
    "ph":  [("s2_NDRE_spring", "NDRE_spr"),  # ρ = -0.616 (dominant)
            ("s2_BSI_spring",  "BSI_spr")],  # ρ = +0.44  (complementary, opposite sign)
    "k":   [("s2_BSI_spring",  "BSI_spr"),   # ρ = -0.478
            ("s2_NDRE_spring", "NDRE_spr")], # ρ = -0.37
    "p":   [("s2_GNDVI_spring","GNDVI_spr"), # ρ = +0.254
            ("s2_BSI_spring",  "BSI_spr")],  # ρ = -0.22  (opposite sign)
    "hu":  [("s2_EVI_summer",  "EVI_sum"),   # ρ = +0.200
            ("s2_NDRE_spring", "NDRE_spr")], # ρ = -0.18
    "s":   [("s2_GNDVI_autumn","GNDVI_aut"), # ρ = +0.323
            ("s2_BSI_spring",  "BSI_spr")],  # ρ = -0.28  (opposite sign)
    "no3": [("s2_GNDVI_spring","GNDVI_spr"), # ρ = -0.298
            ("s2_EVI_summer",  "EVI_sum")],  # ρ = +0.21  (opposite sign)
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

# ─── 1. Загрузка данных ───────────────────────────────────────────
print(f"Loading full_dataset.csv ...")
df_full = pd.read_csv(DATA_PATH)
farm_df = df_full[df_full["farm"] == FARM].copy()
print(f"  Farm '{FARM}': {len(farm_df)} grid points, "
      f"{farm_df['field_name'].nunique()} unique fields")

t_wgs_utm = Transformer.from_crs(WGS84, TIFF_CRS, always_xy=True)

# UTM coords for all farm points
lons = farm_df["centroid_lon"].values
lats = farm_df["centroid_lat"].values
utm_x, utm_y = t_wgs_utm.transform(lons, lats)
farm_df = farm_df.copy()
farm_df["utm_x"] = utm_x
farm_df["utm_y"] = utm_y

# ─── 2. Helper: build feature matrix ─────────────────────────────
def build_X(sub_df, csv_preds, x_min, x_max, y_min, y_max):
    """Normalize UTM and build [spec1, spec2, ..., nx, ny, nx*ny] matrix.

    csv_preds: str or list of str — column name(s) in sub_df.
    Supports multi-predictor input where opposite-sign correlations are combined:
      e.g. NDRE (ρ=-0.616) + BSI (ρ=+0.44) → Ridge finds correct weights automatically.
    """
    ux = sub_df["utm_x"].values
    uy = sub_df["utm_y"].values
    nx = (ux - x_min) / (x_max - x_min + 1e-9)
    ny = (uy - y_min) / (y_max - y_min + 1e-9)
    if isinstance(csv_preds, str):
        csv_preds = [csv_preds]
    spec_cols = [sub_df[c].values for c in csv_preds]
    return np.column_stack(spec_cols + [nx, ny, nx * ny])

# ─── 3. Spatial Leave-One-Field-Out CV ───────────────────────────
print("\nRunning Spatial Leave-One-Field-Out CV ...")
print("  Strategy: each field excluded from training once (test set)")
print(f"  Ridge alpha={ALPHA_RIDGE}, features: [spectral, utm_x, utm_y, utm_x*utm_y]")
print()

all_fields = farm_df["field_name"].unique()
n_fields   = len(all_fields)

# Global UTM normalization range (same as pixel_geo_approx.py)
x_min, x_max = utm_x.min(), utm_x.max()
y_min, y_max = utm_y.min(), utm_y.max()

# Collect OOF results per element
oof_results = {}  # col -> pd.DataFrame with columns [y_true, y_pred, field_name]

for col, preds in BEST_PREDICTOR.items():
    # preds = list of (csv_col, label) tuples
    csv_preds  = [p[0] for p in preds]
    pred_label = "+".join(p[1] for p in preds)

    needed    = ["field_name"] + csv_preds + [col, "utm_x", "utm_y"]
    sub_all   = farm_df[needed].dropna()
    n_total   = len(sub_all)

    if n_total < 30:
        print(f"  Skip {col}: n={n_total} < 30")
        continue

    fields_in_sub = sub_all["field_name"].unique()
    n_fields_sub  = len(fields_in_sub)

    y_true_oof = []
    y_pred_oof = []
    field_oof  = []

    skipped = 0
    for test_field in fields_in_sub:
        train_mask = sub_all["field_name"] != test_field
        test_mask  = sub_all["field_name"] == test_field

        train_df = sub_all[train_mask]
        test_df  = sub_all[test_mask]

        # Need enough training data
        if len(train_df) < 10:
            skipped += 1
            continue

        X_train = build_X(train_df, csv_preds, x_min, x_max, y_min, y_max)
        y_train = train_df[col].values

        X_test  = build_X(test_df,  csv_preds, x_min, x_max, y_min, y_max)
        y_test  = test_df[col].values

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=ALPHA_RIDGE)),
        ])
        pipe.fit(X_train, y_train)
        y_hat = pipe.predict(X_test)

        y_true_oof.extend(y_test.tolist())
        y_pred_oof.extend(y_hat.tolist())
        field_oof.extend([test_field] * len(y_test))

    if len(y_true_oof) < 10:
        print(f"  Skip {col}: not enough OOF predictions")
        continue

    y_true_arr = np.array(y_true_oof)
    y_pred_arr = np.array(y_pred_oof)

    # Cross-validated metrics
    rho_cv, pval_cv = spearmanr(y_true_arr, y_pred_arr)
    rmse_cv = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    mae_cv  = mean_absolute_error(y_true_arr, y_pred_arr)
    r2_cv   = r2_score(y_true_arr, y_pred_arr)

    # In-sample (full model, no hold-out)
    X_full  = build_X(sub_all, csv_preds, x_min, x_max, y_min, y_max)
    y_full  = sub_all[col].values
    pipe_full = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=ALPHA_RIDGE))])
    pipe_full.fit(X_full, y_full)
    y_pred_full = pipe_full.predict(X_full)
    rho_train, _ = spearmanr(y_pred_full, y_full)

    # Bootstrap CI for ρ_cv (resample OOF pairs with replacement)
    rng = np.random.default_rng(RANDOM_SEED)
    boot_rhos = []
    n_oof = len(y_true_arr)
    for _ in range(N_BOOT):
        idx = rng.choice(n_oof, n_oof, replace=True)
        r_b, _ = spearmanr(y_true_arr[idx], y_pred_arr[idx])
        boot_rhos.append(r_b)
    boot_rhos = np.array(boot_rhos)
    ci_lo = float(np.percentile(boot_rhos, 2.5))
    ci_hi = float(np.percentile(boot_rhos, 97.5))

    oof_results[col] = {
        "y_true":     y_true_arr,
        "y_pred":     y_pred_arr,
        "field":      field_oof,
        "rho_train":  rho_train,
        "rho_cv":     rho_cv,
        "pval_cv":    pval_cv,
        "rmse_cv":    rmse_cv,
        "mae_cv":     mae_cv,
        "r2_cv":      r2_cv,
        "ci_lo":      ci_lo,
        "ci_hi":      ci_hi,
        "n_oof":      len(y_true_arr),
        "n_fields":   n_fields_sub,
        "skipped":    skipped,
        "boot_rhos":  boot_rhos,
        "idx_label":  pred_label,
    }

    optimism = rho_train - rho_cv
    print(f"  {col:4s} ({CHEM_LABELS[col]:12s})  [{pred_label}]: "
          f"ρ_train={rho_train:+.3f}  ρ_cv={rho_cv:+.3f}  "
          f"[CI95%: {ci_lo:+.3f}, {ci_hi:+.3f}]  "
          f"RMSE={rmse_cv:.3f}  R²={r2_cv:+.3f}  "
          f"optimism={optimism:+.3f}  n_oof={len(y_true_arr)}  fields={n_fields_sub}")

# ─── 4. Сохранение OOF predictions CSV ───────────────────────────
print(f"\nSaving OOF predictions to {CV_CSV.name} ...")
oof_rows = []
for col, res in oof_results.items():
    for yt, yp, fn in zip(res["y_true"], res["y_pred"], res["field"]):
        oof_rows.append({
            "element":    col,
            "field_name": fn,
            "y_true":     yt,
            "y_pred":     yp,
            "residual":   yt - yp,
        })
oof_df = pd.DataFrame(oof_rows)
oof_df.to_csv(CV_CSV, index=False, float_format="%.6f")
print(f"  Saved {len(oof_df):,} rows")

# ─── 5. Текстовый отчёт ──────────────────────────────────────────
report_lines = [
    "=" * 80,
    f"SPATIAL LEAVE-ONE-FIELD-OUT CV REPORT  (multi-predictor Ridge)",
    f"Farm: {FARM}, Year: {YEAR}",
    f"Model: Ridge(alpha={ALPHA_RIDGE})",
    f"Features: [spec1, spec2, ..., utm_x_norm, utm_y_norm, utm_x·utm_y]",
    f"  (predictors per element shown in table below; opposite-sign ρ are valid — Ridge weights them independently)",
    f"CV strategy: Leave-One-Field-Out (LOFO) — each field held out once",
    f"Bootstrap CI: n={N_BOOT} iterations, 95% percentile interval",
    "=" * 80,
    "",
    f"{'Element':<8} {'Label':<14} {'ρ_train':>8} {'ρ_cv':>8} "
    f"{'CI_lo':>8} {'CI_hi':>8} {'RMSE':>8} {'MAE':>7} {'R²':>7} "
    f"{'Δρ':>7} {'n_oof':>6} {'fields':>6}",
    "-" * 96,
]
for col, res in oof_results.items():
    delta = res["rho_train"] - res["rho_cv"]
    report_lines.append(
        f"{col:<8} {CHEM_LABELS[col]:<14} "
        f"{res['rho_train']:>+8.3f} {res['rho_cv']:>+8.3f} "
        f"{res['ci_lo']:>+8.3f} {res['ci_hi']:>+8.3f} "
        f"{res['rmse_cv']:>8.3f} {res['mae_cv']:>7.3f} {res['r2_cv']:>+7.3f} "
        f"{delta:>+7.3f} {res['n_oof']:>6} {res['n_fields']:>6}"
    )
report_lines += [
    "-" * 96,
    "",
    "Notes:",
    "  ρ_train  = Spearman correlation on FULL training set (in-sample, optimistic)",
    "  ρ_cv     = Spearman correlation on OOF predictions (cross-validated, honest)",
    "  Δρ       = ρ_train - ρ_cv  (model optimism; positive = overfitting signal)",
    "  CI 95%   = Bootstrap percentile interval for ρ_cv",
    "  RMSE/MAE = Out-of-fold errors in original units",
    "  R²       = OOF coefficient of determination (can be negative if worse than mean)",
    "  n_oof    = Total number of out-of-fold predictions",
    "  fields   = Number of unique fields in CV (each held out once)",
    "",
    "Interpretation:",
    "  |ρ_cv| > 0.5  → strong cross-validated signal",
    "  |ρ_cv| 0.3-0.5 → moderate signal",
    "  |ρ_cv| < 0.3  → weak signal (map is mostly model extrapolation)",
    "  Δρ > 0.1 indicates notable overfitting",
    "",
]
report_text = "\n".join(report_lines)
print("\n" + report_text)
with open(CV_REPORT, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"Report saved: {CV_REPORT.name}")

# ─── 6. Summary figure: ρ_train vs ρ_cv + CI ────────────────────
print("\nRendering summary figure ...")

n_elements = len(oof_results)
cols_order  = [c for c in CHEM_LABELS if c in oof_results]

fig = plt.figure(figsize=(16, 5 + 3.5 * n_elements), facecolor="#0a0a0a")
gs  = gridspec.GridSpec(
    n_elements + 1, 2,
    height_ratios=[2.5] + [3.0] * n_elements,
    hspace=0.55, wspace=0.38,
    left=0.10, right=0.95, top=0.94, bottom=0.04,
)

# ── Panel 0: ρ comparison bar chart ──────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.set_facecolor("#111111")

labels   = [CHEM_LABELS[c] for c in cols_order]
rho_tr   = [oof_results[c]["rho_train"] for c in cols_order]
rho_cv_v = [oof_results[c]["rho_cv"]    for c in cols_order]
ci_lo_v  = [oof_results[c]["ci_lo"]     for c in cols_order]
ci_hi_v  = [oof_results[c]["ci_hi"]     for c in cols_order]

x = np.arange(n_elements)
w = 0.35

bars_tr = ax0.bar(x - w/2, rho_tr,   width=w, label="ρ_train (in-sample)",
                  color="#4a9eda", alpha=0.85, zorder=3)
bars_cv = ax0.bar(x + w/2, rho_cv_v, width=w, label="ρ_cv (LOFO, honest)",
                  color="#e05c4a", alpha=0.85, zorder=3)

# Bootstrap CI error bars on ρ_cv bars
err_lo = [rho_cv_v[i] - ci_lo_v[i] for i in range(n_elements)]
err_hi = [ci_hi_v[i]  - rho_cv_v[i] for i in range(n_elements)]
ax0.errorbar(x + w/2, rho_cv_v,
             yerr=[err_lo, err_hi],
             fmt="none", color="white", capsize=4, linewidth=1.2, zorder=4)

# Zero line
ax0.axhline(0, color="#555555", linewidth=0.8, linestyle="--")

ax0.set_xticks(x)
ax0.set_xticklabels(labels, color="white", fontsize=9)
ax0.set_ylabel("Spearman ρ", color="white", fontsize=9)
ax0.tick_params(colors="white")
ax0.set_facecolor("#111111")
for spine in ax0.spines.values():
    spine.set_edgecolor("#333333")
ax0.set_title(
    f"In-sample vs Cross-validated Spearman ρ — Geo-aware Ridge  |  "
    f"LOFO-CV  |  Bootstrap CI 95%  |  {FARM}, {YEAR}",
    color="white", fontsize=10, fontweight="bold", pad=8,
)
ax0.legend(facecolor="#1a1a1a", edgecolor="#444444", labelcolor="white", fontsize=8)
ax0.set_ylim(-1.05, 1.05)

# Annotate bars with values
for i, (rt, rc) in enumerate(zip(rho_tr, rho_cv_v)):
    ax0.text(i - w/2, rt + 0.04 * np.sign(rt), f"{rt:+.2f}",
             ha="center", va="bottom" if rt >= 0 else "top",
             color="#4a9eda", fontsize=7, fontweight="bold")
    ax0.text(i + w/2, rc + 0.04 * np.sign(rc), f"{rc:+.2f}",
             ha="center", va="bottom" if rc >= 0 else "top",
             color="#e05c4a", fontsize=7, fontweight="bold")

# ── Panels 1..n: OOF scatter per element ─────────────────────────
for row_i, col in enumerate(cols_order):
    res    = oof_results[col]
    label  = CHEM_LABELS[col]
    y_true = res["y_true"]
    y_pred = res["y_pred"]
    rho_cv = res["rho_cv"]
    rmse   = res["rmse_cv"]
    r2     = res["r2_cv"]
    ci_lo  = res["ci_lo"]
    ci_hi  = res["ci_hi"]
    boot   = res["boot_rhos"]

    # Left panel: OOF scatter
    ax_sc = fig.add_subplot(gs[row_i + 1, 0])
    ax_sc.set_facecolor("#111111")

    vmin_, vmax_ = np.percentile(y_true, 2), np.percentile(y_true, 98)
    sc = ax_sc.scatter(y_true, y_pred, c=y_true,
                       cmap=CHEM_CMAPS[col], vmin=vmin_, vmax=vmax_,
                       s=18, alpha=0.65, linewidths=0, zorder=3)
    # 1:1 line
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax_sc.plot([lo, hi], [lo, hi], color="#888888", linewidth=1.0,
               linestyle="--", zorder=2, label="1:1 line")
    # Regression line through OOF
    z  = np.polyfit(y_true, y_pred, 1)
    xr = np.linspace(lo, hi, 100)
    ax_sc.plot(xr, np.polyval(z, xr), color="#e05c4a", linewidth=1.2,
               zorder=4, label="OOF trend")

    ax_sc.set_xlabel(f"Observed {label}", color="white", fontsize=8)
    ax_sc.set_ylabel(f"Predicted {label}", color="white", fontsize=8)
    ax_sc.tick_params(colors="white", labelsize=7)
    for spine in ax_sc.spines.values():
        spine.set_edgecolor("#333333")
    ax_sc.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="white", fontsize=6.5)

    txt = (f"ρ_cv = {rho_cv:+.3f}  [95%: {ci_lo:+.3f}, {ci_hi:+.3f}]\n"
           f"RMSE = {rmse:.3f}  R² = {r2:+.3f}  n = {res['n_oof']}")
    ax_sc.set_title(f"{label}  —  OOF scatter (LOFO-CV)", color="white",
                    fontsize=8.5, fontweight="bold")
    ax_sc.text(0.03, 0.97, txt, transform=ax_sc.transAxes,
               ha="left", va="top", color="white", fontsize=7,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                         edgecolor="#555", alpha=0.88))

    cb = fig.colorbar(sc, ax=ax_sc, pad=0.02, shrink=0.85)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
    cb.outline.set_edgecolor("#333")
    cb.set_label(label, color="white", fontsize=6)

    # Right panel: bootstrap distribution of ρ_cv
    ax_bs = fig.add_subplot(gs[row_i + 1, 1])
    ax_bs.set_facecolor("#111111")

    ax_bs.hist(boot, bins=40, color="#e05c4a", alpha=0.75, edgecolor="none", zorder=3)
    ax_bs.axvline(rho_cv, color="white",   linewidth=1.5, linestyle="-",  zorder=4,
                  label=f"ρ_cv = {rho_cv:+.3f}")
    ax_bs.axvline(ci_lo,  color="#4a9eda", linewidth=1.2, linestyle="--", zorder=4,
                  label=f"CI 2.5% = {ci_lo:+.3f}")
    ax_bs.axvline(ci_hi,  color="#4a9eda", linewidth=1.2, linestyle="--", zorder=4,
                  label=f"CI 97.5% = {ci_hi:+.3f}")
    ax_bs.axvline(res["rho_train"], color="#ffc04a", linewidth=1.2,
                  linestyle=":", zorder=4,
                  label=f"ρ_train = {res['rho_train']:+.3f}")

    ax_bs.set_xlabel("Spearman ρ", color="white", fontsize=8)
    ax_bs.set_ylabel("Bootstrap count", color="white", fontsize=8)
    ax_bs.tick_params(colors="white", labelsize=7)
    for spine in ax_bs.spines.values():
        spine.set_edgecolor("#333333")
    ax_bs.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="white",
                 fontsize=6.5, loc="upper left")
    ax_bs.set_title(f"{label}  —  Bootstrap distribution of ρ_cv  (n={N_BOOT})",
                    color="white", fontsize=8.5, fontweight="bold")

fig.suptitle(
    f"Spatial LOFO-CV — Geo-aware Ridge(spectral+UTM)  |  {FARM}, {YEAR}  |  "
    f"field-level trained, pixel-level applied  |  honest out-of-fold evaluation",
    fontsize=11, color="white", y=0.995,
)

out_cv = OUT_DIR / f"cv_summary_{FARM.replace(' ', '_')}_{YEAR}.png"
fig.savefig(out_cv, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig)
print(f"  Saved: {out_cv.name}")

# ─── 7. Individual OOF scatter per element (high-res) ────────────
print("\nRendering individual OOF scatter plots ...")
for col, res in oof_results.items():
    label  = CHEM_LABELS[col]
    y_true = res["y_true"]
    y_pred = res["y_pred"]
    rho_cv = res["rho_cv"]
    rmse   = res["rmse_cv"]
    r2     = res["r2_cv"]
    ci_lo  = res["ci_lo"]
    ci_hi  = res["ci_hi"]

    fig2, ax = plt.subplots(figsize=(6, 6))
    fig2.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#111111")

    vmin_, vmax_ = np.percentile(y_true, 2), np.percentile(y_true, 98)
    sc = ax.scatter(y_true, y_pred, c=y_true,
                    cmap=CHEM_CMAPS[col], vmin=vmin_, vmax=vmax_,
                    s=25, alpha=0.7, linewidths=0, zorder=3)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="#888888", linewidth=1.2,
            linestyle="--", zorder=2, label="1:1 line")
    z  = np.polyfit(y_true, y_pred, 1)
    xr = np.linspace(lo, hi, 200)
    ax.plot(xr, np.polyval(z, xr), color="#e05c4a", linewidth=1.5,
            zorder=4, label="OOF trend")

    cb = fig2.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(label, color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)
    cb.outline.set_edgecolor("#333")

    ax.set_xlabel(f"Observed {label}", color="white", fontsize=10)
    ax.set_ylabel(f"Predicted {label}  (OOF)", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="white", fontsize=8)

    ax.set_title(
        f"{label}  —  Spatial LOFO-CV OOF scatter",
        color="white", fontsize=11, fontweight="bold", pad=8,
    )
    txt = (
        f"ρ_cv = {rho_cv:+.3f}  [95% CI: {ci_lo:+.3f}, {ci_hi:+.3f}]\n"
        f"RMSE = {rmse:.3f}    R² = {r2:+.3f}\n"
        f"n_oof = {res['n_oof']}  fields = {res['n_fields']}"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            ha="left", va="top", color="white", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#1a1a1a",
                      edgecolor="#666", alpha=0.90))

    fig2.text(
        0.5, 0.01,
        f"{FARM}  |  Ridge(spectral+UTM)  |  LOFO-CV  |  "
        f"Bootstrap CI n={N_BOOT}  |  field-level trained, pixel-level applied",
        ha="center", va="bottom", color="#666666", fontsize=7,
    )
    fig2.tight_layout(pad=0.4, rect=[0, 0.03, 1, 1])

    out_sc = OUT_DIR / f"cv_scatter_{FARM.replace(' ', '_')}_{YEAR}_{col}.png"
    fig2.savefig(out_sc, dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig2)
    print(f"  Saved: {out_sc.name}")

print(f"\nDone! Output directory: {OUT_DIR}")
print(f"Files generated:")
print(f"  cv_summary_*.png         — summary: bar + scatter + bootstrap")
print(f"  cv_scatter_*_{{element}}.png — individual OOF scatter per element")
print(f"  cv_oof_predictions.csv   — OOF predictions table")
print(f"  cv_report.txt            — text report with all metrics")
