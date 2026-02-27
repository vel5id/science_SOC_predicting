"""
rf_grid_search.py
=================
Grid search over (n_features × n_estimators) for all 6 soil targets.

Grid:
  n_features   : [10, 15, 20, 25, 30, 35]  (6 levels, step=5, from ranked importance list)
  n_estimators : [100, 150, 200, 250]       (4 levels, step=50)
  → 24 unique model configs per element
  → 144 total model fits (6 elements × 24 configs)

Train / Validation / Test split (spatial):
  Test  : 20% of fields (held-out, untouched during selection)
  Valid : 16% of fields (used for final model ranking)
  Train : 64% of fields (used for all model fitting)

  Field-level split ensures NO data leakage between spatial units.
  Fields sorted by name → deterministic split.

Outputs:
  rf_grid_results.csv        — full grid with all metrics per (element, k, n_est)
  rf_grid_best.csv           — best config per element (by val-ρ)
  rf_grid_heatmap.png        — ρ heatmap: k × n_estimators for each element
  rf_grid_test_summary.png   — test-set bar chart: best model vs all configs
  rf_grid_test_report.txt    — final test-set metrics for best models

Run: python approximated/rf_grid_search.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")
import pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
RF_DATA    = BASE / "data" / "features" / "rf_dataset.csv"
MODELS_DIR = BASE / "math_statistics" / "output" / "rf" / "rf_models"
OUT_RF     = BASE / "math_statistics" / "output" / "rf"
OUT_PLOTS  = BASE / "math_statistics" / "output" / "plots"
OUT_RF.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

# ─── Grid config ──────────────────────────────────────────────────────────────
N_FEATURES_GRID   = [10, 15, 20, 25, 30, 35]   # 6 feature levels
N_ESTIMATORS_GRID = [100, 150, 200, 250]        # 4 tree levels
# → 6 × 4 = 24 unique configs

TARGETS     = ["ph", "k", "p", "hu", "s", "no3"]
LOG_TARGETS = {"p", "s", "no3"}
TARGET_LABELS = {"ph":"pH", "k":"K mg/kg", "p":"P mg/kg",
                 "hu":"Humus %", "s":"S mg/kg", "no3":"NO₃ mg/kg"}

# ─── Split config ─────────────────────────────────────────────────────────────
TEST_FRAC  = 0.20   # 20% fields → test  (held-out completely)
VAL_FRAC   = 0.16   # 16% fields → validation (model selection)
TRAIN_FRAC = 0.64   # 64% fields → training

RANDOM_STATE = 42
RF_SEED      = 42
RF_TREE_MAX_FEAT = "sqrt"
RF_MIN_SAMPLES_LEAF = 3

print("=" * 70)
print("rf_grid_search.py  —  Grid Search: 6 feat-levels × 4 tree-levels")
print("=" * 70)
print(f"\nGrid:")
print(f"  n_features   : {N_FEATURES_GRID}  ({len(N_FEATURES_GRID)} levels)")
print(f"  n_estimators : {N_ESTIMATORS_GRID}  ({len(N_ESTIMATORS_GRID)} levels)")
print(f"  Total configs per element: {len(N_FEATURES_GRID) * len(N_ESTIMATORS_GRID)}")
print(f"  Total model fits: {len(N_FEATURES_GRID) * len(N_ESTIMATORS_GRID) * len(TARGETS)}")
print(f"\nSplit: {int(TRAIN_FRAC*100)}% train | {int(VAL_FRAC*100)}% val | {int(TEST_FRAC*100)}% test (field-level)")

# ─── Load data ────────────────────────────────────────────────────────────────
print("\nLoading rf_dataset.csv ...")
rf_df = pd.read_csv(RF_DATA)
print(f"  Shape: {rf_df.shape}")

# Global field list (all unique fields across all rows)
all_fields = sorted(rf_df["field_name"].unique())
n_fields   = len(all_fields)
print(f"  Unique fields: {n_fields}")

# ─── Deterministic field-level split ─────────────────────────────────────────
rng = np.random.default_rng(RANDOM_STATE)
field_arr   = np.array(all_fields)
shuffled    = rng.permutation(field_arr)

n_test  = max(1, int(np.ceil(n_fields * TEST_FRAC)))
n_val   = max(1, int(np.ceil(n_fields * VAL_FRAC)))
n_train = n_fields - n_test - n_val

test_fields  = set(shuffled[:n_test])
val_fields   = set(shuffled[n_test : n_test + n_val])
train_fields = set(shuffled[n_test + n_val:])

print(f"\nField split:")
print(f"  Train: {len(train_fields)} fields  ({len(train_fields)/n_fields*100:.0f}%)")
print(f"  Val:   {len(val_fields)} fields  ({len(val_fields)/n_fields*100:.0f}%)")
print(f"  Test:  {len(test_fields)} fields  ({len(test_fields)/n_fields*100:.0f}%)")
print(f"  [Test fields (held-out): {sorted(test_fields)[:5]} ...]")

# ─── Grid search loop ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  GRID SEARCH")
print("=" * 70)

all_results  = []   # rows: element, k, n_est, rho_train, rho_val, rho_test, rmse_val, ...
best_models  = {}   # element → bundle of best model

t_global_start = time.time()

for tgt in TARGETS:
    print(f"\n  ── {tgt.upper()} ({TARGET_LABELS[tgt]}) ─────────────────────────")

    # Load pre-ranked feature list from stage-2 RF training
    with open(MODELS_DIR / f"rf_{tgt}.pkl", "rb") as f:
        bundle = pickle.load(f)

    ranked_features = bundle["features"]   # 40 features sorted by perm importance
    log_transform   = bundle["log_transform"]

    # Select data rows for this target
    mask_col = f"mask_{tgt}"
    if mask_col in rf_df.columns:
        valid_mask = rf_df[mask_col].astype(bool)
    else:
        valid_mask = rf_df[tgt].notna()

    df_valid = rf_df[valid_mask].copy()

    tgt_col = f"log_{tgt}" if log_transform and f"log_{tgt}" in df_valid.columns else tgt
    y_all   = df_valid[tgt_col].values.astype(float)
    fids    = df_valid["field_name"].values

    # Row masks per split
    tr_mask = np.array([f in train_fields for f in fids])
    va_mask = np.array([f in val_fields   for f in fids])
    te_mask = np.array([f in test_fields  for f in fids])

    print(f"    n_valid={len(df_valid)} | train={tr_mask.sum()} val={va_mask.sum()} test={te_mask.sum()}")
    print(f"    target={tgt_col}  log={log_transform}")

    best_val_rho = -np.inf
    best_cfg     = None

    for k in N_FEATURES_GRID:
        feats_k  = ranked_features[:k]
        X_all_k  = df_valid[feats_k].values.astype(float)

        X_tr = X_all_k[tr_mask]
        X_va = X_all_k[va_mask]
        X_te = X_all_k[te_mask]
        y_tr = y_all[tr_mask]
        y_va = y_all[va_mask]
        y_te = y_all[te_mask]

        # Imputer fit on train only (no leakage)
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr)
        X_va_imp = imp.transform(X_va)
        X_te_imp = imp.transform(X_te)

        for n_est in N_ESTIMATORS_GRID:
            t0 = time.time()
            rf = RandomForestRegressor(
                n_estimators=n_est,
                max_features=RF_TREE_MAX_FEAT,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                oob_score=True,
                random_state=RF_SEED,
                n_jobs=-1,
            )
            rf.fit(X_tr_imp, y_tr)
            elapsed = time.time() - t0

            # --- Predictions ---
            y_hat_tr = rf.predict(X_tr_imp)
            y_hat_va = rf.predict(X_va_imp)
            y_hat_te = rf.predict(X_te_imp) if te_mask.sum() > 0 else np.array([])

            # Back-transform if log
            def bt(arr):
                return np.expm1(arr) if log_transform else arr

            # Spearman on log-scale (training objective), RMSE on original scale
            rho_tr, _ = spearmanr(y_tr, y_hat_tr)
            rho_va = spearmanr(y_va, y_hat_va)[0] if len(y_va) > 1 else np.nan
            rho_te = spearmanr(y_te, y_hat_te)[0] if len(y_te) > 1 else np.nan

            rmse_va = float(np.sqrt(mean_squared_error(bt(y_va), bt(y_hat_va)))) \
                      if len(y_va) > 1 else np.nan
            rmse_te = float(np.sqrt(mean_squared_error(bt(y_te), bt(y_hat_te)))) \
                      if len(y_te) > 1 else np.nan

            oob_r2 = rf.oob_score_

            row = {
                "element":   tgt,
                "k":         k,
                "n_est":     n_est,
                "rho_train": round(rho_tr, 4),
                "rho_val":   round(rho_va, 4),
                "rho_test":  round(rho_te, 4),
                "rmse_val":  round(rmse_va, 4),
                "rmse_test": round(rmse_te, 4),
                "oob_r2":    round(oob_r2,  4),
                "fit_sec":   round(elapsed, 2),
                "log_transform": log_transform,
            }
            all_results.append(row)

            # Track best by val-ρ
            if rho_va > best_val_rho:
                best_val_rho = rho_va
                best_cfg = {
                    "rf": rf, "imputer": imp,
                    "features": feats_k, "k": k, "n_est": n_est,
                    "rho_train": rho_tr, "rho_val": rho_va, "rho_test": rho_te,
                    "rmse_val": rmse_va, "rmse_test": rmse_te,
                    "oob_r2": oob_r2, "log_transform": log_transform,
                    "tgt": tgt, "tgt_col": tgt_col,
                    "y_te": y_te, "y_hat_te": y_hat_te,
                }

            marker = " ← best-val" if rho_va == best_val_rho else ""
            print(f"    k={k:2d} n_est={n_est:3d} | "
                  f"ρ_tr={rho_tr:+.4f} ρ_val={rho_va:+.4f} ρ_te={rho_te:+.4f} "
                  f"RMSE_val={rmse_va:.3f} OOB={oob_r2:.3f} ({elapsed:.1f}s){marker}")

    best_models[tgt] = best_cfg
    print(f"\n    → BEST for {tgt}: k={best_cfg['k']}  n_est={best_cfg['n_est']}"
          f"  ρ_val={best_cfg['rho_val']:+.4f}  ρ_test={best_cfg['rho_test']:+.4f}")

t_total = time.time() - t_global_start
print(f"\nTotal grid search time: {t_total:.1f}s")

# ─── Save results CSV ─────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_RF / "rf_grid_results.csv", index=False)
print(f"\nSaved: rf_grid_results.csv  ({len(results_df)} rows)")

# Best per element
best_rows = []
for tgt in TARGETS:
    m = best_models[tgt]
    best_rows.append({
        "element":    tgt,
        "best_k":     m["k"],
        "best_n_est": m["n_est"],
        "rho_train":  round(m["rho_train"], 4),
        "rho_val":    round(m["rho_val"],   4),
        "rho_test":   round(m["rho_test"],  4),
        "rmse_val":   round(m["rmse_val"],  4),
        "rmse_test":  round(m["rmse_test"], 4),
        "oob_r2":     round(m["oob_r2"],    4),
        "log_transform": m["log_transform"],
        "top_features": ", ".join(m["features"][:5]),
    })
best_df = pd.DataFrame(best_rows)
best_df.to_csv(OUT_RF / "rf_grid_best.csv", index=False)
print(f"Saved: rf_grid_best.csv")

# ─── Heatmap: ρ_val for each element (k × n_est) ─────────────────────────────
print("\nRendering ρ_val heatmaps ...")
DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
TEXT_CLR = "#e0e0e0"

fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor=DARK_BG)
fig.suptitle(f"Grid Search: ρ_val  (Spearman, validation set)\n"
             f"Grid: {N_FEATURES_GRID} features × {N_ESTIMATORS_GRID} trees  |  "
             f"Split: {int(TRAIN_FRAC*100)}% train / {int(VAL_FRAC*100)}% val / {int(TEST_FRAC*100)}% test",
             color=TEXT_CLR, fontsize=12, fontweight="bold", y=1.01)

for ax_i, tgt in enumerate(TARGETS):
    ax = axes[ax_i // 3][ax_i % 3]
    ax.set_facecolor(PANEL_BG)

    sub = results_df[results_df["element"] == tgt]
    pivot_val  = sub.pivot(index="k", columns="n_est", values="rho_val")
    pivot_test = sub.pivot(index="k", columns="n_est", values="rho_test")

    # Heatmap of rho_val
    vmin = pivot_val.values.min()
    vmax = pivot_val.values.max()
    im = ax.imshow(pivot_val.values, aspect="auto", cmap="YlOrRd",
                   vmin=vmin, vmax=vmax, origin="upper")

    # Annotate cells: rho_val (top) + rho_test (bottom, smaller)
    for ri, k in enumerate(pivot_val.index):
        for ci, n in enumerate(pivot_val.columns):
            rv = pivot_val.loc[k, n]
            rt = pivot_test.loc[k, n]
            ax.text(ci, ri, f"{rv:+.3f}", ha="center", va="center",
                    fontsize=7.5, color="black" if rv > (vmin+vmax)/2 else "white",
                    fontweight="bold")
            ax.text(ci, ri + 0.30, f"te:{rt:+.3f}", ha="center", va="center",
                    fontsize=5.5, color="#333333" if rv > (vmin+vmax)/2 else "#cccccc")

    # Mark best cell
    best_m = best_models[tgt]
    best_k_idx   = list(pivot_val.index).index(best_m["k"])
    best_n_idx   = list(pivot_val.columns).index(best_m["n_est"])
    rect = plt.Rectangle((best_n_idx - 0.5, best_k_idx - 0.5), 1, 1,
                          linewidth=2.5, edgecolor="#ff6b6b", facecolor="none", zorder=5)
    ax.add_patch(rect)

    ax.set_xticks(range(len(pivot_val.columns)))
    ax.set_yticks(range(len(pivot_val.index)))
    ax.set_xticklabels([str(n) for n in pivot_val.columns],
                       color=TEXT_CLR, fontsize=8)
    ax.set_yticklabels([str(k) for k in pivot_val.index],
                       color=TEXT_CLR, fontsize=8)
    ax.set_xlabel("n_estimators", color=TEXT_CLR, fontsize=8, labelpad=4)
    ax.set_ylabel("n_features",   color=TEXT_CLR, fontsize=8, labelpad=4)

    for spine in ax.spines.values():
        spine.set_color("#444444")

    title = (f"{TARGET_LABELS[tgt]}\n"
             f"Best: k={best_m['k']} trees={best_m['n_est']}  "
             f"ρ_val={best_m['rho_val']:+.3f}  ρ_test={best_m['rho_test']:+.3f}")
    ax.set_title(title, color=TEXT_CLR, fontsize=8.5, fontweight="bold", pad=6)

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035, shrink=0.85)
    cb.ax.tick_params(labelsize=6, colors=TEXT_CLR)
    cb.set_label("ρ_val", color=TEXT_CLR, fontsize=7)

fig.tight_layout(pad=1.0)
fig.savefig(OUT_PLOTS / "rf_grid_heatmap.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG)
plt.close(fig)
print(f"  Saved: rf_grid_heatmap.png")

# ─── Test-set summary bar chart ───────────────────────────────────────────────
print("Rendering test-set summary ...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor=DARK_BG)
fig.suptitle(f"Grid Search: ρ_test by config  |  RED = best by ρ_val",
             color=TEXT_CLR, fontsize=12, fontweight="bold", y=1.01)

for ax_i, tgt in enumerate(TARGETS):
    ax = axes[ax_i // 3][ax_i % 3]
    ax.set_facecolor(PANEL_BG)

    sub = results_df[results_df["element"] == tgt].copy()
    sub["config"] = sub.apply(lambda r: f"k{int(r.k)}/n{int(r.n_est)}", axis=1)
    sub = sub.sort_values(["k", "n_est"])

    best_m = best_models[tgt]
    best_label = f"k{best_m['k']}/n{best_m['n_est']}"

    colors = ["#ff6b6b" if c == best_label else "#4fc3f7" for c in sub["config"]]
    bars = ax.barh(sub["config"], sub["rho_test"], color=colors, alpha=0.85,
                   edgecolor="#333333", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, sub["rho_test"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha="left",
                color=TEXT_CLR, fontsize=7)

    # Val-set rho markers (dashed lines)
    for _, row in sub.iterrows():
        y_pos = list(sub["config"]).index(row["config"])
        ax.plot([row["rho_val"]], [y_pos], marker="|", color="#f9c74f",
                ms=8, mew=2, zorder=5)

    ax.axvline(0, color="#666666", lw=0.8, ls="--")
    ax.set_title(f"{TARGET_LABELS[tgt]}\n"
                 f"Best: {best_label}  ρ_val={best_m['rho_val']:+.3f}  ρ_test={best_m['rho_test']:+.3f}",
                 color=TEXT_CLR, fontsize=8.5, fontweight="bold")
    ax.set_xlabel("ρ_test  (Spearman on held-out 20% fields)", color=TEXT_CLR, fontsize=8)
    ax.tick_params(colors=TEXT_CLR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#444444")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            Patch(facecolor="#ff6b6b", label="Best (by ρ_val)"),
            Patch(facecolor="#4fc3f7", label="Other configs"),
            Line2D([0], [0], marker="|", color="#f9c74f", ms=8, lw=0,
                   label="ρ_val marker"),
        ],
        fontsize=6, labelcolor=TEXT_CLR,
        facecolor=PANEL_BG, edgecolor="#444444", loc="lower right",
    )

fig.tight_layout(pad=1.0)
fig.savefig(OUT_PLOTS / "rf_grid_test_summary.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG)
plt.close(fig)
print(f"  Saved: rf_grid_test_summary.png")

# ─── Scatter: predicted vs actual (test set, best model per element) ──────────
print("Rendering test-set scatter plots ...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=DARK_BG)
fig.suptitle("Best Model  |  Test Set: Predicted vs Actual  (held-out 20% fields)",
             color=TEXT_CLR, fontsize=12, fontweight="bold", y=1.01)

for ax_i, tgt in enumerate(TARGETS):
    ax = axes[ax_i // 3][ax_i % 3]
    ax.set_facecolor(PANEL_BG)

    m = best_models[tgt]
    y_te     = m["y_te"]
    y_hat_te = m["y_hat_te"]

    if m["log_transform"]:
        y_te_plot     = np.expm1(y_te)
        y_hat_te_plot = np.expm1(y_hat_te)
    else:
        y_te_plot     = y_te
        y_hat_te_plot = y_hat_te

    if len(y_te_plot) < 2:
        ax.text(0.5, 0.5, "No test data", ha="center", va="center",
                color=TEXT_CLR, transform=ax.transAxes)
        continue

    rho_te = m["rho_test"]
    rmse_te = m["rmse_test"]

    ax.scatter(y_te_plot, y_hat_te_plot, color="#4fc3f7", alpha=0.7,
               s=25, edgecolors="#1a6b8a", linewidths=0.5)

    # 1:1 line
    lo = min(y_te_plot.min(), y_hat_te_plot.min())
    hi = max(y_te_plot.max(), y_hat_te_plot.max())
    ax.plot([lo, hi], [lo, hi], "w--", lw=1.2, alpha=0.7)

    # Regression line
    from numpy.polynomial import polynomial as P
    c = P.polyfit(y_te_plot, y_hat_te_plot, 1)
    ax.plot([lo, hi], P.polyval([lo, hi], c), color="#ff6b6b", lw=1.5, alpha=0.8)

    ax.set_xlabel(f"Actual {TARGET_LABELS[tgt]}", color=TEXT_CLR, fontsize=8)
    ax.set_ylabel(f"Predicted {TARGET_LABELS[tgt]}", color=TEXT_CLR, fontsize=8)
    ax.set_title(
        f"{TARGET_LABELS[tgt]}\n"
        f"k={m['k']}  n_est={m['n_est']}  "
        f"ρ_test={rho_te:+.3f}  RMSE={rmse_te:.3f}  n={len(y_te_plot)}",
        color=TEXT_CLR, fontsize=8.5, fontweight="bold",
    )
    ax.tick_params(colors=TEXT_CLR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#444444")

fig.tight_layout(pad=1.0)
fig.savefig(OUT_PLOTS / "rf_grid_scatter_test.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG)
plt.close(fig)
print(f"  Saved: rf_grid_scatter_test.png")

# ─── Text report ──────────────────────────────────────────────────────────────
report_path = OUT_RF / "rf_grid_test_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("RF Grid Search — Test Set Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Grid:   n_features {N_FEATURES_GRID}\n")
    f.write(f"        n_estimators {N_ESTIMATORS_GRID}\n")
    f.write(f"        {len(N_FEATURES_GRID)}×{len(N_ESTIMATORS_GRID)} = "
            f"{len(N_FEATURES_GRID)*len(N_ESTIMATORS_GRID)} configs per element\n")
    f.write(f"        {len(TARGETS)} elements × "
            f"{len(N_FEATURES_GRID)*len(N_ESTIMATORS_GRID)} = "
            f"{len(TARGETS)*len(N_FEATURES_GRID)*len(N_ESTIMATORS_GRID)} total fits\n\n")
    f.write(f"Split:  {int(TRAIN_FRAC*100)}% train / {int(VAL_FRAC*100)}% val / "
            f"{int(TEST_FRAC*100)}% test  (field-level, no leakage)\n")
    f.write(f"        Train fields: {len(train_fields)}  "
            f"Val fields: {len(val_fields)}  Test fields: {len(test_fields)}\n\n")
    f.write("Best Model per Element (selected by ρ_val):\n")
    f.write("-" * 70 + "\n")
    f.write(f"  {'Element':<8} {'k':>4} {'n_est':>6} "
            f"{'ρ_train':>8} {'ρ_val':>8} {'ρ_test':>8} "
            f"{'RMSE_val':>10} {'RMSE_test':>10} {'OOB_R²':>8}\n")
    f.write("  " + "-" * 70 + "\n")
    for tgt in TARGETS:
        m = best_models[tgt]
        f.write(f"  {tgt:<8} {m['k']:>4} {m['n_est']:>6} "
                f"{m['rho_train']:>+8.4f} {m['rho_val']:>+8.4f} {m['rho_test']:>+8.4f} "
                f"{m['rmse_val']:>10.4f} {m['rmse_test']:>10.4f} {m['oob_r2']:>8.4f}\n")
    f.write("\nTop features (best model):\n")
    for tgt in TARGETS:
        m = best_models[tgt]
        f.write(f"  {tgt}: {', '.join(m['features'][:5])}\n")
    f.write("\nFull grid (all 144 model fits):\n")
    f.write("-" * 70 + "\n")
    for _, row in results_df.iterrows():
        f.write(f"  {row['element']:<5} k={int(row['k']):2d} n={int(row['n_est']):3d} | "
                f"ρ_tr={row['rho_train']:+.4f} ρ_val={row['rho_val']:+.4f} "
                f"ρ_te={row['rho_test']:+.4f} OOB={row['oob_r2']:.4f}\n")

print(f"  Saved: rf_grid_test_report.txt")

# ─── Final summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL TEST SET RESULTS  (best model per element)")
print("=" * 70)
print(f"  {'Element':<8} {'k':>4} {'n_est':>6} "
      f"{'ρ_train':>8} {'ρ_val':>8} {'ρ_test':>8} {'RMSE_test':>10}")
print("  " + "-" * 60)
for tgt in TARGETS:
    m = best_models[tgt]
    print(f"  {tgt:<8} {m['k']:>4} {m['n_est']:>6} "
          f"{m['rho_train']:>+8.4f} {m['rho_val']:>+8.4f} "
          f"{m['rho_test']:>+8.4f} {m['rmse_test']:>10.4f}")

print(f"\nOutputs → {OUT_RF}")
print(f"Plots   → {OUT_PLOTS}")
print(f"Total time: {time.time() - t_global_start:.1f}s")
print("Done.")
