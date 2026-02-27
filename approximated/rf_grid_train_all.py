"""
rf_grid_train_all.py
====================
Trains ALL 288 models:
  • 144 regression models  — 6 elements × 24 grid configs (6 feat-levels × 4 tree-levels)
  • 144 classifier models  — same grid, target = 3 classes (low/mid/high)
    specialised per element using quantile-based balanced thresholds

Grid:
  n_features   : [10, 15, 20, 25, 30, 35]   (step=5, ranked by permutation importance)
  n_estimators : [100, 150, 200, 250]        (step=50)
  → 24 unique configs per element

Split (field-level, no leakage):
  Test  20% of fields — held-out, never used during training/selection
  Val   16% of fields — used to pick best config per element
  Train 64% of fields — model fitting

Saved:
  rf_grid_all_metrics.csv        — all 288 rows: regression + classifier metrics
  rf_grid_registry.csv           — registry of all 288 model specs (no pkl)
  rf_best_regression_{tgt}.pkl   — best regression model per element (6 files)
  rf_best_classifier_{tgt}.pkl   — best classifier model per element (6 files)
  rf_grid_heatmap_reg.png        — ρ_val heatmaps (regression)
  rf_grid_heatmap_clf.png        — F1_val heatmaps (classifier)
  rf_grid_class_report.txt       — classification report on test set

Run: python approximated/rf_grid_train_all.py
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
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_squared_error, f1_score, accuracy_score,
                             classification_report, confusion_matrix)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
RF_DATA    = BASE / "data" / "features" / "rf_dataset.csv"
MODELS_DIR = BASE / "math_statistics" / "output" / "rf" / "rf_models"
OUT_RF     = BASE / "math_statistics" / "output" / "rf"
OUT_PLOTS  = BASE / "math_statistics" / "output" / "plots"
OUT_GRID   = OUT_RF / "grid_models"
OUT_RF.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)
OUT_GRID.mkdir(parents=True, exist_ok=True)

# ─── Grid ─────────────────────────────────────────────────────────────────────
N_FEATURES_GRID   = [10, 15, 20, 25, 30, 35]   # 6 levels
N_ESTIMATORS_GRID = [100, 150, 200, 250]        # 4 levels → 24 configs total

TARGETS     = ["ph", "k", "p", "hu", "s", "no3"]
LOG_TARGETS = {"p", "s", "no3"}
TARGET_LABELS = {
    "ph": "pH", "k": "K mg/kg", "p": "P mg/kg",
    "hu": "Humus %", "s": "S mg/kg", "no3": "NO₃ mg/kg",
}

# ─── Class thresholds (quantile p33/p67 for balanced 3-class) ─────────────────
# Will be computed from training data only (no leakage)
CLASS_NAMES = ["low", "mid", "high"]

# ─── Split config ─────────────────────────────────────────────────────────────
TEST_FRAC = 0.20
VAL_FRAC  = 0.16
RANDOM_STATE = 42
RF_SEED      = 42
RF_TREE_MFEAT    = "sqrt"
RF_MIN_SAMPLES_LEAF = 3

# ─── Target classifier accuracy goals ─────────────────────────────────────────
TARGET_ACCURACY_MIN = 0.70
TARGET_ACCURACY_MAX = 0.90

print("=" * 70)
print("rf_grid_train_all.py  —  288 models (144 reg + 144 clf)")
print("=" * 70)
print(f"\nGrid: {N_FEATURES_GRID} feats × {N_ESTIMATORS_GRID} trees = "
      f"{len(N_FEATURES_GRID)*len(N_ESTIMATORS_GRID)} configs")
print(f"  {len(TARGETS)} elements × {len(N_FEATURES_GRID)*len(N_ESTIMATORS_GRID)} × 2 = "
      f"{len(TARGETS)*len(N_FEATURES_GRID)*len(N_ESTIMATORS_GRID)*2} total models")
print(f"Classifier target accuracy: {TARGET_ACCURACY_MIN:.0%} – {TARGET_ACCURACY_MAX:.0%}")

# ─── Load data ────────────────────────────────────────────────────────────────
print("\nLoading rf_dataset.csv ...")
rf_df = pd.read_csv(RF_DATA)
print(f"  Shape: {rf_df.shape}")

all_fields = sorted(rf_df["field_name"].unique())
n_fields   = len(all_fields)
print(f"  Unique fields: {n_fields}")

# ─── Field-level split (same seed as rf_grid_search.py) ───────────────────────
rng = np.random.default_rng(RANDOM_STATE)
shuffled = rng.permutation(np.array(all_fields))

n_test  = max(1, int(np.ceil(n_fields * TEST_FRAC)))
n_val   = max(1, int(np.ceil(n_fields * VAL_FRAC)))
n_train = n_fields - n_test - n_val

test_fields  = set(shuffled[:n_test])
val_fields   = set(shuffled[n_test : n_test + n_val])
train_fields = set(shuffled[n_test + n_val:])

print(f"\nField split: train={len(train_fields)} | val={len(val_fields)} | "
      f"test={len(test_fields)}  (same seed as rf_grid_search.py)")

# ─── Helper: assign class labels ──────────────────────────────────────────────
def make_labels(y, lo, hi):
    """0=low, 1=mid, 2=high based on thresholds lo/hi."""
    labels = np.ones(len(y), dtype=int)   # default = mid
    labels[y < lo]  = 0                  # low
    labels[y > hi]  = 2                  # high
    return labels

# ─── Main loop ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TRAINING ALL 288 MODELS")
print("=" * 70)

all_metrics  = []   # one row per (element, k, n_est, model_type)
best_reg_models = {}   # tgt → bundle
best_clf_models = {}   # tgt → bundle

t_global = time.time()
n_configs = len(N_FEATURES_GRID) * len(N_ESTIMATORS_GRID)

for tgt in TARGETS:
    print(f"\n{'═'*60}")
    print(f"  ELEMENT: {tgt.upper()}  ({TARGET_LABELS[tgt]})")
    print(f"{'═'*60}")

    # Load pre-ranked feature list
    with open(MODELS_DIR / f"rf_{tgt}.pkl", "rb") as f:
        bundle = pickle.load(f)
    ranked_features = bundle["features"]   # 40 features, sorted by permutation importance
    log_transform   = bundle["log_transform"]

    # Row selection
    mask_col = f"mask_{tgt}"
    valid_mask = rf_df[mask_col].astype(bool) if mask_col in rf_df.columns else rf_df[tgt].notna()
    df_valid = rf_df[valid_mask].copy()

    tgt_col = f"log_{tgt}" if log_transform and f"log_{tgt}" in df_valid.columns else tgt
    y_all   = df_valid[tgt_col].values.astype(float)
    y_orig  = df_valid[tgt].values.astype(float)   # original scale for thresholds
    fids    = df_valid["field_name"].values

    # Split masks
    tr_mask = np.array([f in train_fields for f in fids])
    va_mask = np.array([f in val_fields   for f in fids])
    te_mask = np.array([f in test_fields  for f in fids])

    # ── Compute class thresholds on TRAIN only (no leakage) ───────────────────
    y_tr_orig = y_orig[tr_mask]
    lo_thresh  = np.percentile(y_tr_orig, 33)
    hi_thresh  = np.percentile(y_tr_orig, 67)

    # Class labels (on original scale)
    c_all = make_labels(y_orig, lo_thresh, hi_thresh)
    c_tr  = c_all[tr_mask]
    c_va  = c_all[va_mask]
    c_te  = c_all[te_mask]

    counts_tr = np.bincount(c_tr, minlength=3)
    counts_te = np.bincount(c_te, minlength=3)
    print(f"\n  n_valid={len(df_valid)} | train={tr_mask.sum()} val={va_mask.sum()} test={te_mask.sum()}")
    print(f"  log_transform={log_transform}  target_col={tgt_col}")
    print(f"  Class thresholds: low<{lo_thresh:.3f}  high>{hi_thresh:.3f}")
    print(f"  Train class dist: low={counts_tr[0]} mid={counts_tr[1]} high={counts_tr[2]}")
    print(f"  Test  class dist: low={counts_te[0]} mid={counts_te[1]} high={counts_te[2]}")
    print()
    print(f"  {'Config':<16} | {'REG: ρ_tr':>9} {'ρ_val':>7} {'ρ_te':>7} | "
          f"{'CLF: acc_tr':>10} {'acc_val':>8} {'acc_te':>8} {'f1_val':>7}")
    print(f"  {'-'*90}")

    best_reg_val = -np.inf
    best_clf_val = -np.inf
    best_reg_cfg = None
    best_clf_cfg = None

    config_idx = 0
    for k in N_FEATURES_GRID:
        feats_k = ranked_features[:k]
        X_k     = df_valid[feats_k].values.astype(float)

        X_tr = X_k[tr_mask];  y_tr = y_all[tr_mask]
        X_va = X_k[va_mask];  y_va = y_all[va_mask]
        X_te = X_k[te_mask];  y_te = y_all[te_mask]

        # Imputer fitted on train only
        imp = SimpleImputer(strategy="median")
        X_tr_i = imp.fit_transform(X_tr)
        X_va_i = imp.transform(X_va)
        X_te_i = imp.transform(X_te)

        for n_est in N_ESTIMATORS_GRID:
            config_idx += 1
            cfg_label = f"k={k:2d} n={n_est:3d}"
            t0 = time.time()

            # ── Regression ────────────────────────────────────────────────────
            rf_reg = RandomForestRegressor(
                n_estimators=n_est, max_features=RF_TREE_MFEAT,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                oob_score=True, random_state=RF_SEED, n_jobs=-1,
            )
            rf_reg.fit(X_tr_i, y_tr)

            yh_tr = rf_reg.predict(X_tr_i)
            yh_va = rf_reg.predict(X_va_i)
            yh_te = rf_reg.predict(X_te_i)

            def bt(arr):
                return np.expm1(arr) if log_transform else arr

            rho_tr  = spearmanr(y_tr, yh_tr)[0]
            rho_va  = spearmanr(y_va, yh_va)[0] if len(y_va) > 1 else np.nan
            rho_te  = spearmanr(y_te, yh_te)[0] if len(y_te) > 1 else np.nan
            rmse_va = np.sqrt(mean_squared_error(bt(y_va), bt(yh_va))) if len(y_va) > 1 else np.nan
            rmse_te = np.sqrt(mean_squared_error(bt(y_te), bt(yh_te))) if len(y_te) > 1 else np.nan
            oob_r2  = rf_reg.oob_score_

            # ── Classifier ────────────────────────────────────────────────────
            rf_clf = RandomForestClassifier(
                n_estimators=n_est, max_features=RF_TREE_MFEAT,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                class_weight="balanced",    # handles imbalance
                oob_score=True, random_state=RF_SEED, n_jobs=-1,
            )
            rf_clf.fit(X_tr_i, c_tr)

            ch_tr = rf_clf.predict(X_tr_i)
            ch_va = rf_clf.predict(X_va_i)
            ch_te = rf_clf.predict(X_te_i)

            acc_tr  = accuracy_score(c_tr, ch_tr)
            acc_va  = accuracy_score(c_va, ch_va) if len(c_va) > 0 else np.nan
            acc_te  = accuracy_score(c_te, ch_te) if len(c_te) > 0 else np.nan
            f1_va   = f1_score(c_va, ch_va, average="macro") if len(c_va) > 0 else np.nan
            f1_te   = f1_score(c_te, ch_te, average="macro") if len(c_te) > 0 else np.nan
            oob_clf = rf_clf.oob_score_

            elapsed = time.time() - t0

            # Track bests
            is_best_reg = rho_va > best_reg_val
            is_best_clf = f1_va > best_clf_val

            if is_best_reg:
                best_reg_val = rho_va
                best_reg_cfg = {
                    "rf": rf_reg, "imputer": imp,
                    "features": feats_k, "k": k, "n_est": n_est,
                    "rho_train": rho_tr, "rho_val": rho_va, "rho_test": rho_te,
                    "rmse_val": rmse_va, "rmse_test": rmse_te, "oob_r2": oob_r2,
                    "log_transform": log_transform, "tgt": tgt, "tgt_col": tgt_col,
                }
            if is_best_clf:
                best_clf_val = f1_va
                best_clf_cfg = {
                    "rf": rf_clf, "imputer": imp,
                    "features": feats_k, "k": k, "n_est": n_est,
                    "acc_train": acc_tr, "acc_val": acc_va, "acc_test": acc_te,
                    "f1_val": f1_va, "f1_test": f1_te, "oob_acc": oob_clf,
                    "lo_thresh": lo_thresh, "hi_thresh": hi_thresh,
                    "class_names": CLASS_NAMES, "tgt": tgt,
                    "c_te": c_te, "ch_te": ch_te,
                }

            reg_mark = " ★" if is_best_reg else ""
            clf_mark = " ✦" if is_best_clf else ""

            print(f"  [{config_idx:3d}/{n_configs}] {cfg_label} | "
                  f"{rho_tr:>+9.4f} {rho_va:>+7.4f} {rho_te:>+7.4f} |"
                  f" {acc_tr:>10.4f} {acc_va:>8.4f} {acc_te:>8.4f} {f1_va:>7.4f}"
                  f"  ({elapsed:.1f}s){reg_mark}{clf_mark}")

            # Save all metrics
            all_metrics.append({
                "element": tgt, "k": k, "n_est": n_est,
                # regression
                "reg_rho_train": round(rho_tr,  4), "reg_rho_val":  round(rho_va,  4),
                "reg_rho_test":  round(rho_te,  4), "reg_rmse_val": round(rmse_va, 4),
                "reg_rmse_test": round(rmse_te, 4), "reg_oob_r2":   round(oob_r2,  4),
                "log_transform": log_transform,
                # classifier
                "clf_acc_train": round(acc_tr, 4), "clf_acc_val":  round(acc_va, 4),
                "clf_acc_test":  round(acc_te, 4), "clf_f1_val":   round(f1_va,  4),
                "clf_f1_test":   round(f1_te,  4), "clf_oob_acc":  round(oob_clf, 4),
                "lo_thresh": round(lo_thresh, 4), "hi_thresh": round(hi_thresh, 4),
                "is_best_reg": is_best_reg, "is_best_clf": is_best_clf,
            })

    best_reg_models[tgt] = best_reg_cfg
    best_clf_models[tgt] = best_clf_cfg

    print(f"\n  ★ Best REG: k={best_reg_cfg['k']} n_est={best_reg_cfg['n_est']}"
          f"  ρ_val={best_reg_cfg['rho_val']:+.4f}  ρ_test={best_reg_cfg['rho_test']:+.4f}")
    print(f"  ✦ Best CLF: k={best_clf_cfg['k']} n_est={best_clf_cfg['n_est']}"
          f"  acc_val={best_clf_cfg['acc_val']:.4f}  acc_test={best_clf_cfg['acc_test']:.4f}"
          f"  f1_val={best_clf_cfg['f1_val']:.4f}")

    # Check if classifier meets accuracy target
    acc_te = best_clf_cfg["acc_test"]
    status = "✅ IN RANGE" if TARGET_ACCURACY_MIN <= acc_te <= TARGET_ACCURACY_MAX else \
             ("⬆ ABOVE" if acc_te > TARGET_ACCURACY_MAX else "⬇ BELOW")
    print(f"  Classifier test accuracy: {acc_te:.4f}  {status} ({TARGET_ACCURACY_MIN:.0%}–{TARGET_ACCURACY_MAX:.0%})")

print(f"\nTotal training time: {time.time()-t_global:.1f}s")

# ─── Save best models (pkl) ────────────────────────────────────────────────────
print("\nSaving best models (pkl) ...")
for tgt in TARGETS:
    # Regression
    reg_path = OUT_GRID / f"rf_best_regression_{tgt}.pkl"
    with open(reg_path, "wb") as f:
        pickle.dump(best_reg_models[tgt], f)
    # Classifier
    clf_path = OUT_GRID / f"rf_best_classifier_{tgt}.pkl"
    with open(clf_path, "wb") as f:
        pickle.dump(best_clf_models[tgt], f)
    print(f"  {tgt}: saved best_regression + best_classifier")

# ─── Save metrics CSV ─────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(OUT_RF / "rf_grid_all_metrics.csv", index=False)
print(f"\nSaved: rf_grid_all_metrics.csv  ({len(metrics_df)} rows)")

# ─── Save registry (all 288 model specs, no pkl) ──────────────────────────────
registry_rows = []
for _, row in metrics_df.iterrows():
    for mtype in ["reg", "clf"]:
        registry_rows.append({
            "model_id":   f"{row['element']}_{mtype}_k{int(row['k'])}_n{int(row['n_est'])}",
            "element":    row["element"],
            "model_type": "regression" if mtype == "reg" else "classifier",
            "k":          int(row["k"]),
            "n_est":      int(row["n_est"]),
            "val_metric": row[f"{mtype}_rho_val"] if mtype == "reg" else row["clf_f1_val"],
            "test_metric":row[f"{mtype}_rho_test"] if mtype == "reg" else row["clf_acc_test"],
            "is_best":    row[f"is_best_{mtype}"],
            "features":   f"top-{int(row['k'])} by permutation importance",
        })
reg_df = pd.DataFrame(registry_rows)
reg_df.to_csv(OUT_RF / "rf_grid_registry.csv", index=False)
print(f"Saved: rf_grid_registry.csv  ({len(reg_df)} rows = 288 models)")

# ─── HEATMAP: ρ_val regression ────────────────────────────────────────────────
DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
TEXT_CLR = "#e0e0e0"

def make_heatmap(fig_title, metric_col, label, vmin_override=None, fname="heatmap.png",
                 best_col="is_best_reg", annotate_col2=None, annotate_label2=""):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=DARK_BG)
    fig.suptitle(fig_title, color=TEXT_CLR, fontsize=12, fontweight="bold", y=1.01)

    for ax_i, tgt in enumerate(TARGETS):
        ax = axes[ax_i // 3][ax_i % 3]
        ax.set_facecolor(PANEL_BG)
        sub = metrics_df[metrics_df["element"] == tgt]
        pivot = sub.pivot(index="k", columns="n_est", values=metric_col)
        p2 = sub.pivot(index="k", columns="n_est", values=annotate_col2) if annotate_col2 else None

        vmin = vmin_override if vmin_override is not None else pivot.values.min()
        vmax = pivot.values.max()
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                       vmin=vmin, vmax=vmax, origin="upper")

        best_m = sub[sub[best_col]].iloc[0] if sub[best_col].any() else None

        for ri, k in enumerate(pivot.index):
            for ci, n in enumerate(pivot.columns):
                v = pivot.loc[k, n]
                color = "black" if v > (vmin + vmax) / 2 else "white"
                ax.text(ci, ri - 0.18, f"{v:+.3f}" if v > -1 else f"{v:.3f}",
                        ha="center", va="center", fontsize=7.5,
                        color=color, fontweight="bold")
                if p2 is not None:
                    v2 = p2.loc[k, n]
                    ax.text(ci, ri + 0.28, f"{annotate_label2}{v2:.3f}",
                            ha="center", va="center", fontsize=5.5,
                            color="#555" if v > (vmin + vmax) / 2 else "#bbb")

        if best_m is not None:
            bi = list(pivot.index).index(best_m["k"])
            bj = list(pivot.columns).index(best_m["n_est"])
            ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                         lw=2.5, edgecolor="#ff6b6b", facecolor="none", zorder=5))

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([str(n) for n in pivot.columns], color=TEXT_CLR, fontsize=8)
        ax.set_yticklabels([str(k) for k in pivot.index], color=TEXT_CLR, fontsize=8)
        ax.set_xlabel("n_estimators", color=TEXT_CLR, fontsize=8)
        ax.set_ylabel("n_features",   color=TEXT_CLR, fontsize=8)
        for sp in ax.spines.values(): sp.set_color("#444444")

        best_txt = ""
        if best_m is not None:
            v_best = pivot.loc[best_m["k"], best_m["n_est"]]
            best_txt = f"  Best: k={int(best_m['k'])} n={int(best_m['n_est'])}  {label}={v_best:+.3f}"
        ax.set_title(f"{TARGET_LABELS[tgt]}{best_txt}",
                     color=TEXT_CLR, fontsize=8.5, fontweight="bold", pad=6)

        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.033, shrink=0.85)
        cb.ax.tick_params(labelsize=6, colors=TEXT_CLR)
        cb.set_label(label, color=TEXT_CLR, fontsize=7)

    fig.tight_layout(pad=1.0)
    fig.savefig(OUT_PLOTS / fname, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {fname}")

print("\nRendering heatmaps ...")
make_heatmap(
    fig_title=(f"Regression: ρ_val  |  Grid {N_FEATURES_GRID} feats × {N_ESTIMATORS_GRID} trees\n"
               f"Split: 64% train / 16% val / 20% test  |  Red box = best by ρ_val"),
    metric_col="reg_rho_val", label="ρ_val",
    annotate_col2="reg_rho_test", annotate_label2="te:",
    best_col="is_best_reg", fname="rf_grid_heatmap_reg.png",
)
make_heatmap(
    fig_title=(f"Classifier: F1_val (macro)  |  3 classes: low/mid/high (p33/p67)\n"
               f"Grid {N_FEATURES_GRID} feats × {N_ESTIMATORS_GRID} trees  |  Red box = best by F1_val"),
    metric_col="clf_f1_val", label="F1_val",
    vmin_override=0.0,
    annotate_col2="clf_acc_test", annotate_label2="acc:",
    best_col="is_best_clf", fname="rf_grid_heatmap_clf.png",
)

make_heatmap(
    fig_title=(f"Classifier: Accuracy on TEST set  |  3 classes: low/mid/high (p33/p67)\n"
               f"Target: {TARGET_ACCURACY_MIN:.0%}–{TARGET_ACCURACY_MAX:.0%}  |  Red box = best by F1_val"),
    metric_col="clf_acc_test", label="acc_test",
    vmin_override=0.0,
    best_col="is_best_clf", fname="rf_grid_heatmap_clf_acc.png",
)

# ─── Classification report on test set ────────────────────────────────────────
print("\nSaving classification report ...")
report_path = OUT_RF / "rf_grid_class_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("RF Grid Search — Classification Report (Test Set)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Grid: features={N_FEATURES_GRID}  trees={N_ESTIMATORS_GRID}\n")
    f.write(f"Classes: 0=low (<p33_train)  1=mid  2=high (>p67_train)\n")
    f.write(f"Target accuracy: {TARGET_ACCURACY_MIN:.0%} – {TARGET_ACCURACY_MAX:.0%}\n\n")

    f.write("Best Classifier per Element:\n")
    f.write("-" * 70 + "\n")
    f.write(f"  {'Element':<8} {'k':>4} {'n_est':>6} {'lo_thr':>8} {'hi_thr':>8} "
            f"{'acc_tr':>8} {'acc_val':>8} {'acc_te':>8} {'f1_val':>8}  status\n")
    f.write("  " + "-" * 70 + "\n")

    for tgt in TARGETS:
        m = best_clf_models[tgt]
        acc_te = m["acc_test"]
        status = "IN_RANGE" if TARGET_ACCURACY_MIN <= acc_te <= TARGET_ACCURACY_MAX else \
                 ("ABOVE" if acc_te > TARGET_ACCURACY_MAX else "BELOW")
        f.write(f"  {tgt:<8} {m['k']:>4} {m['n_est']:>6} {m['lo_thresh']:>8.3f} {m['hi_thresh']:>8.3f} "
                f"{m['acc_train']:>8.4f} {m['acc_val']:>8.4f} {m['acc_test']:>8.4f} "
                f"{m['f1_val']:>8.4f}  {status}\n")

    f.write("\n\nDetailed Classification Reports (test set):\n")
    f.write("=" * 70 + "\n\n")
    for tgt in TARGETS:
        m = best_clf_models[tgt]
        f.write(f"[{tgt.upper()}]  k={m['k']}  n_est={m['n_est']}  "
                f"thresh: <{m['lo_thresh']:.3f} | {m['lo_thresh']:.3f}–{m['hi_thresh']:.3f} | >{m['hi_thresh']:.3f}\n")
        f.write(classification_report(m["c_te"], m["ch_te"],
                                      target_names=CLASS_NAMES, digits=3))
        f.write("\nConfusion matrix (rows=actual, cols=predicted):\n")
        cm = confusion_matrix(m["c_te"], m["ch_te"])
        for row in cm:
            f.write("  " + "  ".join(f"{v:4d}" for v in row) + "\n")
        f.write("\n")

print(f"  Saved: rf_grid_class_report.txt")

# ─── Final summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\n  Total models trained: {len(metrics_df)} regression + {len(metrics_df)} classifier = {2*len(metrics_df)}")
print(f"  Saved pkl: 6 best_regression + 6 best_classifier = 12 files\n")

print(f"  {'Element':<8} | {'REGRESSION':^35} | {'CLASSIFIER':^40}")
print(f"  {'':8} | {'k':>4} {'n_est':>5} {'ρ_val':>7} {'ρ_test':>7} | "
      f"{'k':>4} {'n_est':>5} {'acc_val':>8} {'acc_te':>8} {'f1_val':>7}  status")
print("  " + "-" * 88)
for tgt in TARGETS:
    r = best_reg_models[tgt]
    c = best_clf_models[tgt]
    acc_te = c["acc_test"]
    status = "✅" if TARGET_ACCURACY_MIN <= acc_te <= TARGET_ACCURACY_MAX else \
             ("⬆" if acc_te > TARGET_ACCURACY_MAX else "⬇")
    print(f"  {tgt:<8} | {r['k']:>4} {r['n_est']:>5} {r['rho_val']:>+7.4f} {r['rho_test']:>+7.4f} | "
          f"{c['k']:>4} {c['n_est']:>5} {c['acc_val']:>8.4f} {acc_te:>8.4f} {c['f1_val']:>7.4f}  {status}")

print(f"\nOutputs:")
print(f"  Metrics:  {OUT_RF}/rf_grid_all_metrics.csv")
print(f"  Registry: {OUT_RF}/rf_grid_registry.csv")
print(f"  Models:   {OUT_GRID}/rf_best_{{regression,classifier}}_{{tgt}}.pkl  (12 files)")
print(f"  Plots:    rf_grid_heatmap_reg.png  rf_grid_heatmap_clf.png  rf_grid_heatmap_clf_acc.png")
print(f"  Report:   rf_grid_class_report.txt")
print(f"\nTotal time: {time.time()-t_global:.1f}s")
print("Done.")
