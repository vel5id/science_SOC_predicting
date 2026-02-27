"""
rf_train_cv.py
==============
Stage 2 of the RF pipeline: feature selection, RF training, LOFO-CV.

Feature selection (3 stages):
  1. Variance filter (std < 1e-4 or NaN > 40%)
  2. Correlation deduplication (Spearman ρ > 0.92 clustering, per-target)
  3. RF preliminary importance (top-40 by permutation importance)

Training:
  - Final RF on all valid data per target (n_estimators=500)
  - LOFO-CV by field_name (92 unique fields)
  - Bootstrap CI (n=500, seed=42)
  - Log targets (p, s, no3) trained on log1p, predictions back-transformed

Outputs:
  rf/rf_oof_predictions.csv
  rf/rf_models/rf_{element}.pkl
  rf/rf_feature_selected_{element}.csv
  rf/rf_cv_metrics.csv
  rf/rf_report.txt
  rf/rf_cv_summary.png
  rf/rf_scatter_{element}.png

Run: python approximated/rf_train_cv.py
"""

import sys, io, warnings, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

BASE       = Path(__file__).parent.parent
RF_CSV     = BASE / "data" / "features" / "rf_dataset.csv"
OUT_DIR    = BASE / "math_statistics" / "output" / "rf"
MODEL_DIR  = OUT_DIR / "rf_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGETS    = ["ph", "k", "p", "hu", "s", "no3"]
LOG_TARGETS = {"p", "s", "no3"}   # trained on log1p, preds back-transformed

CHEM_LABELS = {
    "ph":  "pH",
    "k":   "K, mg/kg",
    "p":   "P, mg/kg",
    "hu":  "Humus, %",
    "s":   "S, mg/kg",
    "no3": "NO3, mg/kg",
}
CHEM_CMAPS = {
    "ph": "#e05c4a", "k": "#e0a44a", "p": "#6abf69",
    "hu": "#8d6e63", "s": "#4ab5e0", "no3": "#b44ae0",
}

RF_PARAMS = dict(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=3,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    oob_score=True,
)
RF_FAST = dict(n_estimators=100, max_features="sqrt",
               min_samples_leaf=3, n_jobs=-1, random_state=42)

BOOTSTRAP_N    = 500
BOOTSTRAP_SEED = 42
CORR_THRESHOLD = 0.92   # deduplication threshold
TOP_FEATS      = 40     # features retained after stage-3

META_COLS = ["year", "farm", "field_name", "centroid_lon", "centroid_lat"]

print("=" * 70)
print("rf_train_cv.py  —  RF Feature Selection + Training + LOFO-CV")
print("=" * 70)

# ─── 1. Load dataset ─────────────────────────────────────────────
print(f"\nLoading rf_dataset.csv ...")
df = pd.read_csv(RF_CSV)
print(f"  Shape: {df.shape}")

feat_base = [c for c in df.columns
             if c not in META_COLS
             and c not in TARGETS
             and not c.startswith("log_")
             and not c.startswith("mask_")
             and c not in {"cu","mo","fe","zn","mg","mn","soc","b","ca","na"}]
print(f"  Candidate feature columns: {len(feat_base)}")
print(f"  Unique fields: {df['field_name'].nunique()}")
print(f"  Unique farms:  {df['farm'].nunique()}")


# ─── 2. Feature selection helpers ────────────────────────────────

def stage1_variance_filter(feat_cols, df_sub):
    """Remove near-constant features and high-NaN features."""
    keep = []
    for c in feat_cols:
        if c not in df_sub.columns:
            continue
        nan_frac = df_sub[c].isna().mean()
        if nan_frac > 0.40:
            continue
        if df_sub[c].std() < 1e-4:
            continue
        keep.append(c)
    return keep


def stage2_dedup(feat_cols, df_sub, target_col, threshold=CORR_THRESHOLD):
    """Hierarchical correlation clustering; keep best representative per cluster."""
    if len(feat_cols) <= 1:
        return feat_cols

    X = df_sub[feat_cols].copy()
    # Impute NaN with column median for correlation computation
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    # Spearman correlation matrix (pairwise)
    n = X_imp.shape[1]
    rho_mat = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r, _ = spearmanr(X_imp[:, i], X_imp[:, j])
            if np.isnan(r):
                r = 0.0
            rho_mat[i, j] = rho_mat[j, i] = r

    # Distance = 1 - |rho|
    dist_mat = 1.0 - np.abs(rho_mat)
    np.fill_diagonal(dist_mat, 0.0)
    dist_mat = np.clip(dist_mat, 0, None)

    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=(1.0 - threshold), criterion="distance")

    # For each cluster, pick feature with highest |Spearman(feature, target)|
    y = df_sub[target_col].values
    representatives = []
    for cluster_id in np.unique(labels):
        idx = np.where(labels == cluster_id)[0]
        cluster_feats = [feat_cols[i] for i in idx]
        best_feat = cluster_feats[0]
        best_rho  = 0.0
        for cf in cluster_feats:
            x = df_sub[cf].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 20:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r, _ = spearmanr(x[mask], y[mask])
            if np.isnan(r):
                r = 0.0
            if abs(r) > abs(best_rho):
                best_rho  = r
                best_feat = cf
        representatives.append(best_feat)
    return representatives


def stage3_rf_importance(feat_cols, X_tr, y_tr, top_n=TOP_FEATS):
    """Quick RF + permutation importance on held-out validation set."""
    imp = SimpleImputer(strategy="median")
    X_full = imp.fit_transform(X_tr)

    # Hold out 20% for permutation importance to avoid optimistic train-set estimates
    n = len(y_tr)
    val_size = max(int(n * 0.20), 1)
    rng_s3 = np.random.default_rng(42)
    val_idx = rng_s3.choice(n, val_size, replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)

    rf_fast = RandomForestRegressor(**RF_FAST)
    rf_fast.fit(X_full[train_idx], y_tr[train_idx])

    perm = permutation_importance(rf_fast, X_full[val_idx], y_tr[val_idx],
                                  n_repeats=5, random_state=42, n_jobs=-1)
    importances = perm.importances_mean
    top_idx = np.argsort(importances)[::-1][:top_n]
    return [feat_cols[i] for i in top_idx], importances[top_idx]


# ─── 3. Main loop: per-target ────────────────────────────────────
all_oof    = []   # list of dicts → DataFrame
all_metrics = {}  # tgt -> dict
saved_models = {}
selected_features = {}

for tgt in TARGETS:
    print(f"\n{'='*60}")
    print(f"  TARGET: {tgt.upper()}  ({CHEM_LABELS[tgt]})")
    print(f"{'='*60}")

    # Determine training column
    train_col = f"log_{tgt}" if tgt in LOG_TARGETS else tgt
    mask_col  = f"mask_{tgt}"

    # Subset: rows where mask = 1 (sigma filtered, not NaN)
    valid = df[mask_col].astype(bool)
    df_sub = df[valid].copy()
    print(f"  Valid training rows: {len(df_sub)}  "
          f"(fields: {df_sub['field_name'].nunique()}, "
          f"farms: {df_sub['farm'].nunique()})")

    # ── Stage 1: variance filter ──────────────────────────────────
    feats_s1 = stage1_variance_filter(feat_base, df_sub)
    print(f"  Stage 1 (variance):    {len(feat_base)} → {len(feats_s1)} features")

    # ── Stage 2: correlation dedup ────────────────────────────────
    print(f"  Stage 2 (corr dedup, ρ>{CORR_THRESHOLD}): computing ...", end="", flush=True)
    feats_s2 = stage2_dedup(feats_s1, df_sub, train_col)
    print(f" {len(feats_s1)} → {len(feats_s2)} features")

    # ── Stage 3: RF importance top-N ─────────────────────────────
    imp_data   = SimpleImputer(strategy="median")
    X_s2       = imp_data.fit_transform(df_sub[feats_s2])
    y_s2       = df_sub[train_col].values

    print(f"  Stage 3 (RF importance, top-{TOP_FEATS}): computing ...", end="", flush=True)
    feats_final, importances = stage3_rf_importance(feats_s2, X_s2, y_s2, top_n=TOP_FEATS)
    print(f" → {len(feats_final)} features")
    selected_features[tgt] = feats_final

    # Save feature list
    feat_df = pd.DataFrame({
        "feature": feats_final,
        "perm_importance": importances,
    })
    feat_df.to_csv(OUT_DIR / f"rf_feature_selected_{tgt}.csv", index=False)

    print(f"  Top-5 features: {feats_final[:5]}")

    # ── LOFO-CV ───────────────────────────────────────────────────
    fields = df_sub["field_name"].unique()
    print(f"  LOFO-CV: {len(fields)} folds ...")

    oof_true, oof_pred = [], []
    oof_meta = []
    fold_metrics = []

    for fi, field in enumerate(fields):
        test_mask  = df_sub["field_name"] == field
        train_mask = ~test_mask

        if train_mask.sum() < 20 or test_mask.sum() == 0:
            continue

        X_tr_raw = df_sub.loc[train_mask, feats_final].values
        y_tr     = df_sub.loc[train_mask, train_col].values
        X_te_raw = df_sub.loc[test_mask,  feats_final].values
        y_te_orig= df_sub.loc[test_mask,  tgt].values  # always original scale

        # Fit imputer only on train fold to avoid data leakage from test field
        imp_lofo = SimpleImputer(strategy="median")
        imp_lofo.fit(X_tr_raw)
        X_tr = imp_lofo.transform(X_tr_raw)
        X_te = imp_lofo.transform(X_te_raw)

        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_tr, y_tr)
        y_hat_log = rf.predict(X_te)

        # Back-transform if log target
        if tgt in LOG_TARGETS:
            y_hat = np.expm1(y_hat_log)
        else:
            y_hat = y_hat_log

        oof_true.extend(y_te_orig.tolist())
        oof_pred.extend(y_hat.tolist())
        for j in range(len(y_te_orig)):
            oof_meta.append({
                "field_name": field,
                "farm": df_sub.loc[test_mask, "farm"].iloc[j],
                "element": tgt,
                "y_true": y_te_orig[j],
                "y_pred": y_hat[j],
            })

        # Fold metrics
        if len(y_te_orig) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho_fold, _ = spearmanr(y_te_orig, y_hat)
            rmse_fold = np.sqrt(mean_squared_error(y_te_orig, y_hat))
            fold_metrics.append({"field": field, "rho": rho_fold, "rmse": rmse_fold})

        if (fi + 1) % 20 == 0:
            print(f"    fold {fi+1}/{len(fields)} ...", flush=True)

    oof_true = np.array(oof_true)
    oof_pred = np.array(oof_pred)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho_cv, _ = spearmanr(oof_true, oof_pred)
    rmse_cv = np.sqrt(mean_squared_error(oof_true, oof_pred))
    mae_cv  = mean_absolute_error(oof_true, oof_pred)
    r2_cv   = r2_score(oof_true, oof_pred)

    # Bootstrap CI
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_oof = len(oof_true)
    boot_rhos = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.choice(n_oof, n_oof, replace=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rb, _ = spearmanr(oof_true[idx], oof_pred[idx])
        if not np.isnan(rb):
            boot_rhos.append(rb)
    boot_rhos = np.array(boot_rhos)
    ci_lo = float(np.percentile(boot_rhos, 2.5))
    ci_hi = float(np.percentile(boot_rhos, 97.5))
    # Absolute-scale CI for bar chart (error bars around |rho_cv|)
    abs_boot_rhos = np.abs(boot_rhos)
    ci_lo_abs = float(np.percentile(abs_boot_rhos, 2.5))
    ci_hi_abs = float(np.percentile(abs_boot_rhos, 97.5))

    print(f"  OOF results: rho_cv={rho_cv:+.3f} [{ci_lo:+.3f},{ci_hi:+.3f}]  "
          f"RMSE={rmse_cv:.3f}  R2={r2_cv:.3f}  n={n_oof}")

    # ── Train final model on ALL valid data ───────────────────────
    X_full_raw = df_sub[feats_final].values
    y_full     = df_sub[train_col].values
    imp_final = SimpleImputer(strategy="median")
    imp_final.fit(X_full_raw)
    X_full     = imp_final.transform(X_full_raw)

    rf_final = RandomForestRegressor(**RF_PARAMS)
    rf_final.fit(X_full, y_full)

    y_pred_tr_log = rf_final.predict(X_full)
    if tgt in LOG_TARGETS:
        y_pred_tr = np.expm1(y_pred_tr_log)
        y_true_tr = df_sub[tgt].values
    else:
        y_pred_tr = y_pred_tr_log
        y_true_tr = y_full

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho_train, _ = spearmanr(y_pred_tr, y_true_tr)

    oob = rf_final.oob_score_
    print(f"  Final model: rho_train={rho_train:+.3f}  OOB_R2={oob:.3f}  "
          f"n_features={len(feats_final)}")

    # Save model + imputer together
    model_bundle = {
        "rf": rf_final,
        "imputer": imp_final,
        "features": feats_final,
        "tgt": tgt,
        "log_transform": tgt in LOG_TARGETS,
    }
    with open(MODEL_DIR / f"rf_{tgt}.pkl", "wb") as f:
        pickle.dump(model_bundle, f)
    saved_models[tgt] = model_bundle

    all_oof.extend(oof_meta)
    all_metrics[tgt] = {
        "element": tgt, "label": CHEM_LABELS[tgt],
        "rho_train": rho_train, "oob_r2": oob,
        "rho_cv": rho_cv, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "ci_lo_abs": ci_lo_abs, "ci_hi_abs": ci_hi_abs,
        "rmse_cv": rmse_cv, "mae_cv": mae_cv, "r2_cv": r2_cv,
        "n_train": len(df_sub), "n_oof": n_oof,
        "n_features": len(feats_final),
    }

# ─── 4. Save OOF predictions and metrics ─────────────────────────
print("\nSaving outputs ...")

oof_df = pd.DataFrame(all_oof)
oof_df.to_csv(OUT_DIR / "rf_oof_predictions.csv", index=False)
print(f"  Saved: rf_oof_predictions.csv ({len(oof_df)} rows)")

metrics_df = pd.DataFrame(list(all_metrics.values()))
metrics_df.to_csv(OUT_DIR / "rf_cv_metrics.csv", index=False)
print(f"  Saved: rf_cv_metrics.csv")

# ─── 5. Save text report ──────────────────────────────────────────
lines = ["RF TRAINING + LOFO-CV REPORT", "=" * 65, ""]
lines.append(f"Dataset: rf_dataset.csv ({df.shape[0]} rows, {len(feat_base)} candidate features)")
lines.append(f"Model: RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=sqrt)")
lines.append(f"CV: Leave-One-Field-Out (LOFO), {df['field_name'].nunique()} fields")
lines.append(f"Bootstrap CI: n={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}")
lines.append("")
lines.append(f"{'Element':<8}  {'n':>5}  {'n_feat':>6}  "
             f"{'rho_train':>10}  {'rho_cv':>8}  {'95% CI':>16}  "
             f"{'RMSE_cv':>8}  {'R2_cv':>7}  {'OOB_R2':>7}")
lines.append("-" * 90)
for t in TARGETS:
    m = all_metrics[t]
    ci_str = f"[{m['ci_lo']:+.3f},{m['ci_hi']:+.3f}]"
    lines.append(f"  {t:<6}  {m['n_train']:>5}  {m['n_features']:>6}  "
                 f"{m['rho_train']:>+10.3f}  {m['rho_cv']:>+8.3f}  {ci_str:>16}  "
                 f"{m['rmse_cv']:>8.3f}  {m['r2_cv']:>7.3f}  {m['oob_r2']:>7.3f}")
lines.append("")
lines.append("Notes:")
lines.append("  rho_train = in-sample Spearman correlation (optimistic)")
lines.append("  rho_cv    = LOFO cross-validated (generalisation estimate)")
lines.append("  OOB_R2    = out-of-bag R² from RandomForest internal estimate")
lines.append("  Log targets (p, s, no3): trained on log1p, metrics in original scale")
lines.append("  Sigma filter: ±3σ per target (pH unfiltered)")
(OUT_DIR / "rf_report.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"  Saved: rf_report.txt")

# ─── 6. Visualisation: CV summary figure ─────────────────────────
print("\nRendering rf_cv_summary.png ...")

fig = plt.figure(figsize=(22, 16), facecolor="#0a0a0a")
gs = gridspec.GridSpec(
    2, len(TARGETS), figure=fig,
    left=0.06, right=0.97, top=0.90, bottom=0.06,
    hspace=0.45, wspace=0.30,
)

# Row 0: ρ_train vs ρ_cv bar chart
ax_bar = fig.add_subplot(gs[0, :])
ax_bar.set_facecolor("#111111")

x  = np.arange(len(TARGETS))
w  = 0.32
rho_trains = [all_metrics[t]["rho_train"] for t in TARGETS]
rho_cvs    = [all_metrics[t]["rho_cv"]    for t in TARGETS]
ci_los     = [all_metrics[t]["ci_lo"]     for t in TARGETS]
ci_his     = [all_metrics[t]["ci_hi"]     for t in TARGETS]
colors     = [CHEM_CMAPS[t] for t in TARGETS]

bars_tr = ax_bar.bar(x - w/2, [abs(r) for r in rho_trains],
                     width=w, color=colors, alpha=0.50,
                     edgecolor="#444", linewidth=0.6, label="ρ_train (in-sample)")
bars_cv = ax_bar.bar(x + w/2, [abs(r) for r in rho_cvs],
                     width=w, color=colors, alpha=0.95,
                     edgecolor="white", linewidth=0.7, label="ρ_cv (LOFO-CV)")

# CI error bars on ρ_cv
for i, t in enumerate(TARGETS):
    m = all_metrics[t]
    # Use absolute-scale CI so error bars are correct for negative rho_cv
    lo_err = abs(m["rho_cv"]) - m["ci_lo_abs"]
    hi_err = m["ci_hi_abs"] - abs(m["rho_cv"])
    lo_err = max(0, lo_err)
    hi_err = max(0, hi_err)
    ax_bar.errorbar(x[i] + w/2, abs(m["rho_cv"]),
                    yerr=[[lo_err], [hi_err]],
                    fmt="none", color="white", capsize=4, linewidth=1.2)

for i, (t, bar) in enumerate(zip(TARGETS, bars_cv)):
    m = all_metrics[t]
    ax_bar.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.015,
                f"{m['rho_cv']:+.3f}",
                ha="center", va="bottom", color="white", fontsize=8, fontweight="bold")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels([CHEM_LABELS[t] for t in TARGETS], color="white", fontsize=9)
ax_bar.set_ylabel("|Spearman ρ|", color="white", fontsize=9)
ax_bar.set_ylim(0, 1.05)
ax_bar.axhline(0.5, color="#555", lw=0.8, ls=":")
ax_bar.axhline(0.7, color="#888", lw=0.8, ls=":")
ax_bar.tick_params(colors="white")
for sp in ax_bar.spines.values():
    sp.set_edgecolor("#333")
ax_bar.legend(facecolor="#1a1a1a", edgecolor="#555",
              labelcolor="white", fontsize=9, loc="upper right")
ax_bar.set_title(
    "Random Forest LOFO-CV  |  ρ_train vs ρ_cv (|Spearman|)  |  error bars = 95% bootstrap CI",
    color="white", fontsize=10, fontweight="bold", pad=8,
)

# Row 1: OOF scatter per element
for ti, tgt in enumerate(TARGETS):
    ax = fig.add_subplot(gs[1, ti])
    ax.set_facecolor("#111111")

    sub = oof_df[oof_df["element"] == tgt]
    if len(sub) == 0:
        continue

    ax.scatter(sub["y_true"], sub["y_pred"],
               c=CHEM_CMAPS[tgt], s=8, alpha=0.55,
               linewidths=0, zorder=3)

    lo = min(sub["y_true"].min(), sub["y_pred"].min())
    hi = max(sub["y_true"].max(), sub["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], color="#777", lw=1.0, ls="--", zorder=2)

    m = all_metrics[tgt]
    ax.text(0.05, 0.95,
            f"ρ_cv={m['rho_cv']:+.3f}\nn={m['n_oof']}",
            transform=ax.transAxes, ha="left", va="top",
            color="white", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#1a1a1a",
                      edgecolor="#555", alpha=0.92))

    ax.set_xlabel(f"True {CHEM_LABELS[tgt]}", color="white", fontsize=7.5)
    ax.set_ylabel(f"Predicted", color="white", fontsize=7.5)
    ax.set_title(CHEM_LABELS[tgt], color=CHEM_CMAPS[tgt],
                 fontsize=8, fontweight="bold")
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

fig.suptitle(
    "Random Forest  |  LOFO-CV (Leave-One-Field-Out)  |  n_estimators=500  |  "
    "sigma-filtered (±3σ)  |  log1p: P, S, NO3",
    color="white", fontsize=11, fontweight="bold", y=0.96,
)

out_sum = OUT_DIR / "rf_cv_summary.png"
fig.savefig(out_sum, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig)
print(f"  Saved: {out_sum.name}")

# ─── 7. Individual scatter plots ─────────────────────────────────
print("Rendering individual OOF scatter plots ...")
for tgt in TARGETS:
    sub = oof_df[oof_df["element"] == tgt]
    if len(sub) == 0:
        continue
    m = all_metrics[tgt]

    fig_s, ax_s = plt.subplots(figsize=(7, 6))
    fig_s.patch.set_facecolor("#0a0a0a")
    ax_s.set_facecolor("#111111")

    sc = ax_s.scatter(sub["y_true"], sub["y_pred"],
                      c=sub["y_true"], cmap="viridis",
                      s=20, alpha=0.65, linewidths=0, zorder=3)
    lo = min(sub["y_true"].min(), sub["y_pred"].min())
    hi = max(sub["y_true"].max(), sub["y_pred"].max())
    ax_s.plot([lo, hi], [lo, hi], color="#e05c4a", lw=1.5, ls="--", zorder=2)

    cb = fig_s.colorbar(sc, ax=ax_s, shrink=0.8, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)
    cb.outline.set_edgecolor("#333")
    cb.set_label(CHEM_LABELS[tgt], color="white", fontsize=8)

    ax_s.set_xlabel(f"True {CHEM_LABELS[tgt]}", color="white", fontsize=9)
    ax_s.set_ylabel("Predicted (RF, LOFO-CV)", color="white", fontsize=9)
    ax_s.tick_params(colors="white", labelsize=8)
    for sp in ax_s.spines.values():
        sp.set_edgecolor("#333")

    ci_str = f"[{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}]"
    ax_s.text(0.04, 0.97,
              f"ρ_cv = {m['rho_cv']:+.3f}  95%CI: {ci_str}\n"
              f"RMSE = {m['rmse_cv']:.3f}   R² = {m['r2_cv']:.3f}\n"
              f"n = {m['n_oof']}   n_features = {m['n_features']}",
              transform=ax_s.transAxes, ha="left", va="top",
              color="white", fontsize=8,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                        edgecolor="#666", alpha=0.93))

    ax_s.set_title(
        f"{CHEM_LABELS[tgt]}  —  RF LOFO-CV OOF scatter",
        color="white", fontsize=10, fontweight="bold",
    )
    fig_s.tight_layout(pad=0.5)
    out_sc = OUT_DIR / f"rf_scatter_{tgt}.png"
    fig_s.savefig(out_sc, dpi=160, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig_s)
    print(f"  Saved: {out_sc.name}")

# ─── 8. Feature importance bar chart per element ─────────────────
print("Rendering feature importance bar charts ...")
for tgt in TARGETS:
    feats = selected_features.get(tgt, [])
    if not feats:
        continue
    feat_csv = OUT_DIR / f"rf_feature_selected_{tgt}.csv"
    if not feat_csv.exists():
        continue
    fdf = pd.read_csv(feat_csv).head(20)

    fig_f, ax_f = plt.subplots(figsize=(10, 7))
    fig_f.patch.set_facecolor("#0a0a0a")
    ax_f.set_facecolor("#111111")

    y_pos = np.arange(len(fdf))
    ax_f.barh(y_pos, fdf["perm_importance"],
              color=CHEM_CMAPS[tgt], alpha=0.85, edgecolor="#222", linewidth=0.4)
    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels(fdf["feature"], fontsize=7, color="white")
    ax_f.invert_yaxis()
    ax_f.set_xlabel("Permutation importance (mean decrease in score)", color="white", fontsize=8)
    ax_f.tick_params(axis="x", colors="white", labelsize=8)
    for sp in ax_f.spines.values():
        sp.set_edgecolor("#333")

    m = all_metrics[tgt]
    ax_f.set_title(
        f"Feature importance: {CHEM_LABELS[tgt]}  |  "
        f"top-20 of {m['n_features']} selected  |  ρ_cv={m['rho_cv']:+.3f}",
        color="white", fontsize=9, fontweight="bold",
    )
    fig_f.tight_layout(pad=0.5)
    out_fi = OUT_DIR / f"rf_importance_{tgt}.png"
    fig_f.savefig(out_fi, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig_f)
    print(f"  Saved: {out_fi.name}")

# ─── 9. Final console summary ────────────────────────────────────
print()
print("=" * 70)
print("RF TRAINING SUMMARY")
print("=" * 70)
print(f"  {'Element':<8}  {'n':>5}  {'n_feat':>6}  "
      f"{'rho_train':>10}  {'rho_cv':>8}  {'95% CI':>16}  "
      f"{'RMSE_cv':>8}  {'OOB_R2':>7}")
print("  " + "-" * 80)
for t in TARGETS:
    m = all_metrics[t]
    ci_str = f"[{m['ci_lo']:+.3f},{m['ci_hi']:+.3f}]"
    flag = " !" if abs(m["rho_train"]) - abs(m["rho_cv"]) > 0.20 else ""
    print(f"  {t:<8}  {m['n_train']:>5}  {m['n_features']:>6}  "
          f"{m['rho_train']:>+10.3f}  {m['rho_cv']:>+8.3f}  {ci_str:>16}  "
          f"{m['rmse_cv']:>8.3f}  {m['oob_r2']:>7.3f}{flag}")
print()
print("  ! = large train/cv gap (>0.20) — possible overfitting")
print(f"\nModels saved to: {MODEL_DIR}")
print(f"Reports saved to: {OUT_DIR}")
print(f"\nNext step: python approximated/rf_pixel_maps.py")
