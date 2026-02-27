"""
rf_temporal_ablation.py
=======================
Temporal leakage ablation experiment for RF pipeline.

Problem:
  75% of soil samples were collected in SPRING (April-May).
  The RF model uses features from summer/late_summer/autumn of the SAME year,
  which is data that did NOT EXIST at the time of sampling.
  This constitutes temporal (chronological) leakage.

Experiment design — 4 configurations:
  A) BASELINE (leaky):     All 512 features (reproduces current pipeline)
  B) SPRING-CLEAN:         Only spring + topo + climate features (90 features)
                           → no temporal leakage for ANY sample
  C) TOPO+CLIMATE ONLY:    Only 12 topographic + climate features
                           → baseline: does satellite data add anything?
  D) AUTUMN-SUBSET:        Only autumn-sampled probes (Sep-Nov), ALL features
                           → no leakage because features precede/coincide with sampling

All configurations use identical LOFO-CV and RF hyperparameters.
Metrics: R², Spearman ρ, RMSE, MAE — all in original scale.

Output:
  rf/ablation_temporal_results.csv   — per-target metrics for all configs
  rf/ablation_temporal_report.txt    — text report
  rf/ablation_temporal_summary.png   — comparison figure

Run: python approximated/rf_temporal_ablation.py
"""

import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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

# ─── Paths ──────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
RF_CSV      = BASE / "data" / "features" / "rf_dataset.csv"
MASTER_CSV  = BASE / "data" / "features" / "master_dataset.csv"
OUT_DIR     = BASE / "math_statistics" / "output" / "rf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS     = ["ph", "k", "p", "hu", "s", "no3"]
LOG_TARGETS = {"p", "s", "no3"}

CHEM_LABELS = {
    "ph":  "pH",       "k":   "K, mg/kg",   "p":   "P, mg/kg",
    "hu":  "Humus, %", "s":   "S, mg/kg",   "no3": "NO3, mg/kg",
}
CHEM_CMAPS = {
    "ph": "#e05c4a", "k": "#e0a44a", "p": "#6abf69",
    "hu": "#8d6e63", "s": "#4ab5e0", "no3": "#b44ae0",
}

RF_PARAMS = dict(
    n_estimators=500, max_features="sqrt",
    min_samples_leaf=3, max_depth=None,
    n_jobs=-1, random_state=42, oob_score=True,
)
RF_FAST = dict(
    n_estimators=100, max_features="sqrt",
    min_samples_leaf=3, n_jobs=-1, random_state=42,
)

CORR_THRESHOLD = 0.92
TOP_FEATS      = 40
BOOTSTRAP_N    = 500
BOOTSTRAP_SEED = 42

META_COLS = {"year", "farm", "field_name", "centroid_lon", "centroid_lat"}

# ─── Feature classification ────────────────────────────────────────

def classify_features(feat_cols):
    """Classify features by temporal requirement.

    Returns dict with keys:
      'season_independent': topo + climate (always available)
      'spring':   features that only need spring satellite data
      'summer':   features that need summer data
      'late_summer': features that need late_summer data
      'autumn':   features that need autumn data
      'cross_season': deltas, ranges, temporal stats (need multiple seasons)
    """
    cats = {
        "season_independent": [],
        "spring": [],
        "summer": [],
        "late_summer": [],
        "autumn": [],
        "cross_season": [],
    }

    for c in feat_cols:
        # Topographic and climate — always available
        if c.startswith("topo_") or c.startswith("climate_"):
            cats["season_independent"].append(c)

        # Cross-season derived features
        elif c.startswith("delta_") or c.startswith("range_") or c.startswith("ts_"):
            cats["cross_season"].append(c)
        elif c.startswith("glcm_delta_") or c.startswith("glcm_ratio_"):
            if "spr2sum" in c or "spr2aut" in c:
                cats["cross_season"].append(c)
            elif "_summer" in c and "late_summer" not in c:
                cats["summer"].append(c)
            elif "_late_summer" in c:
                cats["late_summer"].append(c)
            elif "_autumn" in c:
                cats["autumn"].append(c)
            elif "_spring" in c:
                cats["spring"].append(c)
            else:
                cats["cross_season"].append(c)
        elif c.startswith("cs_"):
            if "_spring" in c:
                cats["spring"].append(c)
            elif "_summer" in c and "late_summer" not in c:
                cats["summer"].append(c)
            elif "_late_summer" in c:
                cats["late_summer"].append(c)
            elif "_autumn" in c:
                cats["autumn"].append(c)
            else:
                cats["cross_season"].append(c)

        # Regular satellite features with season suffix
        elif "_spring" in c:
            cats["spring"].append(c)
        elif "_summer" in c and "late_summer" not in c:
            cats["summer"].append(c)
        elif "_late_summer" in c:
            cats["late_summer"].append(c)
        elif "_autumn" in c:
            cats["autumn"].append(c)
        else:
            cats["season_independent"].append(c)

    return cats


# ─── Feature selection helpers (from rf_train_cv.py) ────────────────

def stage1_variance_filter(feat_cols, df_sub):
    keep = []
    for c in feat_cols:
        if c not in df_sub.columns:
            continue
        if df_sub[c].isna().mean() > 0.40:
            continue
        if df_sub[c].std() < 1e-4:
            continue
        keep.append(c)
    return keep


def stage2_dedup(feat_cols, df_sub, target_col, threshold=CORR_THRESHOLD):
    if len(feat_cols) <= 1:
        return feat_cols

    X = df_sub[feat_cols].copy()
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

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

    dist_mat = 1.0 - np.abs(rho_mat)
    np.fill_diagonal(dist_mat, 0.0)
    dist_mat = np.clip(dist_mat, 0, None)

    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=(1.0 - threshold), criterion="distance")

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
    imp = SimpleImputer(strategy="median")
    X_full = imp.fit_transform(X_tr)

    n = len(y_tr)
    val_size = max(int(n * 0.20), 1)
    rng = np.random.default_rng(42)
    val_idx = rng.choice(n, val_size, replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)

    rf_fast = RandomForestRegressor(**RF_FAST)
    rf_fast.fit(X_full[train_idx], y_tr[train_idx])

    perm = permutation_importance(rf_fast, X_full[val_idx], y_tr[val_idx],
                                  n_repeats=5, random_state=42, n_jobs=-1)
    importances = perm.importances_mean
    actual_top = min(top_n, len(feat_cols))
    top_idx = np.argsort(importances)[::-1][:actual_top]
    return [feat_cols[i] for i in top_idx], importances[top_idx]


# ─── LOFO-CV runner ────────────────────────────────────────────────

def run_lofo_cv(df_sub, feat_list, tgt, config_name):
    """Run LOFO-CV for a given feature list and target.

    Returns dict with metrics.
    """
    train_col = f"log_{tgt}" if tgt in LOG_TARGETS else tgt

    if len(feat_list) == 0:
        return {
            "config": config_name, "element": tgt,
            "n_samples": len(df_sub), "n_features_input": 0,
            "n_features_selected": 0,
            "rho_cv": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "r2_cv": np.nan, "rmse_cv": np.nan, "mae_cv": np.nan,
            "oob_r2": np.nan,
        }

    # ── Feature selection (3-stage, identical to main pipeline) ───
    feats_s1 = stage1_variance_filter(feat_list, df_sub)
    if len(feats_s1) < 2:
        return {
            "config": config_name, "element": tgt,
            "n_samples": len(df_sub), "n_features_input": len(feat_list),
            "n_features_selected": len(feats_s1),
            "rho_cv": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "r2_cv": np.nan, "rmse_cv": np.nan, "mae_cv": np.nan,
            "oob_r2": np.nan,
        }

    feats_s2 = stage2_dedup(feats_s1, df_sub, train_col)

    imp_data = SimpleImputer(strategy="median")
    X_s2 = imp_data.fit_transform(df_sub[feats_s2])
    y_s2 = df_sub[train_col].values

    actual_top = min(TOP_FEATS, len(feats_s2))
    feats_final, _ = stage3_rf_importance(feats_s2, X_s2, y_s2, top_n=actual_top)

    # ── LOFO-CV ───────────────────────────────────────────────────
    fields = df_sub["field_name"].unique()
    oof_true, oof_pred = [], []

    for field in fields:
        test_mask  = df_sub["field_name"] == field
        train_mask = ~test_mask

        if train_mask.sum() < 20 or test_mask.sum() == 0:
            continue

        X_tr_raw = df_sub.loc[train_mask, feats_final].values
        y_tr     = df_sub.loc[train_mask, train_col].values
        X_te_raw = df_sub.loc[test_mask,  feats_final].values
        y_te_orig= df_sub.loc[test_mask,  tgt].values

        imp_lofo = SimpleImputer(strategy="median")
        imp_lofo.fit(X_tr_raw)
        X_tr = imp_lofo.transform(X_tr_raw)
        X_te = imp_lofo.transform(X_te_raw)

        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_tr, y_tr)
        y_hat_log = rf.predict(X_te)

        if tgt in LOG_TARGETS:
            y_hat = np.expm1(y_hat_log)
        else:
            y_hat = y_hat_log

        oof_true.extend(y_te_orig.tolist())
        oof_pred.extend(y_hat.tolist())

    oof_true = np.array(oof_true)
    oof_pred = np.array(oof_pred)

    if len(oof_true) < 10:
        return {
            "config": config_name, "element": tgt,
            "n_samples": len(df_sub), "n_features_input": len(feat_list),
            "n_features_selected": len(feats_final),
            "rho_cv": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "r2_cv": np.nan, "rmse_cv": np.nan, "mae_cv": np.nan,
            "oob_r2": np.nan,
        }

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
    ci_lo = float(np.percentile(boot_rhos, 2.5)) if len(boot_rhos) > 0 else np.nan
    ci_hi = float(np.percentile(boot_rhos, 97.5)) if len(boot_rhos) > 0 else np.nan

    # OOB from full model
    X_full_raw = df_sub[feats_final].values
    y_full = df_sub[train_col].values
    imp_full = SimpleImputer(strategy="median")
    X_full = imp_full.fit_transform(X_full_raw)
    rf_full = RandomForestRegressor(**RF_PARAMS)
    rf_full.fit(X_full, y_full)
    oob_r2 = rf_full.oob_score_

    return {
        "config": config_name, "element": tgt,
        "n_samples": len(df_sub), "n_features_input": len(feat_list),
        "n_features_selected": len(feats_final),
        "rho_cv": rho_cv, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "r2_cv": r2_cv, "rmse_cv": rmse_cv, "mae_cv": mae_cv,
        "oob_r2": oob_r2,
        "n_oof": n_oof,
        "selected_features": feats_final,
    }


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
print("=" * 72)
print("TEMPORAL LEAKAGE ABLATION EXPERIMENT")
print("=" * 72)

# ─── 1. Load data ──────────────────────────────────────────────────
print("\n[1] Loading datasets ...")
df_rf = pd.read_csv(RF_CSV, low_memory=False)
df_master = pd.read_csv(MASTER_CSV, low_memory=False)

# Add sampling_date from master_dataset
if "sampling_date" in df_master.columns and "sampling_date" not in df_rf.columns:
    df_rf["sampling_date"] = df_master["sampling_date"].values
    print(f"  Added sampling_date from master_dataset.csv")

print(f"  RF dataset: {df_rf.shape}")
print(f"  Unique fields: {df_rf['field_name'].nunique()}")

# ─── 2. Parse sampling dates & classify ───────────────────────────
print("\n[2] Parsing sampling dates ...")
dates = pd.to_datetime(df_rf["sampling_date"], format="%d.%m.%Y", errors="coerce")
df_rf["sampling_month"] = dates.dt.month

# Classify probes by season
spring_mask = df_rf["sampling_month"].isin([3, 4, 5])
summer_mask = df_rf["sampling_month"].isin([6, 7, 8])
autumn_mask = df_rf["sampling_month"].isin([9, 10, 11])

n_total = len(df_rf)
print(f"  Spring probes (Mar-May):  {spring_mask.sum():>5}  ({100*spring_mask.mean():.1f}%)")
print(f"  Summer probes (Jun-Aug):  {summer_mask.sum():>5}  ({100*summer_mask.mean():.1f}%)")
print(f"  Autumn probes (Sep-Nov):  {autumn_mask.sum():>5}  ({100*autumn_mask.mean():.1f}%)")

# ─── 3. Classify features ────────────────────────────────────────
print("\n[3] Classifying features by temporal requirement ...")

feat_base = [c for c in df_rf.columns
             if c not in META_COLS
             and c not in set(TARGETS)
             and not c.startswith("log_")
             and not c.startswith("mask_")
             and c not in {"cu","mo","fe","zn","mg","mn","soc","b","ca","na"}
             and c != "sampling_date"
             and c != "sampling_month"]

feat_cats = classify_features(feat_base)

print(f"  Season-independent (topo+climate): {len(feat_cats['season_independent'])}")
print(f"  Spring-only:                       {len(feat_cats['spring'])}")
print(f"  Needs summer:                      {len(feat_cats['summer'])}")
print(f"  Needs late_summer:                 {len(feat_cats['late_summer'])}")
print(f"  Needs autumn:                      {len(feat_cats['autumn'])}")
print(f"  Cross-season (delta/range/ts):     {len(feat_cats['cross_season'])}")

# Define feature sets for each configuration
feats_ALL    = feat_base  # A: all features
feats_SPRING = feat_cats["season_independent"] + feat_cats["spring"]  # B: spring-clean
feats_TOPO   = feat_cats["season_independent"]  # C: topo+climate only

print(f"\n  Config A (BASELINE):       {len(feats_ALL)} features")
print(f"  Config B (SPRING-CLEAN):   {len(feats_SPRING)} features")
print(f"  Config C (TOPO+CLIMATE):   {len(feats_TOPO)} features")
print(f"  Config D (AUTUMN-SUBSET):  {len(feats_ALL)} features, "
      f"{autumn_mask.sum()} samples")

# ─── 4. Run experiments ──────────────────────────────────────────
print("\n[4] Running LOFO-CV experiments ...")
print("    (This may take 15-30 minutes depending on CPU)")

all_results = []

CONFIGS = [
    ("A_baseline_leaky",  feats_ALL,    None,        "All features (with leakage)"),
    ("B_spring_clean",    feats_SPRING, None,        "Spring + topo + climate only"),
    ("C_topo_climate",    feats_TOPO,   None,        "Topo + climate only (no satellite)"),
    ("D_autumn_subset",   feats_ALL,    autumn_mask, "Autumn probes only, all features"),
]

for config_id, feat_list, subset_mask, description in CONFIGS:
    print(f"\n{'='*60}")
    print(f"  CONFIG: {config_id}")
    print(f"  {description}")
    print(f"  Features: {len(feat_list)}")
    print(f"{'='*60}")

    for tgt in TARGETS:
        train_col = f"log_{tgt}" if tgt in LOG_TARGETS else tgt
        mask_col = f"mask_{tgt}"

        # Subset by target validity mask
        valid = df_rf[mask_col].astype(bool)
        if subset_mask is not None:
            valid = valid & subset_mask

        df_sub = df_rf[valid].copy()

        if len(df_sub) < 30:
            print(f"  {tgt.upper()}: only {len(df_sub)} samples, skipping")
            all_results.append({
                "config": config_id, "element": tgt,
                "description": description,
                "n_samples": len(df_sub), "n_features_input": len(feat_list),
                "n_features_selected": 0,
                "rho_cv": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "r2_cv": np.nan, "rmse_cv": np.nan, "mae_cv": np.nan,
                "oob_r2": np.nan, "n_oof": 0,
            })
            continue

        n_fields = df_sub["field_name"].nunique()
        print(f"\n  {tgt.upper()} ({CHEM_LABELS[tgt]}):  "
              f"n={len(df_sub)}, fields={n_fields}", flush=True)

        result = run_lofo_cv(df_sub, feat_list, tgt, config_id)
        result["description"] = description

        # Remove selected_features list for CSV (too long)
        sel_feats = result.pop("selected_features", [])

        all_results.append(result)

        rho = result.get("rho_cv", np.nan)
        r2  = result.get("r2_cv", np.nan)
        rmse = result.get("rmse_cv", np.nan)
        n_sel = result.get("n_features_selected", 0)
        print(f"    rho_cv={rho:+.3f}  R2_cv={r2:.3f}  RMSE={rmse:.3f}  "
              f"features: {len(feat_list)}->{n_sel}")

        # Save selected features for spring-clean config
        if config_id == "B_spring_clean":
            feat_df = pd.DataFrame({"feature": sel_feats})
            feat_df.to_csv(
                OUT_DIR / f"ablation_spring_features_{tgt}.csv", index=False
            )


# ─── 5. Save results ─────────────────────────────────────────────
print("\n\n[5] Saving results ...")

results_df = pd.DataFrame(all_results)
results_csv = OUT_DIR / "ablation_temporal_results.csv"
results_df.to_csv(results_csv, index=False)
print(f"  Saved: {results_csv.name}")


# ─── 6. Generate comparison report ────────────────────────────────
print("\n[6] Generating report ...")

lines = [
    "TEMPORAL LEAKAGE ABLATION REPORT",
    "=" * 72,
    "",
    "PROBLEM STATEMENT:",
    "  75.3% of soil samples were collected in Spring (April-May).",
    "  The RF model used satellite features from Summer/Autumn of the SAME year,",
    "  which constitutes temporal (chronological) leakage: the model uses data",
    "  that did NOT EXIST at the time soil was sampled.",
    "",
    "EXPERIMENT DESIGN:",
    f"  A) BASELINE (leaky):    {len(feats_ALL)} features, all samples",
    f"  B) SPRING-CLEAN:        {len(feats_SPRING)} features (spring + topo + climate only)",
    f"  C) TOPO+CLIMATE ONLY:   {len(feats_TOPO)} features (no satellite data)",
    f"  D) AUTUMN-SUBSET:       {len(feats_ALL)} features, {autumn_mask.sum()} autumn-only samples",
    "",
    "METHODOLOGY:",
    "  - Leave-One-Field-Out CV (LOFO-CV)",
    "  - RF: n_estimators=500, max_features=sqrt, min_samples_leaf=3",
    "  - 3-stage feature selection: variance > corr dedup > permutation importance",
    "  - Log targets: P, S, NO3 (log1p transform, metrics in original scale)",
    "  - Bootstrap 95% CI on Spearman rho (n=500)",
    "",
    "RESULTS:",
    "=" * 72,
    "",
]

# Table header
lines.append(f"{'Config':<22} {'Element':<8} {'n':>5} {'n_feat':>6} "
             f"{'rho_cv':>8} {'95% CI':>16} {'R2_cv':>7} {'RMSE_cv':>8} "
             f"{'OOB_R2':>7}")
lines.append("-" * 100)

for _, row in results_df.iterrows():
    ci_str = (f"[{row['ci_lo']:+.3f},{row['ci_hi']:+.3f}]"
              if not np.isnan(row.get("ci_lo", np.nan)) else "N/A")
    lines.append(
        f"  {row['config']:<20} {row['element']:<8} "
        f"{int(row['n_samples']):>5} {int(row['n_features_selected']):>6} "
        f"{row['rho_cv']:>+8.3f} {ci_str:>16} "
        f"{row['r2_cv']:>7.3f} {row['rmse_cv']:>8.3f} "
        f"{row['oob_r2']:>7.3f}"
    )

lines.append("")
lines.append("=" * 72)
lines.append("")

# Leakage impact summary
lines.append("LEAKAGE IMPACT SUMMARY (per element):")
lines.append("-" * 72)
lines.append(f"{'Element':<10} {'Baseline R2':>12} {'Spring R2':>11} "
             f"{'Delta R2':>9} {'Baseline rho':>13} {'Spring rho':>11} "
             f"{'Delta rho':>10}")
lines.append("-" * 72)

for tgt in TARGETS:
    baseline = results_df[(results_df["config"] == "A_baseline_leaky") &
                          (results_df["element"] == tgt)]
    spring   = results_df[(results_df["config"] == "B_spring_clean") &
                          (results_df["element"] == tgt)]

    if len(baseline) == 0 or len(spring) == 0:
        continue

    b_r2   = baseline.iloc[0]["r2_cv"]
    s_r2   = spring.iloc[0]["r2_cv"]
    b_rho  = baseline.iloc[0]["rho_cv"]
    s_rho  = spring.iloc[0]["rho_cv"]

    d_r2   = b_r2 - s_r2
    d_rho  = abs(b_rho) - abs(s_rho)

    pct = (d_r2 / max(abs(b_r2), 1e-6)) * 100 if not np.isnan(d_r2) else 0

    lines.append(
        f"  {CHEM_LABELS[tgt]:<10} {b_r2:>11.3f} {s_r2:>11.3f} "
        f"{d_r2:>+8.3f} {b_rho:>+12.3f} {s_rho:>+11.3f} "
        f"{d_rho:>+9.3f}"
    )

lines.append("")
lines.append("INTERPRETATION:")
lines.append("  Delta R2 > 0: baseline is inflated by temporal leakage")
lines.append("  Delta R2 ~ 0: satellite features provide genuine signal")
lines.append("  Delta R2 < 0: spring features actually work better (unlikely)")
lines.append("")
lines.append("CONCLUSION FOR SULFUR (S):")
s_base = results_df[(results_df["config"] == "A_baseline_leaky") &
                     (results_df["element"] == "s")]
s_clean = results_df[(results_df["config"] == "B_spring_clean") &
                      (results_df["element"] == "s")]
if len(s_base) > 0 and len(s_clean) > 0:
    b_r2 = s_base.iloc[0]["r2_cv"]
    c_r2 = s_clean.iloc[0]["r2_cv"]
    lines.append(f"  Baseline R2_cv  = {b_r2:.3f}")
    lines.append(f"  Clean R2_cv     = {c_r2:.3f}")
    lines.append(f"  Leakage contrib = {b_r2 - c_r2:.3f}")
    if c_r2 >= 0.5:
        lines.append("  VERDICT: R2 >= 0.5 holds WITHOUT leakage. Claim is valid.")
    elif b_r2 >= 0.5:
        lines.append("  VERDICT: R2 >= 0.5 ONLY with leakage. Claim is NOT valid.")
    else:
        lines.append("  VERDICT: R2 < 0.5 even WITH leakage. Claim does NOT hold.")

lines.append("")
lines.append("RECOMMENDATIONS:")
lines.append("  1. Report SPRING-CLEAN metrics as primary results (no leakage)")
lines.append("  2. Report BASELINE as 'full-season model' (requires post-harvest data)")
lines.append("  3. Clearly state in paper: 'spring probes + full-season features = retrospective'")
lines.append("  4. If R2_clean < 0.5 for S: do NOT claim R2 >= 0.5")

report_path = OUT_DIR / "ablation_temporal_report.txt"
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"  Saved: {report_path.name}")


# ─── 7. Visualization ──────────────────────────────────────────────
print("\n[7] Rendering ablation summary figure ...")

config_labels = {
    "A_baseline_leaky": "A: Baseline\n(all features,\nwith leakage)",
    "B_spring_clean":   "B: Spring-clean\n(spring+topo\n+climate only)",
    "C_topo_climate":   "C: Topo+climate\n(no satellite)",
    "D_autumn_subset":  "D: Autumn subset\n(no leakage,\nsmall n)",
}
config_order = ["A_baseline_leaky", "B_spring_clean", "C_topo_climate", "D_autumn_subset"]
config_colors = ["#e05c4a", "#4ab5e0", "#888888", "#6abf69"]

fig = plt.figure(figsize=(20, 14), facecolor="#0a0a0a")
gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1],
                       left=0.08, right=0.95, top=0.90, bottom=0.08,
                       hspace=0.40)

# ── Row 0: R2_cv comparison ──────────────────────────────────────
ax_r2 = fig.add_subplot(gs[0])
ax_r2.set_facecolor("#111111")

x_pos = np.arange(len(TARGETS))
n_configs = len(config_order)
total_width = 0.75
bar_w = total_width / n_configs

for ci, cfg in enumerate(config_order):
    r2_vals = []
    for tgt in TARGETS:
        row = results_df[(results_df["config"] == cfg) &
                         (results_df["element"] == tgt)]
        if len(row) > 0 and not np.isnan(row.iloc[0]["r2_cv"]):
            r2_vals.append(max(row.iloc[0]["r2_cv"], -0.1))
        else:
            r2_vals.append(0)

    offset = (ci - n_configs/2 + 0.5) * bar_w
    bars = ax_r2.bar(x_pos + offset, r2_vals,
                     width=bar_w * 0.9, color=config_colors[ci],
                     alpha=0.85, edgecolor="#222", linewidth=0.5,
                     label=config_labels[cfg].replace("\n", " "))

    for i, v in enumerate(r2_vals):
        if v != 0:
            ax_r2.text(x_pos[i] + offset, v + 0.01, f"{v:.2f}",
                       ha="center", va="bottom", color="white",
                       fontsize=6.5, fontweight="bold")

ax_r2.set_xticks(x_pos)
ax_r2.set_xticklabels([CHEM_LABELS[t] for t in TARGETS], color="white", fontsize=10)
ax_r2.set_ylabel("R$^2$ (LOFO-CV)", color="white", fontsize=10)
ax_r2.axhline(0.0, color="#444", lw=0.8, ls="-")
ax_r2.axhline(0.5, color="#e05c4a", lw=1.2, ls="--", alpha=0.7, label="R$^2$ = 0.5 threshold")
ax_r2.set_ylim(-0.15, 1.0)
ax_r2.tick_params(colors="white")
for sp in ax_r2.spines.values():
    sp.set_edgecolor("#333")
ax_r2.legend(facecolor="#1a1a1a", edgecolor="#555", labelcolor="white",
             fontsize=7.5, loc="upper right", ncol=3)
ax_r2.set_title("R$^2$ (LOFO-CV)  |  Temporal Leakage Ablation",
                color="white", fontsize=12, fontweight="bold", pad=10)

# ── Row 1: Spearman rho comparison ─────────────────────────────
ax_rho = fig.add_subplot(gs[1])
ax_rho.set_facecolor("#111111")

for ci, cfg in enumerate(config_order):
    rho_vals = []
    ci_lo_vals = []
    ci_hi_vals = []
    for tgt in TARGETS:
        row = results_df[(results_df["config"] == cfg) &
                         (results_df["element"] == tgt)]
        if len(row) > 0 and not np.isnan(row.iloc[0]["rho_cv"]):
            rho_vals.append(abs(row.iloc[0]["rho_cv"]))
            ci_lo_vals.append(abs(row.iloc[0]["rho_cv"]) -
                              abs(row.iloc[0].get("ci_lo", row.iloc[0]["rho_cv"])))
            ci_hi_vals.append(abs(row.iloc[0].get("ci_hi", row.iloc[0]["rho_cv"])) -
                              abs(row.iloc[0]["rho_cv"]))
        else:
            rho_vals.append(0)
            ci_lo_vals.append(0)
            ci_hi_vals.append(0)

    offset = (ci - n_configs/2 + 0.5) * bar_w
    bars = ax_rho.bar(x_pos + offset, rho_vals,
                      width=bar_w * 0.9, color=config_colors[ci],
                      alpha=0.85, edgecolor="#222", linewidth=0.5,
                      label=config_labels[cfg].replace("\n", " "))

    for i, v in enumerate(rho_vals):
        if v != 0:
            ax_rho.text(x_pos[i] + offset, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", color="white",
                        fontsize=6.5, fontweight="bold")

ax_rho.set_xticks(x_pos)
ax_rho.set_xticklabels([CHEM_LABELS[t] for t in TARGETS], color="white", fontsize=10)
ax_rho.set_ylabel("|Spearman rho| (LOFO-CV)", color="white", fontsize=10)
ax_rho.axhline(0.5, color="#e05c4a", lw=1.2, ls="--", alpha=0.7, label="|rho| = 0.5 threshold")
ax_rho.axhline(0.7, color="#888", lw=0.8, ls=":", alpha=0.5)
ax_rho.set_ylim(0, 1.05)
ax_rho.tick_params(colors="white")
for sp in ax_rho.spines.values():
    sp.set_edgecolor("#333")
ax_rho.legend(facecolor="#1a1a1a", edgecolor="#555", labelcolor="white",
              fontsize=7.5, loc="upper right", ncol=3)
ax_rho.set_title("|Spearman rho| (LOFO-CV)  |  Temporal Leakage Ablation",
                 color="white", fontsize=12, fontweight="bold", pad=10)

fig.suptitle(
    "Temporal Leakage Ablation Experiment\n"
    "75% of samples are spring probes using summer/autumn 'future' satellite data",
    color="white", fontsize=13, fontweight="bold", y=0.97,
)

out_fig = OUT_DIR / "ablation_temporal_summary.png"
fig.savefig(out_fig, dpi=160, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig)
print(f"  Saved: {out_fig.name}")


# ─── 8. Final console summary ────────────────────────────────────
print("\n" + "=" * 72)
print("ABLATION EXPERIMENT COMPLETE")
print("=" * 72)
print(f"\n{'Config':<22} {'Element':<8} {'n':>5} {'n_feat':>6} "
      f"{'rho_cv':>8} {'R2_cv':>7} {'RMSE_cv':>8}")
print("-" * 80)

for _, row in results_df.iterrows():
    rho = row['rho_cv'] if not np.isnan(row['rho_cv']) else 0
    r2  = row['r2_cv'] if not np.isnan(row['r2_cv']) else 0
    rmse = row['rmse_cv'] if not np.isnan(row['rmse_cv']) else 0
    print(f"  {row['config']:<20} {row['element']:<8} "
          f"{int(row['n_samples']):>5} {int(row['n_features_selected']):>6} "
          f"{rho:>+8.3f} {r2:>7.3f} {rmse:>8.3f}")

# Highlight S specifically
print(f"\n{'='*72}")
print("SULFUR (S) — KEY RESULTS:")
print(f"{'='*72}")
for cfg in config_order:
    row = results_df[(results_df["config"] == cfg) &
                     (results_df["element"] == "s")]
    if len(row) > 0:
        r = row.iloc[0]
        rho = r['rho_cv'] if not np.isnan(r['rho_cv']) else 0
        r2  = r['r2_cv'] if not np.isnan(r['r2_cv']) else 0
        ci_str = (f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
                  if not np.isnan(r.get('ci_lo', np.nan)) else "N/A")
        verdict = "PASS" if r2 >= 0.5 else "FAIL"
        print(f"  {cfg:<22}  rho={rho:+.3f} {ci_str:>16}  "
              f"R2={r2:.3f}  [{verdict} R2>=0.5]")

print(f"\nOutputs:")
print(f"  {results_csv}")
print(f"  {report_path}")
print(f"  {out_fig}")
print(f"\nDone.")
