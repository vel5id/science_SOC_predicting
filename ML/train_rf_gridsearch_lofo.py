"""
train_rf_gridsearch_lofo.py
===========================
RF with per-fold MDI feature selection + GridSearchCV
under Farm-LOFO-CV (20 folds).

For each fold (= held-out farm):
  1. Train a quick RF to compute MDI importances on training data ONLY
  2. Select top-K features (K=15 by default)
  3. Run GridSearchCV (nested GroupKFold by field_name) to tune HPs
  4. Predict on the held-out farm

Records:
  - Per-fold: selected features, best hyperparameters
  - Overall: OOF metrics (rho, R2, RMSE, MAE, RPD, CCC)
  - Summary: hyperparameter ranges, feature stability (IoU)

Output:
  ML/results/rf_gridsearch_lofo/
    {target}_oof_predictions.csv
    {target}_fold_details.json
    {target}_metrics.json
    summary_all_targets.json
    hyperparameter_ranges.csv
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "features" / "master_dataset.csv"
SELECTED_DIR = ROOT / "data" / "features" / "selected"
OUT_DIR = ROOT / "ML" / "results" / "rf_gridsearch_lofo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["ph", "soc", "no3", "p", "k", "s"]
TARGET_LABELS = {
    "ph": "pH", "soc": "SOC", "no3": "NO3",
    "p": "P2O5", "k": "K2O", "s": "S",
}
FIELD_COL = "field_name"
FARM_COL = "farm"
SEED = 42
TOP_K = 15  # features per fold

# GridSearchCV grid
RF_GRID = {
    "n_estimators": [300, 500, 800],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [2, 3, 5],
}


# -- helpers ---------------------------------------------------------------
def compute_ccc(y_true, y_pred):
    """Lin's Concordance Correlation Coefficient."""
    mean_t, mean_p = np.mean(y_true), np.mean(y_pred)
    var_t = np.var(y_true, ddof=1)
    var_p = np.var(y_pred, ddof=1)
    cov_tp = np.cov(y_true, y_pred, ddof=1)[0, 1]
    return float(2 * cov_tp / (var_t + var_p + (mean_t - mean_p) ** 2))


def compute_rpd(y_true, rmse):
    """Ratio of Performance to Deviation."""
    sd = float(np.std(y_true, ddof=1))
    return sd / rmse if rmse > 0 else float("inf")


def get_candidate_features(df):
    """All numeric columns except targets and metadata."""
    exclude_prefixes = [
        "id", "year", "farm", "field", "grid", "centroid",
        "geometry", "protocol", "analysis", "sampling",
        "ph", "soc", "no3", "hu", "p", "k", "s",
    ]
    soilgrids_prefixes = ["sg_", "soilgrid"]

    candidates = []
    for col in df.select_dtypes(include="number").columns:
        col_lower = col.lower()
        skip = False
        for pref in exclude_prefixes:
            if col_lower == pref or col_lower.startswith(pref + "_"):
                if pref == "s" and col.startswith("s2"):
                    continue
                if pref == "p" and col.startswith(("pca", "spectral_PCA")):
                    continue
                if pref == "k" and col != "k":
                    continue
                if col in {"ph", "soc", "no3", "hu", "p", "k", "s"}:
                    skip = True; break
                if pref in ("id", "year", "farm", "field", "grid",
                            "centroid", "geometry", "protocol",
                            "analysis", "sampling"):
                    skip = True; break
        for pref in soilgrids_prefixes:
            if col_lower.startswith(pref):
                skip = True; break
        if not skip:
            candidates.append(col)
    return candidates


def mdi_feature_selection(X_train, y_train, feature_names, top_k=15):
    """Quick MDI-based feature selection."""
    rf = RandomForestRegressor(
        n_estimators=300, max_features="sqrt",
        min_samples_leaf=3, random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:top_k]
    selected = [feature_names[i] for i in idx]
    imp_dict = {feature_names[i]: float(imp[i]) for i in idx}
    return selected, imp_dict


def compute_iou(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


# -- main per-target routine -----------------------------------------------
def train_target(df, target):
    sub = df.dropna(subset=[target]).reset_index(drop=True)
    farms = sub[FARM_COL].values
    fields = sub[FIELD_COL].values
    unique_farms = np.unique(farms)
    n_farms = len(unique_farms)

    # Reference features (for IoU comparison)
    ref_path = SELECTED_DIR / f"{target}_best_features.txt"
    with open(ref_path) as f:
        ref_features = [l.strip() for l in f if l.strip()]

    # Candidate pool
    all_candidates = get_candidate_features(sub)
    all_candidates = sorted(set(all_candidates) | set(ref_features))
    all_candidates = [c for c in all_candidates if c in sub.columns]

    X_full = sub[all_candidates].values
    y_full = sub[target].values

    oof_preds = np.full(len(sub), np.nan)
    fold_details = []
    all_selected_features = []
    all_best_params = []

    t0_total = time.time()

    for fold_i, farm_name in enumerate(tqdm(
        unique_farms,
        desc=f"  {TARGET_LABELS[target]:>4s}",
        unit="farm",
        leave=True,
        ncols=90,
    )):
        t0 = time.time()

        test_mask = (farms == farm_name)
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(test_idx) == 0:
            continue

        X_tr_raw = X_full[train_idx].copy()
        X_te_raw = X_full[test_idx].copy()
        y_tr = y_full[train_idx]

        # Median imputation (train-derived)
        medians = np.nanmedian(X_tr_raw, axis=0)
        medians[np.isnan(medians)] = 0.0
        for c in range(X_tr_raw.shape[1]):
            X_tr_raw[np.isnan(X_tr_raw[:, c]), c] = medians[c]
            X_te_raw[np.isnan(X_te_raw[:, c]), c] = medians[c]

        # Step 1 -- MDI selection on training data only
        sel_feats, feat_imp = mdi_feature_selection(
            X_tr_raw, y_tr, all_candidates, top_k=TOP_K)
        all_selected_features.append(sel_feats)

        feat_idx = [all_candidates.index(f) for f in sel_feats]
        X_tr = X_tr_raw[:, feat_idx]
        X_te = X_te_raw[:, feat_idx]

        # Step 2 -- nested GridSearchCV, grouped by field_name
        train_fields = fields[train_idx]
        n_groups = len(np.unique(train_fields))
        n_cv = min(5, n_groups)
        gkf = GroupKFold(n_splits=n_cv)

        gs = GridSearchCV(
            estimator=RandomForestRegressor(random_state=SEED, n_jobs=-1),
            param_grid=RF_GRID,
            cv=gkf.split(X_tr, y_tr, train_fields),
            scoring="r2",
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gs.fit(X_tr, y_tr)
        best_params = gs.best_params_
        all_best_params.append(best_params)

        # Step 3 -- predict held-out farm
        oof_preds[test_idx] = gs.best_estimator_.predict(X_te)

        fold_details.append({
            "fold": fold_i + 1,
            "farm": str(farm_name),
            "n_fields": int(len(np.unique(fields[test_idx]))),
            "n_test": int(len(test_idx)),
            "selected_features": sel_feats,
            "best_params": {k: str(v) for k, v in best_params.items()},
            "cv_best_r2": round(float(gs.best_score_), 4),
            "time_s": round(time.time() - t0, 1),
        })

    total_time = time.time() - t0_total

    # -- OOF metrics -------------------------------------------------------
    valid = np.isfinite(oof_preds) & np.isfinite(y_full)
    yv, pv = y_full[valid], oof_preds[valid]

    rho, _ = spearmanr(yv, pv)
    r2 = r2_score(yv, pv)
    rmse = float(np.sqrt(mean_squared_error(yv, pv)))
    mae = float(mean_absolute_error(yv, pv))
    rpd = compute_rpd(yv, rmse)
    ccc = compute_ccc(yv, pv)

    # -- Feature stability -------------------------------------------------
    ious = [compute_iou(a, b)
            for i, a in enumerate(all_selected_features)
            for b in all_selected_features[i + 1:]]
    mean_iou = float(np.mean(ious))

    feat_freq = Counter()
    for feats in all_selected_features:
        feat_freq.update(feats)
    top_frequent = feat_freq.most_common(TOP_K)

    ref_ious = [compute_iou(f, ref_features) for f in all_selected_features]
    mean_ref_iou = float(np.mean(ref_ious))

    # -- HP ranges ---------------------------------------------------------
    hp_ranges = {}
    for key in RF_GRID:
        vals = [str(p[key]) for p in all_best_params]
        cnt = Counter(vals)
        mc = cnt.most_common(1)[0]
        hp_ranges[key] = {
            "unique_values": sorted(set(vals)),
            "counts": dict(cnt),
            "most_common": mc[0],
            "most_common_pct": round(100 * mc[1] / n_farms, 1),
        }

    tqdm.write(
        f"  {TARGET_LABELS[target]:>4s}: rho={rho:.4f}  R2={r2:.4f}  "
        f"RMSE={rmse:.4f}  RPD={rpd:.3f}  CCC={ccc:.4f}  "
        f"IoU={mean_iou:.3f}  ({total_time:.0f}s)"
    )

    metrics = {
        "target": TARGET_LABELS[target],
        "strategy": "Farm-LOFO-CV",
        "rho": round(rho, 4),
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "RPD": round(rpd, 3),
        "CCC": round(ccc, 4),
        "n_samples": int(valid.sum()),
        "n_folds": n_farms,
        "feature_stability": {
            "mean_pairwise_iou": round(mean_iou, 3),
            "min_iou": round(float(np.min(ious)), 3),
            "max_iou": round(float(np.max(ious)), 3),
            "mean_ref_iou": round(mean_ref_iou, 3),
            "top_features": {f: c for f, c in top_frequent},
        },
        "hyperparameter_ranges": hp_ranges,
        "total_time_s": round(total_time, 1),
    }

    # -- persist -----------------------------------------------------------
    oof_df = sub.copy()
    oof_df["oof_pred"] = oof_preds
    oof_df.to_csv(OUT_DIR / f"{target}_oof_predictions.csv", index=False)

    with open(OUT_DIR / f"{target}_fold_details.json", "w") as f:
        json.dump(fold_details, f, indent=2, default=str)

    with open(OUT_DIR / f"{target}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


# -- entry point -----------------------------------------------------------
def main():
    print("=" * 70)
    print("  RF  per-fold MDI + GridSearchCV  --  Farm-LOFO-CV (20 farms)")
    print("=" * 70)

    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"  Dataset: {len(df)} rows, {df[FIELD_COL].nunique()} fields, "
          f"{df[FARM_COL].nunique()} farms")
    print(f"  Grid: {RF_GRID}")
    print()

    all_metrics = []
    hp_summary = []

    for target in TARGETS:
        out_path = OUT_DIR / f"{target}_metrics.json"
        if out_path.exists():
            with open(out_path) as f:
                m = json.load(f)
            # Skip only if computed under Farm-LOFO
            if m.get("strategy") == "Farm-LOFO-CV":
                all_metrics.append(m)
                print(f"  >> {TARGET_LABELS[target]:>4s}: already done  "
                      f"rho={m['rho']:.4f}  R2={m['R2']:.4f}  RPD={m['RPD']:.3f}")
                continue
        metrics = train_target(df, target)
        all_metrics.append(metrics)

    # HP summary
    for m in all_metrics:
        for key, info in m["hyperparameter_ranges"].items():
            hp_summary.append({
                "target": m["target"],
                "hyperparameter": key,
                "values_seen": ", ".join(info["unique_values"]),
                "most_common": info["most_common"],
                "most_common_pct": info["most_common_pct"],
            })

    with open(OUT_DIR / "summary_all_targets.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    hp_df = pd.DataFrame(hp_summary)
    hp_df.to_csv(OUT_DIR / "hyperparameter_ranges.csv", index=False)

    # Final table
    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY -- RF GridSearchCV + per-fold MDI  (Farm-LOFO)")
    print(f"{'=' * 70}")
    print(f"  {'Target':<8} {'rho':>8} {'R2':>8} {'RMSE':>10} "
          f"{'RPD':>8} {'CCC':>8} {'IoU':>8}")
    print(f"  {'-' * 58}")
    for m in all_metrics:
        print(f"  {m['target']:<8} {m['rho']:>8.4f} {m['R2']:>8.4f} "
              f"{m['RMSE']:>10.4f} {m['RPD']:>8.3f} {m['CCC']:>8.4f} "
              f"{m['feature_stability']['mean_pairwise_iou']:>8.3f}")

    print(f"\n  HP ranges:")
    for _, row in hp_df.iterrows():
        print(f"  {row['target']:<6} {row['hyperparameter']:<20} "
              f"seen: [{row['values_seen']}]  "
              f"dominant: {row['most_common']} ({row['most_common_pct']}%)")

    print(f"\n  Results -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
