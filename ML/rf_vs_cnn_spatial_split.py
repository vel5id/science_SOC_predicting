"""
rf_vs_cnn_spatial_split.py
==========================
Fair comparison: RF vs ConvNeXt on the EXACT same spatial split.

Uses the identical field-based split as train_multiseason_convnext.py:
  - field_name column, seed=42, val=10 fields, test=12 fields
  - train = remaining 59 fields

Trains RF (with GridSearchCV) on 15 MDI-selected tabular features,
evaluates on the same test set, and computes all metrics:
  ρ (Spearman), R², RMSE, MAE, RPD, CCC

Loads ConvNeXt results from convnext_ablation_summary_all.csv and
multiseason_ablation_summary_all.csv for direct comparison.

Output:
  ML/results/rf_vs_cnn_comparison.csv
  ML/results/rf_vs_cnn_comparison.json
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "features" / "master_dataset.csv"
SELECTED_DIR = ROOT / "data" / "features" / "selected"
OUT_DIR = ROOT / "ML" / "results"
OUT_DIR.mkdir(exist_ok=True)

TARGETS = ["ph", "soc", "no3", "p", "k", "s"]
TARGET_LABELS = {"ph": "pH", "soc": "SOC", "no3": "NO3", "p": "P2O5", "k": "K2O", "s": "S"}
SEED = 42

# Same split parameters as train_multiseason_convnext.py
FARM_COL = "field_name"
VAL_COUNT = 10
TEST_COUNT = 12

# RF GridSearchCV parameter grid (compact but meaningful)
RF_GRID = {
    "n_estimators": [300, 500, 800],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [2, 3, 5],
}


def compute_ccc(y_true, y_pred):
    """Lin's Concordance Correlation Coefficient."""
    mean_t = np.mean(y_true)
    mean_p = np.mean(y_pred)
    var_t = np.var(y_true, ddof=1)
    var_p = np.var(y_pred, ddof=1)
    cov_tp = np.cov(y_true, y_pred, ddof=1)[0, 1]
    ccc = (2 * cov_tp) / (var_t + var_p + (mean_t - mean_p) ** 2)
    return ccc


def compute_rpd(y_true, rmse):
    """Ratio of Performance to Deviation."""
    sd_obs = np.std(y_true, ddof=1)
    return sd_obs / rmse if rmse > 0 else np.inf


def load_features(target):
    path = SELECTED_DIR / f"{target}_best_features.txt"
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def load_convnext_results():
    """Load best ConvNeXt results per target from existing CSV files."""
    results = {}

    # Single-season ConvNeXt (convnext_ablation_summary_all.csv)
    single_path = OUT_DIR / "convnext_ablation_summary_all.csv"
    if single_path.exists():
        df = pd.read_csv(single_path)
        for target in TARGETS:
            key = "HU" if target == "soc" else target.upper()
            sub = df[df["Target"] == key]
            if len(sub) > 0:
                best = sub.loc[sub["R2"].idxmax()]
                results[f"{target}_single"] = {
                    "config": best["Features"],
                    "R2": best["R2"],
                    "RMSE": best["RMSE"],
                }

    # Multi-season ConvNeXt (multiseason_ablation_summary_all.csv)
    multi_path = OUT_DIR / "multiseason_ablation_summary_all.csv"
    if multi_path.exists():
        df = pd.read_csv(multi_path)
        for target in TARGETS:
            key = "HU" if target == "soc" else target.upper()
            sub = df[df["Target"] == key]
            if len(sub) > 0:
                best = sub.loc[sub["R2"].idxmax()]
                results[f"{target}_multi"] = {
                    "config": best["Features"],
                    "R2": best["R2"],
                    "RMSE": best["RMSE"],
                }

    return results


def main():
    print("=" * 70)
    print("  FAIR COMPARISON: RF vs ConvNeXt (same spatial split)")
    print("=" * 70)

    # Load data
    df_raw = pd.read_csv(DATA_CSV, low_memory=False)

    # Replicate exact same split as ConvNeXt
    df_valid = df_raw.dropna(subset=["ph", "k", "p"]).reset_index(drop=True)
    unique_locs = df_valid[["grid_id", "centroid_lon", "centroid_lat", "sampling_date"]].drop_duplicates()
    base_df = df_valid.loc[unique_locs.index].copy().reset_index(drop=True)

    unique_fields = np.array(base_df[FARM_COL].unique().tolist())
    np.random.seed(SEED)
    np.random.shuffle(unique_fields)

    train_fields = unique_fields[:-(VAL_COUNT + TEST_COUNT)]
    val_fields = unique_fields[-(VAL_COUNT + TEST_COUNT):-TEST_COUNT]
    test_fields = unique_fields[-TEST_COUNT:]

    print(f"  Split: {len(train_fields)} train / {len(val_fields)} val / {len(test_fields)} test fields")
    print(f"  Base dataset: {len(base_df)} samples, {len(unique_fields)} fields")

    # Load ConvNeXt baseline results
    convnext_results = load_convnext_results()

    all_records = []

    for target in TARGETS:
        print(f"\n{'─' * 60}")
        print(f"  TARGET: {TARGET_LABELS[target]}")
        print(f"{'─' * 60}")

        # Prepare data for current target
        target_df = base_df.dropna(subset=[target]).reset_index(drop=True)

        # Load 15 MDI-selected features
        features = load_features(target)
        missing = [f for f in features if f not in target_df.columns]
        if missing:
            print(f"  WARNING: missing features: {missing}")
            features = [f for f in features if f in target_df.columns]

        # Compute split indices for this target
        train_mask = target_df[FARM_COL].isin(train_fields)
        val_mask = target_df[FARM_COL].isin(val_fields)
        test_mask = target_df[FARM_COL].isin(test_fields)

        X_train = target_df.loc[train_mask, features].values.copy()
        X_val = target_df.loc[val_mask, features].values.copy()
        X_test = target_df.loc[test_mask, features].values.copy()
        y_train = target_df.loc[train_mask, target].values
        y_val = target_df.loc[val_mask, target].values
        y_test = target_df.loc[test_mask, target].values

        print(f"  Samples: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

        # Median imputation (train-only statistics)
        train_median = np.nanmedian(X_train, axis=0)
        train_median[np.isnan(train_median)] = 0.0
        for c in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, c]), c] = train_median[c]
            X_val[np.isnan(X_val[:, c]), c] = train_median[c]
            X_test[np.isnan(X_test[:, c]), c] = train_median[c]

        # Merge train + val for RF (RF has no early stopping, so val is not needed)
        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])

        # Groups for GroupKFold within GridSearchCV
        # Use field_name as group to prevent spatial leakage in nested CV
        fields_trainval = pd.concat([
            target_df.loc[train_mask, FARM_COL],
            target_df.loc[val_mask, FARM_COL],
        ]).values

        # GridSearchCV for RF
        print(f"  Running GridSearchCV for RF...")
        gkf = GroupKFold(n_splits=5)
        rf_base = RandomForestRegressor(random_state=SEED, n_jobs=-1)
        search = GridSearchCV(
            estimator=rf_base,
            param_grid=RF_GRID,
            cv=gkf.split(X_trainval, y_trainval, fields_trainval),
            scoring="r2",
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        search.fit(X_trainval, y_trainval)
        best_rf = search.best_estimator_

        print(f"  Best RF params: {search.best_params_}")
        print(f"  Best CV R²: {search.best_score_:.4f}")

        # Predict on test
        y_pred = best_rf.predict(X_test)

        # Compute all metrics
        rho, p_val = spearmanr(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        rpd = compute_rpd(y_test, rmse)
        ccc = compute_ccc(y_test, y_pred)

        record = {
            "target": TARGET_LABELS[target],
            "model": "RF (GridSearchCV)",
            "rho": round(rho, 4),
            "R2": round(r2, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "RPD": round(rpd, 3),
            "CCC": round(ccc, 4),
            "best_params": str(search.best_params_),
            "cv_best_R2": round(search.best_score_, 4),
        }
        all_records.append(record)

        print(f"  RF: ρ={rho:.4f}  R²={r2:.4f}  RMSE={rmse:.4f}  RPD={rpd:.3f}  CCC={ccc:.4f}")

        # ConvNeXt results
        for variant in ["single", "multi"]:
            key = f"{target}_{variant}"
            if key in convnext_results:
                cr = convnext_results[key]
                print(f"  ConvNeXt ({variant}): R²={cr['R2']:.4f}  RMSE={cr['RMSE']:.4f}  config={cr['config']}")

                # Compute ρ, RPD, CCC for ConvNeXt requires raw predictions which we don't have
                # We can only report R2 and RMSE
                all_records.append({
                    "target": TARGET_LABELS[target],
                    "model": f"ConvNeXt ({'multi-season' if variant == 'multi' else 'single-season'})",
                    "rho": None,  # Not available from existing results
                    "R2": cr["R2"],
                    "RMSE": cr["RMSE"],
                    "MAE": None,
                    "RPD": round(np.std(y_test, ddof=1) / cr["RMSE"], 3) if cr["RMSE"] > 0 else None,
                    "CCC": None,
                    "best_params": cr["config"],
                    "cv_best_R2": None,
                })

    # Save results
    out_df = pd.DataFrame(all_records)
    out_csv = OUT_DIR / "rf_vs_cnn_comparison.csv"
    out_df.to_csv(out_csv, index=False)

    out_json = OUT_DIR / "rf_vs_cnn_comparison.json"
    with open(out_json, "w") as f:
        json.dump(all_records, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  Results saved to {out_csv}")
    print(f"{'=' * 70}")

    # Print comparison table
    print("\n  === COMPARISON TABLE (same spatial split) ===")
    for target in TARGETS:
        label = TARGET_LABELS[target]
        rows = [r for r in all_records if r["target"] == label]
        print(f"\n  {label}:")
        for r in rows:
            rho_str = f"ρ={r['rho']:.4f}" if r['rho'] is not None else "ρ=N/A"
            rpd_str = f"RPD={r['RPD']:.3f}" if r['RPD'] is not None else "RPD=N/A"
            print(f"    {r['model']:40s}  R²={r['R2']:.4f}  RMSE={r['RMSE']:.4f}  {rho_str}  {rpd_str}")


if __name__ == "__main__":
    main()
