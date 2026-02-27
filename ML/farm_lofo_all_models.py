"""
farm_lofo_all_models.py
=======================
Farm-LOFO-CV (Leave-One-Farm-Out Cross-Validation) for ALL 11 ML models
across all 6 target agrochemical properties.

Uses the SAME 15 MDI-selected features per target as Field-LOFO-CV
to ensure methodological comparability.

Addresses reviewer comment M3: "Farm-LOFO only for RF".

Output: ML/results/farm_lofo_all_models.csv
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "features" / "master_dataset.csv"
SELECTED_DIR = ROOT / "data" / "features" / "selected"
OUT_CSV = ROOT / "ML" / "results" / "farm_lofo_all_models.csv"

TARGETS = ["ph", "soc", "no3", "p", "k", "s"]
TARGET_FEATURE_MAP = {
    "ph": "ph", "soc": "soc", "hu": "soc",
    "no3": "no3", "p": "p", "k": "k", "s": "s",
}
FARM_COL = "farm"
SEED = 42


def load_selected_features(target: str) -> list[str]:
    """Load the 15 MDI-selected features for a target."""
    key = TARGET_FEATURE_MAP.get(target, target)
    path = SELECTED_DIR / f"{key}_best_features.txt"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    with open(path) as f:
        feats = [line.strip() for line in f if line.strip()]
    return feats


def build_models() -> dict:
    """Return dict of model_name → unfitted estimator."""
    return {
        "RF": RandomForestRegressor(
            n_estimators=500, max_features="sqrt",
            min_samples_leaf=3, random_state=SEED, n_jobs=-1,
        ),
        "ET": ExtraTreesRegressor(
            n_estimators=500, max_features="sqrt",
            min_samples_leaf=3, random_state=SEED, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbosity=0,
        ),
        "CatBoost": cb.CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.05,
            random_seed=SEED, verbose=0,
        ),
        "GBDT": GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=SEED,
        ),
        "CART": DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=5, random_state=SEED,
        ),
        "KNN": KNeighborsRegressor(
            n_neighbors=7, weights="distance", n_jobs=-1,
        ),
        "LR": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=SEED),
        "SGD": SGDRegressor(
            alpha=1e-4, penalty="l2", max_iter=1000, random_state=SEED,
        ),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    }


# Models that need feature scaling
NEEDS_SCALING = {"KNN", "LR", "Ridge", "SGD", "SVR"}


def farm_lofo_cv(df: pd.DataFrame, target: str, features: list[str],
                 model_name: str, model_factory) -> dict:
    """
    Run Farm-LOFO-CV for a single (target, model) pair.
    Returns dict with aggregated metrics.
    """
    sub = df.dropna(subset=[target]).reset_index(drop=True)
    farms = sub[FARM_COL].unique()
    n_farms = len(farms)

    all_preds = np.full(len(sub), np.nan)
    all_actual = sub[target].values

    for farm in farms:
        train_mask = (sub[FARM_COL] != farm).values
        test_mask = (sub[FARM_COL] == farm).values

        X_train_raw = sub.loc[train_mask, features].values.copy()
        X_test_raw = sub.loc[test_mask, features].values.copy()
        y_train = sub.loc[train_mask, target].values
        # y_test = sub.loc[test_mask, target].values  # not needed, we use all_actual

        # Median imputation based on train only
        train_median = np.nanmedian(X_train_raw, axis=0)
        train_median[np.isnan(train_median)] = 0.0
        for c in range(X_train_raw.shape[1]):
            X_train_raw[np.isnan(X_train_raw[:, c]), c] = train_median[c]
            X_test_raw[np.isnan(X_test_raw[:, c]), c] = train_median[c]

        # Scaling for models that need it
        if model_name in NEEDS_SCALING:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)
        else:
            X_train = X_train_raw
            X_test = X_test_raw

        # Build fresh model
        model = model_factory()

        # Fit
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)
        all_preds[test_mask] = preds

    # Aggregate metrics over all folds
    valid = np.isfinite(all_preds) & np.isfinite(all_actual)
    if valid.sum() < 10:
        return dict(
            target=target, model=model_name,
            rho=np.nan, r2=np.nan, rmse=np.nan, mae=np.nan,
            n=int(valid.sum()), n_farms=n_farms,
        )

    rho, _ = spearmanr(all_actual[valid], all_preds[valid])
    r2 = r2_score(all_actual[valid], all_preds[valid])
    rmse = np.sqrt(mean_squared_error(all_actual[valid], all_preds[valid]))
    mae = mean_absolute_error(all_actual[valid], all_preds[valid])

    return dict(
        target=target, model=model_name,
        rho=round(rho, 4), r2=round(r2, 4),
        rmse=round(rmse, 4), mae=round(mae, 4),
        n=int(valid.sum()), n_farms=n_farms,
    )


def main():
    print("=" * 70)
    print("  Farm-LOFO-CV: ALL 11 ML MODELS × 6 TARGETS")
    print("=" * 70)

    print(f"\nLoading {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"  {len(df)} rows, {df[FARM_COL].nunique()} farms, "
          f"{df['field_name'].nunique()} fields")

    all_records = []
    total_combos = len(TARGETS) * 11
    done = 0

    for target in TARGETS:
        features = load_selected_features(target)
        # Verify all features exist
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"  WARNING: {target} missing features: {missing}")
            features = [f for f in features if f in df.columns]

        print(f"\n{'─' * 60}")
        print(f"  TARGET: {target.upper()}  ({len(features)} features)")
        print(f"{'─' * 60}")

        model_specs = build_models()

        for model_name, model_template in model_specs.items():
            done += 1
            t0 = time.time()

            # Factory function to create fresh model each fold
            def make_model(_m=model_template):
                import copy
                return copy.deepcopy(_m)

            rec = farm_lofo_cv(df, target, features, model_name, make_model)
            elapsed = time.time() - t0

            rec["time_s"] = round(elapsed, 1)
            all_records.append(rec)

            print(f"  [{done:3d}/{total_combos}] {model_name:10s}  "
                  f"ρ={rec['rho']:+.4f}  R²={rec['r2']:+.4f}  "
                  f"RMSE={rec['rmse']:.4f}  ({elapsed:.1f}s)")

    # Save results
    out_df = pd.DataFrame(all_records)
    os.makedirs(OUT_CSV.parent, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n{'=' * 70}")
    print(f"  DONE. Saved {len(out_df)} results → {OUT_CSV}")
    print(f"{'=' * 70}")

    # Print summary pivot table
    print("\n  Summary (Spearman ρ):")
    pivot = out_df.pivot(index="model", columns="target", values="rho")
    pivot = pivot[TARGETS]  # order columns
    print(pivot.to_string())


if __name__ == "__main__":
    main()
