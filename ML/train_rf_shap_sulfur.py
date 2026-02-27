"""
XGBoost + SHAP Analysis for Sulfur (S) — TEMPORALLY CLEAN version.

TEMPORAL LEAKAGE FIX
====================
75.3% of soil samples were collected in Spring (April-May).
Previous version used summer/autumn satellite features and cross-season
delta/range/ts statistics — data that did NOT EXIST at sampling time.

This version uses ONLY temporally safe features:
  - Topographic features (topo_*): DEM, slope, aspect, curvature, TWI, TPI
  - Climate features (climate_*): MAT, MAP, GS_temp, GS_precip
  - Spring-only satellite features (*_spring): S2, L8, spectral, GLCM, PCA

EXCLUDED as temporally leaky:
  - Any feature with _summer, _late_summer, _autumn suffix
  - Delta features (delta_*): require 2+ seasons
  - Range features (range_*): require all 4 seasons
  - Temporal statistics (ts_*): mean/std/slope over all seasons
  - Cross-sensor ratios for summer/autumn (cs_*_summer etc.)

ALSO REMOVED: SOC×NO3 interaction term
  - SOC and NO3 are lab measurements from the SAME soil analysis as S
  - Using them as predictors constitutes measurement leakage

Outputs (in ML/results/shap_sulfur_clean/):
  - shap_beeswarm_sulfur.png    — SHAP beeswarm (top-20)
  - feature_importance_bar.png  — XGBoost MDI importance
  - feature_importance_all.csv  — all features ranked by MDI
  - shap_mean_abs_values.csv    — all features ranked by |SHAP|
  - scatter_sulfur.png          — OOF scatter plot
  - metrics.json                — R², rho, RMSE, MAE
  - temporal_audit.txt          — sampling date distribution report
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import shap

OUT_DIR = os.path.join(_PROJECT_ROOT, "ML/results/shap_sulfur_clean")
os.makedirs(OUT_DIR, exist_ok=True)

# Columns that are NOT features (metadata / targets / geo)
META_COLS = {
    "id", "year", "farm", "field_name", "grid_id",
    "centroid_lon", "centroid_lat", "geometry_wkt",
    "protocol_number", "analysis_date", "sampling_date",
    "sample_id",
}

# Primary targets + all lab chemistry = never use as predictors
# (measurement leakage: they come from the same lab analysis as S)
TARGET_COLS = {
    "ph", "k", "p", "hu", "s", "no3",
    "soc", "b", "ca", "na",
    "zn", "mo", "fe", "mg", "mn", "cu",
}

# Suffixes that indicate "future" satellite data for spring-sampled probes
FUTURE_SUFFIXES = ("_summer", "_late_summer", "_autumn")

# Prefixes of cross-season engineered features (always leaky for spring probes)
CROSS_SEASON_PREFIXES = ("delta_", "range_", "ts_")

XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
    "tree_method":      "hist",
}


def is_temporally_clean(col: str) -> bool:
    """Return True if the feature is safe to use for spring-sampled probes.

    Safe features:
      - topo_*    : topographic indices (DEM, slope, aspect ...)
      - climate_* : climate normals (MAT, MAP, GS_temp, GS_precip)
      - *_spring  : satellite data collected in spring (Mar-May)
        This includes: s2_*_spring, l8_*_spring, spectral_*_spring,
                       glcm_*_spring, cs_*_spring, spectral_PCA_*_spring

    Unsafe (leaky) for spring probes:
      - *_summer, *_late_summer, *_autumn : collected AFTER spring sampling
      - delta_*, range_*, ts_*            : derived from 2+ seasons
      - cs_*_summer / cs_*_autumn         : cross-sensor ratio using future data
      - glcm_delta_*_spr2sum / spr2aut    : GLCM deltas spanning future seasons
    """
    # Always safe
    if col.startswith("topo_") or col.startswith("climate_"):
        return True

    # Cross-season derived: always leaky
    if any(col.startswith(pfx) for pfx in CROSS_SEASON_PREFIXES):
        return False

    # GLCM temporal deltas (contain spr2sum, spr2aut patterns)
    if col.startswith("glcm_delta_"):
        return False

    # Feature uses future-season data
    if any(col.endswith(sfx) or (sfx[1:] + "_") in col for sfx in FUTURE_SUFFIXES):
        # Catch e.g. glcm_glcm_nir_asm_summer, cs_NDVI_ratio_late_summer
        return False

    # Spring-only satellite features — safe
    if col.endswith("_spring"):
        return True

    # glcm_ratio_*_spring pattern
    if "_spring" in col:
        return True

    # spectral_PCA_*_spring
    if "PCA" in col and "_spring" in col:
        return True

    # Anything else (unknown prefix/suffix) — exclude conservatively
    return False


# =================================================================
# Data loading — TEMPORALLY CLEAN features for S
# =================================================================

def load_clean_features_for_sulfur():
    """Load features with NO temporal leakage for sulfur prediction.

    Uses only spring satellite data + topography + climate.
    Parsing sampling_date to audit temporal composition of the dataset.
    """
    data_path = os.path.join(_PROJECT_ROOT, "data/features/master_dataset.csv")
    raw_df = pd.read_csv(data_path, low_memory=False)

    # 1. Drop rows with missing S target
    df = raw_df.dropna(subset=["s"]).copy().reset_index(drop=True)

    # 2. Temporal audit — verify sampling date distribution
    audit_lines = ["TEMPORAL AUDIT — sampling_date distribution", "=" * 55]
    if "sampling_date" in df.columns:
        dates = pd.to_datetime(df["sampling_date"], format="%d.%m.%Y", errors="coerce")
        month_counts = dates.dt.month.value_counts().sort_index()
        month_names = {
            3: "March",   4: "April",    5: "May",
            6: "June",    7: "July",     8: "August",
            9: "September", 10: "October", 11: "November",
        }
        spring_n  = int(dates.dt.month.isin([3, 4, 5]).sum())
        summer_n  = int(dates.dt.month.isin([6, 7, 8]).sum())
        autumn_n  = int(dates.dt.month.isin([9, 10, 11]).sum())
        total     = len(df)
        audit_lines += [
            f"  Total S samples: {total}",
            f"  Spring (Mar-May):  {spring_n:4d}  ({100*spring_n/total:.1f}%)",
            f"  Summer (Jun-Aug):  {summer_n:4d}  ({100*summer_n/total:.1f}%)",
            f"  Autumn (Sep-Nov):  {autumn_n:4d}  ({100*autumn_n/total:.1f}%)",
            "",
            f"  → For {spring_n}/{total} ({100*spring_n/total:.1f}%) of samples,",
            f"    summer and autumn satellite features are FUTURE data.",
            f"  → This version uses ONLY spring + topo + climate features.",
        ]
    else:
        audit_lines.append("  WARNING: sampling_date column not found in master_dataset.csv")

    audit_text = "\n".join(audit_lines)
    print(audit_text)
    with open(os.path.join(OUT_DIR, "temporal_audit.txt"), "w", encoding="utf-8") as f:
        f.write(audit_text + "\n")

    # 3. Select ONLY temporally clean numeric feature columns
    exclude = META_COLS | TARGET_COLS
    all_numeric = [c for c in df.columns
                   if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]

    feature_cols = [c for c in all_numeric if is_temporally_clean(c)]

    # Audit: show what was removed and kept
    removed = [c for c in all_numeric if not is_temporally_clean(c)]
    print(f"\nFeature filtering:")
    print(f"  Numeric features in dataset : {len(all_numeric)}")
    print(f"  Excluded (temporally leaky) : {len(removed)}")
    print(f"  Kept (temporally clean)     : {len(feature_cols)}")

    # Breakdown by category
    cats = {
        "topo":    [c for c in feature_cols if c.startswith("topo_")],
        "climate": [c for c in feature_cols if c.startswith("climate_")],
        "spring satellite": [c for c in feature_cols
                             if not c.startswith("topo_") and not c.startswith("climate_")],
    }
    for cat, cols in cats.items():
        print(f"    {cat:<22}: {len(cols)}")

    # Save audit of removed vs kept features
    audit_feats = pd.DataFrame({
        "feature": all_numeric,
        "kept":    [is_temporally_clean(c) for c in all_numeric],
    })
    audit_feats.to_csv(
        os.path.join(OUT_DIR, "feature_temporal_audit.csv"), index=False
    )

    X_df = df[feature_cols].copy()
    y = df["s"].values
    fields = df["field_name"].values
    feature_names = list(X_df.columns)

    return X_df.values, y, fields, feature_names


# =================================================================
# LOFO CV iterator (spatial)
# =================================================================

def iter_lofo(fields):
    unique_fields = np.unique(fields)
    for uf in unique_fields:
        test_mask = fields == uf
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx, str(uf)


# =================================================================
# Train XGBoost + Compute SHAP
# =================================================================

def run_shap_analysis():
    print("=" * 60)
    print("  XGBoost + SHAP Analysis for S (Sulfur)")
    print("  [TEMPORALLY CLEAN — spring + topo + climate only]")
    print("=" * 60)

    X, y, fields, feature_names = load_clean_features_for_sulfur()
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Unique fields: {len(np.unique(fields))}")

    # S is right-skewed (skew ≈ 3.5): train on log1p, predict then expm1 back.
    # Identical to the RF pipeline convention (LOG_TARGETS = {"p", "s", "no3"}).
    y_log = np.log1p(y)
    print(f"  log1p(S): skew reduced, train in log-space, metrics in original scale")

    oof_log_preds = np.zeros_like(y, dtype=float)   # accumulate log-space preds
    fold_importances = []
    all_shap_values = []
    all_X_test = []

    fold = 0
    for train_idx, test_idx, test_field in iter_lofo(fields):
        X_train = X[train_idx]
        y_train_log = y_log[train_idx]          # log-transformed train targets
        X_test = X[test_idx].copy()

        # Internal validation from training fields (15%)
        unique_train_fields = np.unique(fields[train_idx])
        np.random.seed(42)
        n_val = max(1, int(len(unique_train_fields) * 0.15))
        val_fields = np.random.choice(unique_train_fields, size=n_val, replace=False)
        val_mask = np.isin(fields[train_idx], val_fields)
        tr_mask = ~val_mask

        X_tr  = X_train[tr_mask].copy()
        y_tr  = y_train_log[tr_mask]
        X_val = X_train[val_mask].copy()
        y_val = y_train_log[val_mask]

        # Per-fold Median Imputation (using tr_mask only)
        tr_median = np.nanmedian(X_tr, axis=0)
        tr_median[np.isnan(tr_median)] = 0.0  # fallback for all-NaN columns
        for col_idx in range(X_tr.shape[1]):
            med = tr_median[col_idx]
            X_tr[np.isnan(X_tr[:, col_idx]), col_idx]    = med
            X_val[np.isnan(X_val[:, col_idx]), col_idx]  = med
            X_test[np.isnan(X_test[:, col_idx]), col_idx] = med

        model = XGBRegressor(**XGB_PARAMS, early_stopping_rounds=30)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        oof_log_preds[test_idx] = model.predict(X_test)
        fold_importances.append(model.feature_importances_)

        # SHAP values for the test fold
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        all_shap_values.append(shap_vals)
        all_X_test.append(X_test)

        fold += 1
        if fold % 20 == 0:
            print(f"  Completed {fold} LOFO folds...")

    print(f"  Total folds completed: {fold}")

    # --- Back-transform predictions to original scale ---
    oof_preds = np.expm1(oof_log_preds)   # original mg/kg scale

    # --- Aggregate SHAP values across all folds ---
    shap_values_full = np.vstack(all_shap_values)  # [N, F] — in log-space
    X_test_full = np.vstack(all_X_test)             # [N, F]

    # --- Metrics in original scale ---
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    mae = mean_absolute_error(y, oof_preds)
    r2 = r2_score(y, oof_preds)
    rho, p_val = spearmanr(y, oof_preds)  # same as on log scale (monotonic)

    print(f"\n[Results S (Sulfur)] — TEMPORALLY CLEAN (spring + topo + climate)")
    print(f"  Spearman rho: {rho:.3f} (p={p_val:.2e})")
    print(f"  RMSE:         {rmse:.3f}")
    print(f"  MAE:          {mae:.3f}")
    print(f"  R2:           {r2:.3f}")
    if r2 >= 0.5:
        print(f"  ✅ R² ≥ 0.5  — valid claim WITHOUT temporal leakage")
    else:
        print(f"  ⚠️  R² < 0.5  — claim of R² ≥ 0.5 does NOT hold without leakage")

    metrics = {"rho": float(rho), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # --- Feature Importance (MDI) ---
    mean_imp = np.mean(fold_importances, axis=0)
    imp_df = pd.DataFrame({"feature": feature_names, "importance": mean_imp})
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(OUT_DIR, "feature_importance_all.csv"), index=False)

    # --- SHAP CSV (mean |SHAP|) ---
    mean_abs_shap = np.mean(np.abs(shap_values_full), axis=0)
    shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(os.path.join(OUT_DIR, "shap_mean_abs_values.csv"), index=False)

    print(f"\nTop-10 SHAP features for S:")
    for _, row in shap_df.head(10).iterrows():
        print(f"  {row['feature']:40s}  |SHAP| = {row['mean_abs_shap']:.4f}")

    # --- SHAP Beeswarm Plot ---
    print("\nGenerating SHAP beeswarm plot...")
    plt.figure(figsize=(12, 10))
    shap_explanation = shap.Explanation(
        values=shap_values_full,
        data=X_test_full,
        feature_names=feature_names,
    )
    shap.plots.beeswarm(shap_explanation, max_display=20, show=False)
    plt.title(
        f"SHAP Feature Impact — Sulfur (S, mg/kg)\n"
        f"[Clean: spring+topo+climate | log1p target | ρ={rho:.3f}, R²={r2:.3f}]",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_beeswarm_sulfur.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/shap_beeswarm_sulfur.png")

    # --- Feature Importance Bar Plot ---
    plt.figure(figsize=(12, 7))
    top_n = 20
    top_imp = imp_df.head(top_n)
    plt.barh(
        top_imp["feature"][::-1],
        top_imp["importance"][::-1],
        color="#2196F3", alpha=0.85, edgecolor="white"
    )
    plt.title(
        "XGBoost Feature Importance — S (Sulfur)\n"
        "(Mean Gain, LOFO-CV | Clean: spring+topo+climate | log1p target)",
        fontsize=12,
    )
    plt.xlabel("Mean Gain")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/feature_importance_bar.png")

    # --- Scatter Plot ---
    plt.figure(figsize=(6, 6))
    plt.scatter(y, oof_preds, alpha=0.4, edgecolor="k", s=20, c="#FF5722")
    mn, mx = min(y.min(), oof_preds.min()), max(y.max(), oof_preds.max())
    plt.plot([mn, mx], [mn, mx], "k--", lw=1.5)
    plt.title(
        f"XGBoost LOFO-CV — S (mg/kg)\n"
        f"Spearman ρ = {rho:.3f}  |  R² = {r2:.3f}\n"
        f"[Clean: spring+topo+climate | log1p→expm1]",
        fontsize=11,
    )
    plt.xlabel("True S (mg/kg)")
    plt.ylabel("Predicted S (mg/kg)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scatter_sulfur.png"), dpi=300)
    plt.close()
    print(f"  Saved: {OUT_DIR}/scatter_sulfur.png")

    print(f"\n✅ SHAP analysis complete! All outputs in {OUT_DIR}/")
    print(f"\nNOTE: These results are temporally clean — no summer/autumn/delta features used.")
    print(f"  Compare with leaky baseline to quantify leakage contribution.")
    return metrics


if __name__ == "__main__":
    run_shap_analysis()
