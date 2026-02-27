import os
import sys
# Ensure project root is in path regardless of CWD
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from ML.data_loader import SpatialDataLoader

OUT_DIR = os.path.join(_PROJECT_ROOT, "ML/results/xgb")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = ["ph", "soc", "no3", "p", "k", "s"]

# XGBoost hyperparameters (tuned for tabular regression, small dataset)
XGB_PARAMS = {
    "n_estimators":     400,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha":        0.1,     # L1 regularization
    "reg_lambda":       1.0,     # L2 regularization
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
    "tree_method":      "hist",  # Fast histogram-based method
}

TARGET_LABELS = {
    "ph": "pH (KCl)", "soc": "SOC, %", "no3": "NO3, mg/kg",
    "p": "P2O5, mg/kg", "k": "K2O, mg/kg", "s": "S, mg/kg"
}

def train_and_evaluate(target: str):
    print(f"\n{'='*50}")
    print(f" Training XGBoost for: {target.upper()}")
    print(f"{'='*50}")

    loader = SpatialDataLoader(target=target, scale_features=False)
    X, y, fields = loader.get_data()
    feature_names = loader.get_feature_names()

    print(f"Data shape: {X.shape}, Unique fields: {len(np.unique(fields))}")

    oof_preds = np.zeros_like(y, dtype=float)
    fold_importances = []

    for train_idx, test_idx, test_field in loader.iter_lofo_cv():
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        # Use 15% of train fields as early-stopping validation set
        unique_train_fields = np.unique(fields[train_idx])
        np.random.seed(42)
        n_val = max(1, int(len(unique_train_fields) * 0.15))
        val_fields = np.random.choice(unique_train_fields, size=n_val, replace=False)
        val_mask = np.isin(fields[train_idx], val_fields)
        tr_mask = ~val_mask

        X_tr, y_tr = X_train[tr_mask], y_train[tr_mask]
        X_val, y_val = X_train[val_mask], y_train[val_mask]

        model = XGBRegressor(**XGB_PARAMS, early_stopping_rounds=30)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        oof_preds[test_idx] = model.predict(X_test)
        fold_importances.append(model.feature_importances_)

    # Overall OOF metrics
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    mae  = mean_absolute_error(y, oof_preds)
    r2   = r2_score(y, oof_preds)
    rho, p_val = spearmanr(y, oof_preds)

    print(f"\n[Results {target.upper()}]")
    print(f"  Spearman rho: {rho:.3f} (p={p_val:.2e})")
    print(f"  RMSE:         {rmse:.3f}")
    print(f"  MAE:          {mae:.3f}")
    print(f"  R2:           {r2:.3f}")

    # Save OOF predictions
    oof_df = loader.df.copy()
    oof_df['oof_pred'] = oof_preds
    oof_df.to_csv(os.path.join(OUT_DIR, f"{target}_oof_predictions.csv"), index=False)

    # Feature importance
    mean_imp = np.mean(fold_importances, axis=0)
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': mean_imp})
    imp_df = imp_df.sort_values('importance', ascending=False)
    imp_df.to_csv(os.path.join(OUT_DIR, f"{target}_feature_importance.csv"), index=False)

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y, oof_preds, alpha=0.5, edgecolor='k', s=20)
    mn, mx = min(y.min(), oof_preds.min()), max(y.max(), oof_preds.max())
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2)
    plt.title(f"XGBoost LOFO-CV — {TARGET_LABELS.get(target, target)}\nSpearman ρ = {rho:.3f}")
    plt.xlabel(f"True {TARGET_LABELS.get(target, target)}")
    plt.ylabel(f"Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{target}_scatter.png"), dpi=300)
    plt.close()

    # Feature importance bar
    plt.figure(figsize=(10, 5))
    top_n = min(15, len(imp_df))
    plt.barh(imp_df['feature'].head(top_n)[::-1], imp_df['importance'].head(top_n)[::-1], color='#2ca02c', alpha=0.85)
    plt.title(f"XGBoost Feature Importance — {TARGET_LABELS.get(target, target)}")
    plt.xlabel("Mean Gain (avg over LOFO folds)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{target}_importance.png"), dpi=300)
    plt.close()

    return {"rho": float(rho), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


if __name__ == "__main__":
    results_all = {}
    for t in TARGETS:
        try:
            results_all[t] = train_and_evaluate(t)
        except Exception as e:
            print(f"Error training {t}: {e}")

    # Save summary
    with open(os.path.join(OUT_DIR, "xgb_metrics_summary.json"), "w") as f:
        json.dump(results_all, f, indent=4)

    print(f"\n{'='*50}")
    print("FINAL XGBoost SUMMARY (Spearman ρ, LOFO-CV)")
    print(f"{'='*50}")
    for t, m in results_all.items():
        print(f"  {TARGET_LABELS.get(t, t):<15}: ρ = {m['rho']:.3f}  RMSE = {m['rmse']:.3f}  R² = {m['r2']:.3f}")

    print(f"\n[DONE] Results saved in {OUT_DIR}/")
