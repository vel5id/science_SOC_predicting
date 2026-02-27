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
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from ML.data_loader import SpatialDataLoader

OUT_DIR = os.path.join(_PROJECT_ROOT, "ML/results/cart")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = ["ph", "soc", "no3", "p", "k", "s"]

TARGET_LABELS = {
    "ph": "pH (KCl)", "soc": "SOC, %", "no3": "NO3, mg/kg",
    "p": "P2O5, mg/kg", "k": "K2O, mg/kg", "s": "S, mg/kg"
}

# CART hyperparameters
# Notes:
#  - max_depth limits the tree to prevent pure overfitting
#  - min_samples_leaf ensures each leaf has at least N samples (regularization)
#  - ccp_alpha: cost-complexity pruning (alpha > 0 prunes more aggressively)
CART_PARAMS = {
    "max_depth":        6,       # Shallow enough to generalize, deep enough to catch non-linearity
    "min_samples_leaf": 10,      # Each leaf must cover at least 10 samples (~1% of data)
    "min_samples_split": 20,     # Min samples needed to split an internal node
    "ccp_alpha":        0.01,    # Pruning: removes subtrees that don't improve performance enough
    "random_state":    42,
}

def train_and_evaluate(target: str):
    print(f"\n{'='*50}")
    print(f" Training CART for: {target.upper()}")
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

        model = DecisionTreeRegressor(**CART_PARAMS)
        model.fit(X_train, y_train)

        oof_preds[test_idx] = model.predict(X_test)
        fold_importances.append(model.feature_importances_)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    mae  = mean_absolute_error(y, oof_preds)
    r2   = r2_score(y, oof_preds)
    rho, p_val = spearmanr(y, oof_preds)

    print(f"\n[Results {target.upper()}]")
    print(f"  Spearman rho: {rho:.3f} (p={p_val:.2e})")
    print(f"  RMSE:         {rmse:.3f}")
    print(f"  MAE:          {mae:.3f}")
    print(f"  R2:           {r2:.3f}")

    # Save OOF
    oof_df = loader.df.copy()
    oof_df['oof_pred'] = oof_preds
    oof_df.to_csv(os.path.join(OUT_DIR, f"{target}_oof_predictions.csv"), index=False)

    # Feature importance (averaged over folds)
    mean_imp = np.mean(fold_importances, axis=0)
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': mean_imp})
    imp_df = imp_df.sort_values('importance', ascending=False)
    imp_df.to_csv(os.path.join(OUT_DIR, f"{target}_feature_importance.csv"), index=False)

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y, oof_preds, alpha=0.5, edgecolor='k', s=20)
    mn, mx = min(y.min(), oof_preds.min()), max(y.max(), oof_preds.max())
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2)
    plt.title(f"CART LOFO-CV — {TARGET_LABELS.get(target, target)}\nSpearman ρ = {rho:.3f}")
    plt.xlabel(f"True {TARGET_LABELS.get(target, target)}")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{target}_scatter.png"), dpi=300)
    plt.close()

    # Feature Importance bar chart
    plt.figure(figsize=(10, 5))
    top_n = min(15, len(imp_df))
    plt.barh(imp_df['feature'].head(top_n)[::-1], imp_df['importance'].head(top_n)[::-1], color='#d62728', alpha=0.80)
    plt.title(f"CART Feature Importance — {TARGET_LABELS.get(target, target)}")
    plt.xlabel("Mean Decrease Impurity (avg over LOFO folds)")
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

    with open(os.path.join(OUT_DIR, "cart_metrics_summary.json"), "w") as f:
        json.dump(results_all, f, indent=4)

    print(f"\n{'='*50}")
    print("FINAL CART SUMMARY (Spearman ρ, LOFO-CV)")
    print(f"{'='*50}")
    for t, m in results_all.items():
        print(f"  {TARGET_LABELS.get(t, t):<15}: ρ = {m['rho']:.3f}  RMSE = {m['rmse']:.3f}  R² = {m['r2']:.3f}")

    print(f"\n[DONE] Results saved in {OUT_DIR}/")
