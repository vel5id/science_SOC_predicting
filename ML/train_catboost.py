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
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from tqdm import tqdm
from ML.data_loader import SpatialDataLoader

OUT_DIR = os.path.join(_PROJECT_ROOT, "ML/results/catboost")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = ["ph", "soc", "no3", "p", "k", "s"]

TARGET_LABELS = {
    "ph": "pH (KCl)", "soc": "SOC, %", "no3": "NO3, mg/kg",
    "p": "P2O5, mg/kg", "k": "K2O, mg/kg", "s": "S, mg/kg"
}

# CatBoost hyperparameters
# task_type="GPU" will automatically use the available GPU.
# If no GPU is present, it gracefully falls back to CPU.
CATBOOST_PARAMS = {
    "iterations":        1000,
    "learning_rate":     0.05,
    "depth":             6,
    "l2_leaf_reg":       3.0,
    "min_data_in_leaf":  5,
    "random_seed":       42,
    "eval_metric":       "RMSE",
    "early_stopping_rounds": 50,
    "verbose":           False,
    "task_type":         "GPU",   # Will use GPU if available, else raises. Caught below.
}

# Detect GPU availability
try:
    _probe = CatBoostRegressor(iterations=1, task_type="GPU", verbose=False)
    _probe.fit([[1, 2], [3, 4]], [1, 2])
    USE_GPU = True
except Exception:
    USE_GPU = False

CATBOOST_PARAMS["task_type"] = "GPU" if USE_GPU else "CPU"

print("\n" + "="*54)
print(f"  CatBoost device : {'GPU  ✅' if USE_GPU else 'CPU  (no GPU available)'}")
print("="*54 + "\n")


def train_and_evaluate(target: str) -> dict:
    print(f"\n{'='*50}")
    print(f" Training CatBoost for: {target.upper()}")
    print(f"{'='*50}")

    loader = SpatialDataLoader(target=target, scale_features=False)  # CatBoost handles its own scaling
    X, y, fields = loader.get_data()
    feature_names = loader.get_feature_names()

    print(f"Data shape   : {X.shape}, Unique fields: {len(np.unique(fields))}")
    print(f"Features used: {len(feature_names)}")

    oof_preds = np.zeros_like(y, dtype=float)
    fold_importances = []

    lofo_iter = list(loader.iter_lofo_cv())
    fold_bar = tqdm(lofo_iter, desc=f"  Folds [{target.upper()}]",
                    unit="fold", ncols=90, colour="cyan")

    for fold, (train_idx, test_idx, test_field) in enumerate(fold_bar):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        # Inner validation split (15% of train fields) for early stopping
        unique_train_fields = np.unique(fields[train_idx])
        np.random.seed(42 + fold)
        n_val = max(1, int(len(unique_train_fields) * 0.15))
        val_fields = np.random.choice(unique_train_fields, size=n_val, replace=False)

        val_mask = np.isin(fields[train_idx], val_fields)
        tr_mask  = ~val_mask

        X_tr, y_tr   = X_train[tr_mask],  y_train[tr_mask]
        X_val, y_val = X_train[val_mask], y_train[val_mask]

        train_pool = Pool(X_tr,  y_tr,  feature_names=feature_names)
        eval_pool  = Pool(X_val, y_val, feature_names=feature_names)

        model = CatBoostRegressor(**CATBOOST_PARAMS)
        model.fit(train_pool, eval_set=eval_pool)

        oof_preds[test_idx] = model.predict(X_test)
        fold_importances.append(model.get_feature_importance())

        fold_bar.set_postfix(field=test_field,
                             best_iter=model.best_iteration_)

    # Overall OOF metrics
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    mae  = mean_absolute_error(y, oof_preds)
    r2   = r2_score(y, oof_preds)
    rho, p_val = spearmanr(y, oof_preds)

    print(f"\n[Results {target.upper()}]")
    print(f"  Spearman rho : {rho:.3f} (p={p_val:.2e})")
    print(f"  RMSE         : {rmse:.3f}")
    print(f"  MAE          : {mae:.3f}")
    print(f"  R²           : {r2:.3f}")

    # Save OOF predictions
    oof_df = loader.df.copy()
    oof_df["oof_pred"] = oof_preds
    oof_df.to_csv(os.path.join(OUT_DIR, f"{target}_oof_predictions.csv"), index=False)

    # Feature importance (mean over folds)
    mean_imp = np.mean(fold_importances, axis=0)
    imp_df = pd.DataFrame({"feature": feature_names, "importance": mean_imp})
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(OUT_DIR, f"{target}_feature_importance.csv"), index=False)

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y, oof_preds, alpha=0.5, edgecolor="k", s=20)
    mn, mx = min(y.min(), oof_preds.min()), max(y.max(), oof_preds.max())
    plt.plot([mn, mx], [mn, mx], "r--", lw=2)
    plt.title(f"CatBoost LOFO-CV — {TARGET_LABELS.get(target, target)}\nSpearman ρ = {rho:.3f}")
    plt.xlabel(f"True {TARGET_LABELS.get(target, target)}")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{target}_scatter.png"), dpi=300)
    plt.close()

    # Feature importance bar
    plt.figure(figsize=(10, 5))
    top_n = min(15, len(imp_df))
    plt.barh(imp_df["feature"].head(top_n)[::-1],
             imp_df["importance"].head(top_n)[::-1],
             color="#9467bd", alpha=0.85)
    plt.title(f"CatBoost Feature Importance — {TARGET_LABELS.get(target, target)}")
    plt.xlabel("Feature Importance (FeatureImportance avg over LOFO folds)")
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
            print(f"\n[ERROR] {t}: {e}")

    with open(os.path.join(OUT_DIR, "catboost_metrics_summary.json"), "w") as f:
        json.dump(results_all, f, indent=4)

    print(f"\n{'='*50}")
    print("FINAL CatBoost SUMMARY (Spearman ρ, LOFO-CV)")
    print(f"{'='*50}")
    for t, m in results_all.items():
        print(f"  {TARGET_LABELS.get(t, t):<15}: ρ = {m['rho']:.3f}  "
              f"RMSE = {m['rmse']:.3f}  R² = {m['r2']:.3f}")

    print(f"\n[DONE] Results saved in {OUT_DIR}/")
