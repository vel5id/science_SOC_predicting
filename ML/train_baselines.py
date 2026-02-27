import os
import sys

# Ensure project root is in path regardless of CWD
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from ML.data_loader import SpatialDataLoader

OUT_DIR = os.path.join(_PROJECT_ROOT, "ML/results/baselines")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = ["ph", "soc", "no3", "p", "k", "s"]

# Define the models to train
MODELS = {
    "ET": ExtraTreesRegressor(n_estimators=300, max_features="sqrt", min_samples_leaf=2, random_state=42, n_jobs=-1),
    "KNN": KNeighborsRegressor(n_neighbors=7, weights='distance', n_jobs=-1),
    "LR": LinearRegression(n_jobs=-1),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "SGD": SGDRegressor(loss='squared_error', penalty='l2', alpha=0.01, random_state=42, max_iter=2000),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "GBDT": GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, subsample=0.8)
}

def train_and_evaluate_model(target: str, model_name: str, model_instance) -> dict:
    # Most baseline models (KNN, LR, Ridge, SGD, SVR) require features to be scaled!
    # ExtraTrees and GBDT theoretically don't, but scaling doesn't hurt them and provides a uniform pipeline.
    loader = SpatialDataLoader(target=target, scale_features=True)
    X, y, fields = loader.get_data()

    oof_preds = np.zeros_like(y, dtype=float)

    for train_idx, test_idx, test_field in loader.iter_lofo_cv():
        X_train, X_test, y_train, y_test = loader.get_fold_data(train_idx, test_idx)

        # Clone the model to ensure a clean state
        from sklearn.base import clone
        model = clone(model_instance)
        
        model.fit(X_train, y_train)
        oof_preds[test_idx] = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    mae  = mean_absolute_error(y, oof_preds)
    r2   = r2_score(y, oof_preds)
    rho, _ = spearmanr(y, oof_preds)

    # Save OOF
    model_dir = os.path.join(OUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    oof_df = loader.df.copy()
    oof_df['oof_pred'] = oof_preds
    oof_df.to_csv(os.path.join(model_dir, f"{target}_oof_predictions.csv"), index=False)

    return {"rho": float(rho), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


if __name__ == "__main__":
    results_all = {}
    
    for model_name, model_instance in MODELS.items():
        print(f"\n{'='*50}")
        print(f" Training Baseline Model: {model_name}")
        print(f"{'='*50}")
        
        model_results = {}
        for t in TARGETS:
            try:
                metrics = train_and_evaluate_model(t, model_name, model_instance)
                model_results[t] = metrics
                print(f"  [{t.upper()}] ρ = {metrics['rho']:.3f}, RMSE = {metrics['rmse']:.3f}, R² = {metrics['r2']:.3f}")
            except Exception as e:
                print(f"  Error training {t}: {e}")
                
        results_all[model_name] = model_results

    # Save summary containing all baseline models
    with open(os.path.join(OUT_DIR, "baselines_metrics_summary.json"), "w") as f:
        json.dump(results_all, f, indent=4)

    print(f"\n[DONE] Baseline results saved in {OUT_DIR}/")
