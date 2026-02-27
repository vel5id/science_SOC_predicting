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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from ML.data_loader import SpatialDataLoader

OUT_DIR = os.path.join(_PROJECT_ROOT, "ML/results/rf")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = ["ph", "soc", "no3", "p", "k", "s"]

# Model parameters
RF_PARAMS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": -1
}

def train_and_evaluate(target: str):
    print(f"\n{'='*50}")
    print(f" Training Random Forest for: {target.upper()}")
    print(f"{'='*50}")
    
    loader = SpatialDataLoader(target=target, scale_features=False) # RF does not strictly need scaling, but loader supports it. We use unscaled for transparency.
    X, y, fields = loader.get_data()
    feature_names = loader.get_feature_names()
    
    print(f"Data shape: {X.shape}, Unique fields: {len(np.unique(fields))}")
    print(f"Features used: {len(feature_names)}")
    
    # We will collect Out-Of-Fold (OOF) predictions
    oof_preds = np.zeros_like(y)
    
    # Also collect feature importances to average them later
    fold_importances = []
    
    cv_gen = loader.iter_lofo_cv()
    
    fold = 0
    for train_idx, test_idx, test_field in cv_gen:
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Initialize and Train
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train, y_train)
        
        # Predict on hold-out field
        preds = model.predict(X_test)
        oof_preds[test_idx] = preds
        
        fold_importances.append(model.feature_importances_)
        fold += 1
        
    # Calculate overall OOF metrics
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    mae = mean_absolute_error(y, oof_preds)
    r2 = r2_score(y, oof_preds)
    
    # Spearman rho (which is our main metric from the article)
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
    
    # Feature Importance average
    mean_importances = np.mean(fold_importances, axis=0)
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_importances
    }).sort_values(by='importance', ascending=False)
    
    imp_df.to_csv(os.path.join(OUT_DIR, f"{target}_feature_importance.csv"), index=False)
    
    # Plot Scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(y, oof_preds, alpha=0.5, edgecolor='k')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.title(f"Random Forest LOFO CV - {target.upper()}\nSpearman $\\rho$ = {rho:.3f}")
    plt.xlabel(f"True {target.upper()}")
    plt.ylabel(f"Predicted {target.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{target}_scatter.png"), dpi=300)
    plt.close()
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    top_n = min(15, len(imp_df))
    plt.barh(imp_df['feature'].head(top_n)[::-1], imp_df['importance'].head(top_n)[::-1], color='skyblue')
    plt.title(f"Top Features Importance for {target.upper()} (MDI averaged over folds)")
    plt.xlabel("Mean Decrease Impurity")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{target}_importance.png"), dpi=300)
    plt.close()
    
    return {
        "rho": float(rho),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }

if __name__ == "__main__":
    results_all = {}
    for t in TARGETS:
        try:
            metrics = train_and_evaluate(t)
            results_all[t] = metrics
        except Exception as e:
            print(f"Error training {t}: {e}")
            
    # Save overall summary
    with open(os.path.join(OUT_DIR, "rf_metrics_summary.json"), "w") as f:
        json.dump(results_all, f, indent=4)
        
    print(f"\n[DONE] All targets processed. Results saved in {OUT_DIR}/")
