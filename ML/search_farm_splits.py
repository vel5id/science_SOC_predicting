import io, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_CSV = ROOT / "data" / "features" / "master_dataset.csv"

def get_tabular_cols(df):
    num_cols = df.select_dtypes(include="number").columns
    spring  = sorted([c for c in num_cols if "_spring" in c and "_summer" not in c and "_autumn" not in c and "_late_summer" not in c])
    topo    = sorted([c for c in num_cols if c.startswith("topo_")])
    climate = sorted([c for c in num_cols if c.startswith("climate_")])
    return spring + topo + climate

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    df = df.dropna(subset=["s"]).reset_index(drop=True)
    
    tab_cols = get_tabular_cols(df)
    for col in tab_cols:
        df[col] = df[col].fillna(df[col].median())

    X = df[tab_cols].values
    y_orig = df["s"].values
    y_log = np.log1p(y_orig)
    farms = df["farm"].values
    unique_farms = np.unique(farms)
    
    print(f"Total farms: {len(unique_farms)}")

    test_sizes = [4, 5, 6, 7, 8]
    val_size = 3
    n_iterations = 15 # 15 random splits per test size to get stable statistics

    results = []

    for ts in test_sizes:
        metrics = {'rho': [], 'r2': [], 'rmse': [], 'mae': []}
        for seed in range(n_iterations):
            rng = np.random.default_rng(seed + ts * 100)
            shuffled_farms = rng.permutation(unique_farms)
            
            test_farms = shuffled_farms[:ts]
            val_farms = shuffled_farms[ts:ts+val_size]
            train_farms = shuffled_farms[ts+val_size:]
            
            test_idx = np.isin(farms, test_farms)
            val_idx = np.isin(farms, val_farms)
            train_idx = np.isin(farms, train_farms)
            
            X_tr, y_tr = X[train_idx], y_log[train_idx]
            X_va, y_va = X[val_idx], y_log[val_idx]
            X_te, y_te = X[test_idx], y_log[test_idx]
            
            model = xgb.XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=0, tree_method="hist",
                early_stopping_rounds=50
            )
            
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            preds_log = model.predict(X_te)
            preds = np.expm1(preds_log)
            y_te_orig = np.expm1(y_te)
            
            r2 = r2_score(y_te_orig, preds)
            rmse = np.sqrt(mean_squared_error(y_te_orig, preds))
            mae = mean_absolute_error(y_te_orig, preds)
            rho, _ = spearmanr(y_te_orig, preds)
            
            # Handle NaNs in spearmanr if predictions are constant
            if np.isnan(rho): rho = 0.0
            
            metrics['rho'].append(rho)
            metrics['r2'].append(r2)
            metrics['rmse'].append(rmse)
            metrics['mae'].append(mae)
            
        results.append({
            'Test_Farms': ts,
            'Train_Farms': len(unique_farms) - ts - val_size,
            'Val_Farms': val_size,
            'Mean_Rho': np.mean(metrics['rho']),
            'Std_Rho': np.std(metrics['rho']),
            'Mean_R2': np.mean(metrics['r2']),
            'Std_R2': np.std(metrics['r2']),
            'Mean_RMSE': np.mean(metrics['rmse']),
            'Mean_MAE': np.mean(metrics['mae'])
        })

    res_df = pd.DataFrame(results)
    print("\n=== Results of Farm Split Search (Averaged over 15 random splits) ===")
    print(res_df.round(3).to_string(index=False))

if __name__ == "__main__":
    main()