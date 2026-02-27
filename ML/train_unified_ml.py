import pandas as pd
import numpy as np
import time
import csv
import os
import torch.nn.functional as F
import torch
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import catboost as cb

def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true) + 1e-12))

MODELS_DIR = 'ML/ml_models/baselines_ablation_split'
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv("data/features/master_dataset.csv", low_memory=False)
df_valid = df.dropna(subset=['ph', 'k', 'p']).reset_index(drop=True)
unique_locations = df_valid[['grid_id', 'centroid_lon', 'centroid_lat', 'sampling_date']].drop_duplicates()
base_df = df_valid.loc[unique_locations.index].copy()

targets = ['ph', 'k', 's', 'p', 'hu', 'no3']
all_results = []

for t in targets:
    print(f"\n============================================================")
    print(f"🚀 STARTING CLASSICAL ML BASELINES FOR TARGET: {t.upper()}")
    print(f"============================================================")

    target_df = base_df.copy()
    target_df = target_df.dropna(subset=[t]).reset_index(drop=True)
    
    # EXACT same split logic as train_ablation.py
    farm_col = 'field_name'
    unique_farms = np.array(target_df[farm_col].unique().tolist())
    np.random.seed(42) # fixed single split
    np.random.shuffle(unique_farms)
    
    val_count = 10
    test_count = 12
    train_farms = unique_farms[:-(val_count + test_count)]
    val_farms = unique_farms[-(val_count + test_count):-test_count]
    test_farms = unique_farms[-test_count:]
    
    train_idx = target_df.index[target_df[farm_col].isin(train_farms)].tolist()
    val_idx   = target_df.index[target_df[farm_col].isin(val_farms)].tolist()
    test_idx  = target_df.index[target_df[farm_col].isin(test_farms)].tolist()
    
    # Load 1D features from 'data/features/selected'
    feature_name = 'soc' if t == 'hu' else t
    feature_path = f"data/features/selected/{feature_name}_best_features.txt"
    if not os.path.exists(feature_path):
        print(f"Warning: {feature_path} not found. Skipping {t}.")
        continue
        
    with open(feature_path, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
        
    X_full = target_df[features].values
    y_full = target_df[t].values
    
    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val, y_val     = X_full[val_idx], y_full[val_idx]
    X_test, y_test   = X_full[test_idx], y_full[test_idx]
    
    # Fill Nans (median) if any exist in the 1D features
    def fill_nans(arr):
        col_medians = np.nanmedian(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_medians, inds[1])
        return arr
    
    X_train = fill_nans(X_train)
    X_val = fill_nans(X_val)
    X_test = fill_nans(X_test)

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    
    models = {
        'RF': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        'ET': ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.01, random_state=42, n_jobs=-1, early_stopping_rounds=20),
        'CatBoost': cb.CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.01, random_seed=42, verbose=0, early_stopping_rounds=20),
        'CART': DecisionTreeRegressor(random_state=42),
        'LR': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'SGD': SGDRegressor(random_state=42),
        'SVR': SVR(kernel='rbf', C=10.0),
        'GBDT': GradientBoostingRegressor(n_estimators=300, random_state=42)
    }
    
    target_models_dir = os.path.join(MODELS_DIR, t)
    os.makedirs(target_models_dir, exist_ok=True)
    
    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        start_time = time.time()
        
        if model_name == 'XGBoost':
            model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
        elif model_name == 'CatBoost':
            model.fit(X_train_s, y_train, eval_set=(X_val_s, y_val))
        else:
            # Train only on train set, strictly mimicking train_ablation CNN behavior
            model.fit(X_train_s, y_train)
            
        test_preds = model.predict(X_test_s)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        mae = mean_absolute_error(y_test, test_preds)
        r2 = r2_score(y_test, test_preds)
        
        y_true_t = torch.tensor(y_test, dtype=torch.float32)
        y_pred_t = torch.tensor(test_preds, dtype=torch.float32)
        huber = F.huber_loss(y_pred_t, y_true_t).item()
        log_cosh = log_cosh_loss(y_pred_t, y_true_t).item()
        
        # Save model
        model_filename = os.path.join(target_models_dir, f"{model_name}.pkl")
        joblib.dump(model, model_filename)
        
        all_results.append({
            'Target': t.upper(),
            'Model': model_name,
            'R2': round(r2, 4),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'Huber': round(huber, 4),
            'LogCosh': round(log_cosh, 4),
            'Time_s': round(elapsed_time, 1)
        })

global_csv_path = 'ML/results/classic_ml_ablation_split.csv'
with open(global_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)
print(f"\n🎉 FULL BASELINE EXPERIMENT COMPLETE! Saved logs to {global_csv_path}")
