import pandas as pd
import numpy as np
import time
import csv
import os
import torch
import torch.nn.functional as F
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true) + 1e-12))

MODELS_DIR = 'ML/ml_models/tuned'
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv("data/features/master_dataset.csv", low_memory=False)
df_valid = df.dropna(subset=['ph', 'k', 'p']).reset_index(drop=True)
unique_locations = df_valid[['grid_id', 'centroid_lon', 'centroid_lat', 'sampling_date']].drop_duplicates()
base_df = df_valid.loc[unique_locations.index].copy()

targets = ['ph', 'k', 's', 'p', 'hu', 'no3']
all_results = []

param_grids = {
    'CART': {
        'model': DecisionTreeRegressor(random_state=42),
        'grid': {'max_depth': [10, 20, 30, 40, 50], 'max_leaf_nodes': [10, 20, 30, 40, 50]}
    },
    'ET': {
        'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
        'grid': {'max_depth': [100, 150, 200, 250, 300, 400], 'max_leaf_nodes': [50, 100, 150, 200, 250]}
    },
    'KNN': {
        'model': KNeighborsRegressor(n_jobs=-1),
        'grid': {'leaf_size': [50, 100, 150, 200, 250], 'n_neighbors': [2, 4, 6, 8, 10]}
    },
    'Ridge': {
        'model': Ridge(random_state=42),
        'grid': {'alpha': [1, 2, 4, 6, 8], 'tol': [0.00001, 0.001, 0.01, 0.1, 1]}
    },
    'SGD': {
        'model': SGDRegressor(random_state=42),
        'grid': {'alpha': [0.0001, 0.01, 0.1, 1, 10], 'tol': [0.001, 0.01, 0.1, 1, 10]}
    },
    'SVR': {
        'model': SVR(),
        'grid': {'gamma': [0.001, 0.01, 0.1, 1, 10], 'C': [0.0001, 0.01, 0.1, 1, 10]}
    },
    'GBDT': {
        'model': GradientBoostingRegressor(random_state=42),
        'grid': {'max_depth': [2, 4, 6, 8, 10], 'n_estimators': [50, 100, 150, 200, 300]}
    },
    'RF': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'grid': {'max_depth': [10, 20, 30, 40, 50], 'max_leaf_nodes': [50, 100, 150, 200, 250]}
    },
    'XGBoost': {
        'model': xgb.XGBRegressor(learning_rate=0.01, random_state=42, n_jobs=-1),
        'grid': {'n_estimators': [50, 100, 150, 200, 300], 'min_child_weight': [2, 4, 6, 8, 10]}
    }
}

for t in targets:
    print(f"\n============================================================")
    print(f"🚀 STARTING HYPERPARAMETER TUNING FOR TARGET: {t.upper()}")
    print(f"============================================================")

    target_df = base_df.copy()
    target_df = target_df.dropna(subset=[t]).reset_index(drop=True)
    
    farm_col = 'field_name'
    unique_farms = np.array(target_df[farm_col].unique().tolist())
    np.random.seed(42) # fixed single split
    np.random.shuffle(unique_farms)
    
    val_count = 10
    test_count = 12
    # For GridSearch, we merge train and val farms into one cross-validation pool
    tuning_farms = unique_farms[:-test_count] 
    test_farms = unique_farms[-test_count:]
    
    tuning_mask = target_df[farm_col].isin(tuning_farms)
    test_mask = target_df[farm_col].isin(test_farms)
    
    tuning_idx = target_df.index[tuning_mask].tolist()
    test_idx = target_df.index[test_mask].tolist()
    
    groups_tuning = target_df[farm_col].values[tuning_idx]
    
    feature_name = 'soc' if t == 'hu' else t
    feature_path = f"data/features/selected/{feature_name}_best_features.txt"
    if not os.path.exists(feature_path):
        print(f"Warning: {feature_path} not found. Skipping {t}.")
        continue
        
    with open(feature_path, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
        
    X_full = target_df[features].values
    y_full = target_df[t].values
    
    X_tuning, y_tuning = X_full[tuning_idx], y_full[tuning_idx]
    X_test, y_test   = X_full[test_idx], y_full[test_idx]
    
    def fill_nans(arr):
        col_medians = np.nanmedian(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_medians, inds[1])
        return arr
    
    X_tuning = fill_nans(X_tuning)
    X_test = fill_nans(X_test)

    scaler = StandardScaler()
    X_tun_s = scaler.fit_transform(X_tuning)
    X_test_s  = scaler.transform(X_test)
    
    target_models_dir = os.path.join(MODELS_DIR, t)
    os.makedirs(target_models_dir, exist_ok=True)
    
    # 5-fold CV based on Spatial Fields
    gkf = GroupKFold(n_splits=5)
    
    for model_name, config in param_grids.items():
        print(f"  Tuning {model_name}...")
        start_time = time.time()
        
        search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['grid'],
            cv=gkf.split(X_tun_s, y_tuning, groups_tuning),
            scoring='r2',
            n_jobs=-1,
            refit=True, # Will refit on entire X_tun_s using best params
            verbose=0
        )
        
        search.fit(X_tun_s, y_tuning)
        best_model = search.best_estimator_
        
        # Predict on Test (which was never seen by StandardScaler or GridSearchCV)
        test_preds = best_model.predict(X_test_s)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        mae = mean_absolute_error(y_test, test_preds)
        r2 = r2_score(y_test, test_preds)
        
        y_true_t = torch.tensor(y_test, dtype=torch.float32)
        y_pred_t = torch.tensor(test_preds, dtype=torch.float32)
        huber = F.huber_loss(y_pred_t, y_true_t).item()
        log_cosh = log_cosh_loss(y_pred_t, y_true_t).item()
        
        model_filename = os.path.join(target_models_dir, f"{model_name}_tuned.pkl")
        joblib.dump(best_model, model_filename)
        
        print(f"    Best params: {search.best_params_} -> Test R2: {r2:.4f}")
        
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

global_csv_path = 'ML/results/tuned_ml_ablation_split.csv'
with open(global_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)
print(f"\n🎉 TUNING COMPLETE! Saved logs to {global_csv_path}")
