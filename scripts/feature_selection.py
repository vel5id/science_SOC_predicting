import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data/features/master_dataset.csv"
OUT_DIR = "data/features/selected"
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = ["ph", "soc", "no3", "p", "k", "s"]

# Tracking to avoid giving target or metadata features to the model
META_COLS = ['id', 'year', 'field_name', 'farm', 'grid_id', 'centroid_lon', 'centroid_lat', 'hu']

def get_feature_family(name):
    """
    Categorizes a feature into a 'family'.
    The rule is to avoid selecting highly correlated features from the same family
    (e.g., if NDVI_spring is selected, we don't pick NDVI_summer or delta_NDVI).
    """
    name = name.lower()
    
    # Mathematical combinations
    if 'diff' in name or 'ratio' in name or 'comp' in name:
        return 'math_composite'
    if 'delta' in name:
        return 'delta_indices'
        
    # Standard vegetation indices
    if 'ndvi' in name and 'gndvi' not in name: return 'ndvi'
    if 'gndvi' in name: return 'gndvi'
    if 'ndre' in name: return 'ndre'
    if 'savi' in name: return 'savi'
    if 'bsi' in name: return 'bsi'
    if 'evi' in name: return 'evi'
    if 'cl_red_edge' in name or 'cl_re' in name: return 'cl_red_edge'
    
    # Texture features
    if 'glcm' in name: 
        if 'asm' in name: return 'glcm_asm'
        if 'contrast' in name: return 'glcm_contrast'
        if 'ent' in name: return 'glcm_entropy'
        if 'idm' in name: return 'glcm_idm'
        return 'glcm_other'
        
    # Topography, Climate, Radar
    if 'temp' in name or 'map' in name or 'precip' in name: return 'climate'
    if 'elev' in name or 'slope' in name or 'aspect' in name or 'twi' in name or 'ls_' in name: return 'topo'
    if 'vv' in name or 'vh' in name or 'radar' in name: return 'radar'
    
    # Raw spectral bands
    import re
    band_match = re.search(r'_(b\d+a?)', name)
    if band_match: return f"band_{band_match.group(1)}"
    
    return 'other'

def select_best_features(df, target, n_features=15, corr_threshold=0.75):
    print(f"\n{'='*50}")
    print(f" Feature Selection for: {target.upper()}")
    print(f"{'='*50}")
    
    # 1. Prepare data
    valid_data = df.dropna(subset=[target])
    if valid_data.empty:
        print(f"No valid data for {target}.")
        return []
        
    y = valid_data[target].values
    drop_cols = TARGETS + META_COLS
    X_df = valid_data.drop(columns=[c for c in drop_cols if c in valid_data.columns])
    
    # Strictly select numeric columns to avoid string/geometry columns
    X_df = X_df.select_dtypes(include=[np.number])
    
    # Remove columns that are all NaNs
    X_df = X_df.dropna(axis=1, how='all')
    
    # 2. Impute missing values (median)
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X_df), columns=X_df.columns)
    
    # 3. Random Forest Feature Importance
    rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_imputed, y)
    
    importances = pd.Series(rf.feature_importances_, index=X_df.columns).sort_values(ascending=False)
    
    # 4. Iterative selection with constraints
    selected = []
    selection_info = []
    
    for feat, imp in importances.items():
        if len(selected) >= n_features:
            break
            
        fam = get_feature_family(feat)
        
        # ==========================================
        # 1. Мягкий фильтр по названиям (семействам)
        # ==========================================
        # Мы сильно ослабляем квоты, чтобы алгоритм сам решал на основе математики. 
        # Допускаем до 3-х признаков из одного спектрального семейства 
        # (например NDVI весной, летом и осенью — если они реально независимы).
        # Для климата, топографии и GLCM квоты тоже увеличены.
        if fam in ['ndvi', 'gndvi', 'ndre', 'savi', 'bsi', 'evi', 'cl_red_edge']:
            max_per_fam = 3
        elif fam in ['delta_indices', 'math_composite', 'climate', 'topo', 'radar', 'glcm']:
            max_per_fam = 4
        elif fam.startswith('band_') or fam.startswith('glcm_'):
            max_per_fam = 3
        else:
            max_per_fam = 4
            
        fam_count = sum(1 for s in selected if get_feature_family(s) == fam)
        if fam_count >= max_per_fam:
            # Отсекаем только если модель зациклилась на ОДНОМ семействе (защита от переобучения на шуме)
            continue
            
        # ==========================================
        # 2. ЖЕСТКИЙ МАТЕМАТИЧЕСКИЙ ФИЛЬТР (Корреляция Спирмена)
        # ==========================================
        # Если корреляция |ρ| >= 0.70 (R^2 > 0.49), значит переменные несут >50% общей дисперсии. 
        # С точки зрения информатики они избыточны (мультиколлинеарность).
        if len(selected) > 0:
            corr_abs = X_imputed[selected + [feat]].corr(method='spearman').abs()
            max_corr = corr_abs.loc[feat, selected].max()
            
            if max_corr >= corr_threshold:
                # Математически дублирующий сигнал, отбрасываем
                continue
                
        # If passed all checks, accept it
        selected.append(feat)
        selection_info.append({
            "feature": feat,
            "family": fam,
            "rf_importance": round(imp, 4)
        })
        
    print(f"Selected {len(selected)} best orthogonal features:")
    for i, info in enumerate(selection_info):
        print(f" {i+1:2d}. {info['feature']:<40} (Fam: {info['family']:<15} | RF Imp: {info['rf_importance']:.4f})")
        
    return selection_info

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Merge step might have failed.")
        exit(1)
        
    df = pd.read_csv(DATA_PATH)
    results = {}
    
    for t in TARGETS:
        if t in df.columns:
            features = select_best_features(df, t, n_features=15, corr_threshold=0.70)
            results[t] = features
            
    # Save full JSON report
    with open(os.path.join(OUT_DIR, "best_features.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # Save plain text lists for models
    for t, feats in results.items():
        with open(os.path.join(OUT_DIR, f"{t}_best_features.txt"), "w") as f:
            for feat_info in feats:
                f.write(feat_info["feature"] + "\n")
                
    print(f"\n[DONE] Saved optimized feature sets to {OUT_DIR}/")
