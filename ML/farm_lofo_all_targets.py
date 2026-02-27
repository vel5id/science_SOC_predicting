"""
farm_lofo_all_targets.py
========================
Выполняет farm-LOFO (leave-one-farm-out CV) для всех 6 таргетов.
Использует RF (best tabular model) с 15 признаками (spectral + topo + climate).
Сохраняет результаты в ML/results/farm_lofo_all_targets.csv
"""
import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
DATA_CSV = ROOT / "data" / "features" / "master_dataset.csv"
OUT_CSV  = ROOT / "ML" / "results" / "farm_lofo_all_targets.csv"

TARGETS  = ["ph", "soc", "no3", "p", "k", "s"]
FARM_COL = "farm"
FIELD_COL = "field_name"
SEED = 42

RF_PARAMS = dict(
    n_estimators=400, max_depth=None, min_samples_leaf=2,
    max_features=0.5, random_state=SEED, n_jobs=-1,
)

def get_features(df):
    """Return ~15 tabular features used in production models."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    spring  = [c for c in num_cols if "_spring" in c and "_summer" not in c
               and "_autumn" not in c and "_late_summer" not in c]
    topo    = [c for c in num_cols if c.startswith("topo_")]
    climate = [c for c in num_cols if c.startswith("climate_")]
    feats   = sorted(set(spring + topo + climate))
    # exclude target columns from features
    feats = [c for c in feats if c not in TARGETS and c not in
             ["ph","soc","no3","p","k","s","hu","hum"]]
    return feats

def farm_lofo_cv(df, target_col, features):
    """Run farm-LOFO CV and return aggregated metrics."""
    sub = df.dropna(subset=[target_col]).copy()
    farms = sub[FARM_COL].unique()
    
    all_preds  = np.full(len(sub), np.nan)
    all_actual = sub[target_col].values
    
    for farm in farms:
        train_mask = sub[FARM_COL] != farm
        test_mask  = sub[FARM_COL] == farm
        X_train = sub.loc[train_mask, features].fillna(0)
        y_train = sub.loc[train_mask, target_col]
        X_test  = sub.loc[test_mask, features].fillna(0)
        
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        all_preds[test_mask.values] = preds
    
    valid = np.isfinite(all_preds) & np.isfinite(all_actual)
    rho, _  = spearmanr(all_actual[valid], all_preds[valid])
    r2      = r2_score(all_actual[valid], all_preds[valid])
    rmse    = np.sqrt(mean_squared_error(all_actual[valid], all_preds[valid]))
    mae     = np.mean(np.abs(all_actual[valid] - all_preds[valid]))
    return dict(target=target_col, rho=round(rho,4), r2=round(r2,4),
                rmse=round(rmse,4), mae=round(mae,4), n=int(valid.sum()),
                n_farms=len(farms))

print("Loading master dataset ...")
df = pd.read_csv(DATA_CSV, low_memory=False)
print(f"  {len(df)} rows, {df[FARM_COL].nunique()} farms")

features = get_features(df)
# limit to ~90 features matching production setup
if len(features) > 120:
    features = features[:90]
print(f"  Using {len(features)} features")

records = []
for tgt in TARGETS:
    if tgt not in df.columns:
        print(f"  [{tgt}] NOT FOUND, skipping")
        continue
    print(f"  [{tgt}] running farm-LOFO ...")
    rec = farm_lofo_cv(df, tgt, features)
    records.append(rec)
    print(f"    rho={rec['rho']:.4f}  R2={rec['r2']:.4f}  RMSE={rec['rmse']:.4f}  n={rec['n']}")

out = pd.DataFrame(records)
out.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}")
print(out.to_string(index=False))
