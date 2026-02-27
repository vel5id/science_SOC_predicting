"""
rerun_s_clean.py  –  Re-run Field-LOFO and Farm-LOFO for S
using spring-only + static + temporal-aggregate features (no temporal leakage).
Prints new S metrics for all 11 models and updates the two results CSVs.
"""
import copy, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
DATA_CSV  = ROOT / 'data' / 'features' / 'master_dataset.csv'
SELECTED  = ROOT / 'data' / 'features' / 'selected'
FARM_CSV  = ROOT / 'ML' / 'results' / 'farm_lofo_all_models.csv'
FIELD_CSV = ROOT / 'ML' / 'results' / 'all_models_comparison.csv'
SEED = 42; TARGET = 's'; FARM_COL = 'farm'
NEEDS_SCALING = {'KNN', 'LR', 'Ridge', 'SGD', 'SVR'}

def make_model(name):
    zoo = {
        'RF':       RandomForestRegressor(n_estimators=500, max_features='sqrt',
                        min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        'ET':       ExtraTreesRegressor(n_estimators=500, max_features='sqrt',
                        min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        'XGBoost':  xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.8, reg_lambda=1.0, random_state=SEED,
                        n_jobs=-1, verbosity=0),
        'CatBoost': cb.CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                        random_seed=SEED, verbose=0),
        'GBDT':     GradientBoostingRegressor(n_estimators=300, max_depth=5,
                        learning_rate=0.1, subsample=0.8, random_state=SEED),
        'CART':     DecisionTreeRegressor(max_depth=10, min_samples_leaf=5,
                        random_state=SEED),
        'KNN':      KNeighborsRegressor(n_neighbors=7, weights='distance', n_jobs=-1),
        'LR':       LinearRegression(),
        'Ridge':    Ridge(alpha=1.0, random_state=SEED),
        'SGD':      SGDRegressor(alpha=1e-4, penalty='l2', max_iter=1000,
                        random_state=SEED),
        'SVR':      SVR(kernel='rbf', C=1.0, epsilon=0.1),
    }
    return copy.deepcopy(zoo[name])

MODELS = ['RF', 'ET', 'XGBoost', 'CatBoost', 'GBDT', 'CART',
          'KNN', 'LR', 'Ridge', 'SGD', 'SVR']

def prep(X_tr, X_te, name):
    med = np.nanmedian(X_tr, axis=0); med[np.isnan(med)] = 0.0
    for c in range(X_tr.shape[1]):
        X_tr[np.isnan(X_tr[:, c]), c] = med[c]
        X_te[np.isnan(X_te[:, c]), c] = med[c]
    if name in NEEDS_SCALING:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    return X_tr, X_te

def metrics(y, p):
    rho, _ = spearmanr(y, p)
    r2 = r2_score(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    mae  = mean_absolute_error(y, p)
    return round(rho, 4), round(r2, 4), round(rmse, 4), round(mae, 4)

df = pd.read_csv(DATA_CSV, low_memory=False)
with open(SELECTED / 's_best_features.txt') as f:
    features = [l.strip() for l in f if l.strip()]
features = [ft for ft in features if ft in df.columns]
print(f'Clean S features ({len(features)}): {features}')

sub = df.dropna(subset=[TARGET]).reset_index(drop=True)
print(f'Samples: {len(sub)}')

# ── Farm-LOFO ─────────────────────────────────────────────────
farms = sub[FARM_COL].unique()
print(f'Farms: {len(farms)}\n--- Farm-LOFO results ---')
farm_rows = []
for mname in MODELS:
    all_pred  = np.full(len(sub), np.nan)
    for farm in farms:
        tr = (sub[FARM_COL] != farm).values
        te = (sub[FARM_COL] == farm).values
        Xtr, Xte = prep(sub.loc[tr, features].values.copy(),
                         sub.loc[te, features].values.copy(), mname)
        ytr = sub.loc[tr, TARGET].values
        m = make_model(mname); m.fit(Xtr, ytr)
        all_pred[te] = m.predict(Xte)
    ok  = np.isfinite(all_pred)
    rho, r2, rmse, mae = metrics(sub[TARGET].values[ok], all_pred[ok])
    print(f'  {mname:10s}: rho={rho:7.4f}  R2={r2:7.4f}  RMSE={rmse:7.4f}  MAE={mae:7.4f}')
    farm_rows.append({'target': TARGET, 'model': mname, 'rho': rho,
                      'r2': r2, 'rmse': rmse, 'mae': mae,
                      'n': int(ok.sum()), 'n_farms': len(farms), 'time_s': 0})

# ── Field-LOFO ────────────────────────────────────────────────
if 'field' in sub.columns:
    fields_col = 'field'
elif 'Field' in sub.columns:
    fields_col = 'Field'
else:
    # try to find field column
    candidates = [c for c in sub.columns if 'field' in c.lower() or 'lofo' in c.lower()]
    fields_col = candidates[0] if candidates else None

print(f'\nField column: {fields_col}')
field_rows = []
if fields_col:
    fields = sub[fields_col].unique()
    print(f'Fields: {len(fields)}\n--- Field-LOFO results ---')
    for mname in MODELS:
        all_pred = np.full(len(sub), np.nan)
        for fld in fields:
            tr = (sub[fields_col] != fld).values
            te = (sub[fields_col] == fld).values
            Xtr, Xte = prep(sub.loc[tr, features].values.copy(),
                             sub.loc[te, features].values.copy(), mname)
            ytr = sub.loc[tr, TARGET].values
            m = make_model(mname); m.fit(Xtr, ytr)
            all_pred[te] = m.predict(Xte)
        ok = np.isfinite(all_pred)
        rho, r2, rmse, mae = metrics(sub[TARGET].values[ok], all_pred[ok])
        print(f'  {mname:10s}: rho={rho:7.4f}  R2={r2:7.4f}  RMSE={rmse:7.4f}  MAE={mae:7.4f}')
        field_rows.append({'model': mname, 'rho': rho, 'r2': r2, 'rmse': rmse, 'mae': mae})
else:
    print('  No field column found — skipping Field-LOFO')

# ── Update Farm-LOFO CSV ──────────────────────────────────────
farm_df = pd.read_csv(FARM_CSV)
farm_df = farm_df[farm_df['target'] != TARGET]
new_farm = pd.DataFrame(farm_rows)
farm_df = pd.concat([farm_df, new_farm], ignore_index=True)
farm_df.to_csv(FARM_CSV, index=False)
print(f'\nUpdated {FARM_CSV}')

# ── Update Field-LOFO CSV ─────────────────────────────────────
if field_rows:
    field_df = pd.read_csv(FIELD_CSV)
    # The field CSV uses display labels – find the S label
    s_labels = [t for t in field_df['Target'].unique() if t.startswith('S')]
    print(f'S labels in field CSV: {s_labels}')
    if s_labels:
        slbl = s_labels[0]
        field_df = field_df[field_df['Target'] != slbl]
        new_field = pd.DataFrame([
            {'Target': slbl, 'Model': r['model'], 'Spearman_rho': r['rho'],
             'RMSE': r['rmse'], 'R2': r['r2']}
            for r in field_rows
        ])
        field_df = pd.concat([field_df, new_field], ignore_index=True)
        field_df.to_csv(FIELD_CSV, index=False)
        print(f'Updated {FIELD_CSV}')

print('\nDone.')
