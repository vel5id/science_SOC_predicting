#!/usr/bin/env python3
"""
generate_farm_lofo_figures.py
=============================
Generates Farm-LOFO analogues of Figure 3 and Figure 4:
  fig_farm_lofo_model_comparison.png  — bar chart of Spearman ρ (all 11 models)
  fig_farm_lofo_scatter.png           — predicted vs observed for best model/target

Saves to ./figures/
"""

import warnings
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent
DATA_CSV    = BASE / 'data' / 'features' / 'master_dataset.csv'
SELECTED    = BASE / 'data' / 'features' / 'selected'
RESULTS_CSV = BASE / 'ML' / 'results' / 'farm_lofo_all_models.csv'
FIG         = BASE / 'figures'
FIG.mkdir(exist_ok=True)

TARGETS     = ['ph', 'soc', 'no3', 'p', 'k', 's']
FARM_COL    = 'farm'
SEED        = 42

TARGET_LABELS = {
    'ph': 'pH (KCl)', 'soc': 'SOC (%)', 'no3': 'NO₃ (mg/kg)',
    'p':  'P₂O₅ (mg/kg)', 'k': 'K₂O (mg/kg)', 's': 'S (mg/kg)',
}

# Best model per target at Farm-LOFO (from farm_lofo_all_models.csv)
BEST_MODEL = {
    'ph': 'RF', 'soc': 'CatBoost', 'no3': 'RF',
    'p':  'SVR', 'k':  'RF',       's':   'XGBoost',
}

NEEDS_SCALING = {'KNN', 'LR', 'Ridge', 'SGD', 'SVR'}

# ── Style (match generate_figures.py) ─────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def load_features(target: str) -> list[str]:
    path = SELECTED / f'{target}_best_features.txt'
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def make_model(name: str):
    models = {
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
    return copy.deepcopy(models[name])


def farm_lofo_oof(df: pd.DataFrame, target: str, features: list[str],
                  model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (actual, predicted) OOF arrays for a given target/model."""
    sub = df.dropna(subset=[target]).reset_index(drop=True)
    farms = sub[FARM_COL].unique()
    all_preds = np.full(len(sub), np.nan)
    all_actual = sub[target].values

    for farm in farms:
        train_mask = (sub[FARM_COL] != farm).values
        test_mask  = (sub[FARM_COL] == farm).values
        X_tr = sub.loc[train_mask, features].values.copy()
        X_te = sub.loc[test_mask,  features].values.copy()
        y_tr = sub.loc[train_mask, target].values

        med = np.nanmedian(X_tr, axis=0)
        med[np.isnan(med)] = 0.0
        for c in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, c]), c] = med[c]
            X_te[np.isnan(X_te[:, c]), c] = med[c]

        if model_name in NEEDS_SCALING:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

        m = make_model(model_name)
        m.fit(X_tr, y_tr)
        all_preds[test_mask] = m.predict(X_te)

    valid = np.isfinite(all_preds) & np.isfinite(all_actual)
    return all_actual[valid], all_preds[valid]


# ══════════════════════════════════════════════════════════════════════
# Figure A: Bar chart — Farm-LOFO-CV model comparison (analogue of fig3)
# ══════════════════════════════════════════════════════════════════════
def fig_farm_lofo_bar():
    print('[Farm-LOFO Fig A] Bar chart — all 11 models ...')
    res = pd.read_csv(RESULTS_CSV)

    # Models to show (sorted roughly best→worst for pH)
    MODEL_ORDER = ['RF', 'CatBoost', 'ET', 'XGBoost', 'SVR', 'GBDT',
                   'SGD', 'Ridge', 'LR', 'CART', 'KNN']
    COLOURS = {
        'RF':       '#d73027', 'ET':    '#fc8d59', 'GBDT': '#fee090',
        'XGBoost':  '#91bfdb', 'CatBoost': '#4575b4', 'SVR': '#313695',
        'KNN':      '#74c476', 'LR':   '#969696',  'Ridge': '#bdbdbd',
        'SGD':      '#d9d9d9', 'CART': '#8856a7',
    }

    piv = res.pivot(index='model', columns='target', values='rho')
    piv = piv.reindex(MODEL_ORDER)[TARGETS]

    x = np.arange(len(TARGETS))
    n = len(MODEL_ORDER)
    width = 0.072
    fig, ax = plt.subplots(figsize=(13, 5))

    for j, model in enumerate(MODEL_ORDER):
        vals = piv.loc[model, TARGETS].values.astype(float)
        offset = (j - n / 2 + 0.5) * width
        ax.bar(x + offset, np.clip(vals, 0, None), width,
               label=model, color=COLOURS[model], edgecolor='white', linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGETS], fontsize=10)
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Model Comparison: Spearman ρ (Farm-LOFO-CV — Leave-One-Farm-Out, 20 хозяйств)')
    ax.set_ylim(0, 0.92)
    ax.legend(ncol=4, loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='grey', linestyle=':', alpha=0.4)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'fig_farm_lofo_model_comparison.{ext}')
    plt.close(fig)
    print('  ✓ Saved fig_farm_lofo_model_comparison.pdf/png')


# ══════════════════════════════════════════════════════════════════════
# Figure B: Scatter plots — predicted vs observed (Farm-LOFO best models)
# ══════════════════════════════════════════════════════════════════════
def fig_farm_lofo_scatter():
    print('[Farm-LOFO Fig B] Scatter plots — best model per target ...')

    df = pd.read_csv(DATA_CSV, low_memory=False)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    axes = axes.flatten()

    for i, tgt in enumerate(TARGETS):
        model_name = BEST_MODEL[tgt]
        features   = load_features(tgt)
        features   = [f for f in features if f in df.columns]

        print(f'  [{tgt}] running Farm-LOFO OOF with {model_name} ...')
        actual, pred = farm_lofo_oof(df, tgt, features, model_name)

        rho, _  = stats.spearmanr(actual, pred)
        r2      = 1 - np.sum((actual - pred)**2) / np.sum((actual - actual.mean())**2)
        rmse    = np.sqrt(np.mean((actual - pred)**2))

        ax = axes[i]
        # density-coloured scatter
        xy = np.vstack([actual, pred])
        try:
            from scipy.stats import gaussian_kde
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            ax.scatter(actual[idx], pred[idx], s=7, c=z[idx],
                       cmap='viridis', alpha=0.6, edgecolors='none')
        except Exception:
            ax.scatter(actual, pred, s=7, alpha=0.4, c='#2b83ba', edgecolors='none')

        mn = min(actual.min(), pred.min())
        mx = max(actual.max(), pred.max())
        margin = (mx - mn) * 0.05
        ax.plot([mn - margin, mx + margin], [mn - margin, mx + margin],
                'k--', lw=1, alpha=0.6)
        ax.set_xlim(mn - margin, mx + margin)
        ax.set_ylim(mn - margin, mx + margin)
        ax.set_xlabel(f'Observed {TARGET_LABELS[tgt]}')
        ax.set_ylabel(f'Predicted {TARGET_LABELS[tgt]}')
        ax.set_title(f'{model_name} \u2014 {TARGET_LABELS[tgt]}  [Farm-LOFO-CV, 20 хозяйств]')
        ax.set_aspect('equal')
        ax.text(0.05, 0.95,
                f'ρ = {rho:.3f}\nR² = {r2:.3f}\nRMSE = {rmse:.3f}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'fig_farm_lofo_scatter.{ext}')
    plt.close(fig)
    print('  ✓ Saved fig_farm_lofo_scatter.pdf/png')


if __name__ == '__main__':
    fig_farm_lofo_bar()
    fig_farm_lofo_scatter()
    print('\nDone.')
