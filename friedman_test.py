#!/usr/bin/env python3
"""
Friedman + Nemenyi test using per-field Mean Absolute Error (MAE) 
as the fold-level metric across 81 LOFO folds × 11 ML models.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

BASE = Path('/home/h621l/projects/science-article')
ML_RESULTS = BASE / 'ML' / 'results'
FIG_DIR = BASE / 'articles' / 'article2_prediction' / 'figures'
FIG_DIR.mkdir(exist_ok=True)

TARGETS = ['ph', 'soc', 'no3', 'p', 'k', 's']
TARGET_LABELS = {'ph': 'pH', 'soc': 'SOC', 'no3': r'NO$_3$',
                 'p': r'P$_2$O$_5$', 'k': r'K$_2$O', 's': 'S'}
TARGET_COLS = {'ph': 'ph', 'soc': 'hu', 'no3': 'no3', 'p': 'p', 'k': 'k', 's': 's'}

MODEL_ORDER = ['GBDT', 'rf', 'ET', 'xgb', 'catboost', 'SVR', 'KNN', 'LR', 'Ridge', 'SGD', 'cart']
MODEL_DIRS = {
    'rf': ML_RESULTS / 'rf',
    'xgb': ML_RESULTS / 'xgb',
    'catboost': ML_RESULTS / 'catboost',
    'cart': ML_RESULTS / 'cart',
    'ET': ML_RESULTS / 'baselines' / 'ET',
    'KNN': ML_RESULTS / 'baselines' / 'KNN',
    'SVR': ML_RESULTS / 'baselines' / 'SVR',
    'LR': ML_RESULTS / 'baselines' / 'LR',
    'Ridge': ML_RESULTS / 'baselines' / 'Ridge',
    'SGD': ML_RESULTS / 'baselines' / 'SGD',
    'GBDT': ML_RESULTS / 'baselines' / 'GBDT',
}
MODEL_LABELS = {
    'rf': 'RF', 'xgb': 'XGBoost', 'catboost': 'CatBoost',
    'cart': 'CART', 'ET': 'ET', 'KNN': 'KNN', 'SVR': 'SVR',
    'LR': 'LR', 'Ridge': 'Ridge', 'SGD': 'SGD', 'GBDT': 'GBDT'
}

def load_oof(model_key, target):
    model_dir = MODEL_DIRS[model_key]
    fname = f'{target}_oof_predictions.csv'
    fpath = model_dir / fname
    if not fpath.exists():
        return None
    return pd.read_csv(fpath, low_memory=False)


def compute_per_field_mae(df, target):
    """Returns Series: field_name -> MAE."""
    ycol = TARGET_COLS[target]
    df = df.copy()
    df['abs_err'] = (df[ycol] - df['oof_pred']).abs()
    return df.groupby('field_name')['abs_err'].mean()


def friedman_nemenyi():
    print("=== Friedman + Nemenyi Test (per-field MAE) ===\n")
    
    all_friedman = {}
    
    for target in TARGETS:
        # Collect per-field MAE for each model
        model_maes = {}
        for mk in MODEL_ORDER:
            df = load_oof(mk, target)
            if df is None:
                continue
            mae_series = compute_per_field_mae(df, target)
            model_maes[mk] = mae_series
        
        if len(model_maes) < 3:
            print(f"  Skip {target}: too few models")
            continue
        
        # Find common fields
        common_fields = None
        for mk, ms in model_maes.items():
            if common_fields is None:
                common_fields = set(ms.index)
            else:
                common_fields &= set(ms.index)
        
        common_fields = sorted(common_fields)
        n_fields = len(common_fields)
        n_models = len(model_maes)
        model_keys = [mk for mk in MODEL_ORDER if mk in model_maes]
        
        # Build matrix: fields × models (MAE, lower is better)
        matrix = np.zeros((n_fields, n_models))
        for j, mk in enumerate(model_keys):
            for i, field in enumerate(common_fields):
                matrix[i, j] = model_maes[mk][field]
        
        # Friedman test
        stat, p = stats.friedmanchisquare(*[matrix[:, j] for j in range(n_models)])
        print(f"  {TARGET_LABELS[target]}: χ²={stat:.2f}, p={p:.2e}, "
              f"n_fields={n_fields}, n_models={n_models}")
        
        # Compute average ranks (lower MAE = rank 1)
        ranks = np.zeros_like(matrix)
        for i in range(n_fields):
            ranks[i] = stats.rankdata(matrix[i])  # lower MAE → lower rank → better
        
        avg_ranks = ranks.mean(axis=0)
        
        # Nemenyi CD
        # q_alpha for k=11, alpha=0.05 ≈ 3.219
        q_alpha = 3.219
        cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6.0 * n_fields))
        
        all_friedman[target] = {
            'chi2': float(stat),
            'p_value': float(p),
            'n_fields': n_fields,
            'cd': float(cd),
            'avg_ranks': {MODEL_LABELS[model_keys[j]]: float(avg_ranks[j])
                          for j in range(n_models)},
            'model_keys': model_keys,
            'avg_ranks_arr': avg_ranks,
        }
    
    # ---- Plot CD diagrams ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    
    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        if target not in all_friedman:
            ax.set_visible(False)
            continue
        
        res = all_friedman[target]
        model_keys = res['model_keys']
        avg_ranks = res['avg_ranks_arr']
        cd = res['cd']
        n_models = len(model_keys)
        
        # Sort by avg rank (ascending = better)
        order = np.argsort(avg_ranks)
        labels = [MODEL_LABELS[model_keys[o]] for o in order]
        avg_r = avg_ranks[order]
        
        # Color: green if within CD of best, red if outside CD of best + CD
        best_rank = avg_r[0]
        colors = []
        for r in avg_r:
            if abs(r - best_rank) <= cd:
                colors.append('#2ca02c')  # not significantly different from best
            else:
                colors.append('#d62728')  # significantly worse than best
        
        y_pos = np.arange(n_models)
        bars = ax.barh(y_pos, avg_r, color=colors, edgecolor='black', linewidth=0.5, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Mean rank (lower = better)', fontsize=10)
        
        sig_str = '***' if res['p_value'] < 0.001 else '**' if res['p_value'] < 0.01 else '*' if res['p_value'] < 0.05 else 'n.s.'
        ax.set_title(f'{TARGET_LABELS[target]}  (χ²={res["chi2"]:.1f}, p={res["p_value"]:.1e} {sig_str})\nCD={cd:.2f}',
                      fontsize=11, fontweight='bold')
        
        # CD line from best
        ax.axvline(best_rank + cd, color='red', linestyle='--', alpha=0.6, linewidth=1.2,
                   label=f'Best + CD ({best_rank:.1f}+{cd:.1f})')
        ax.legend(fontsize=7, loc='lower right')
        ax.invert_yaxis()
        ax.set_xlim(0, max(avg_r) * 1.15)
    
    plt.suptitle('Friedman + Nemenyi test: mean model ranks by MAE\n(Field-LOFO-CV, 81 folds)',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = FIG_DIR / 'fig10_friedman_nemenyi.png'
    plt.savefig(fig_path)
    plt.close()
    print(f"\n  Saved: {fig_path}")
    
    # Save JSON results
    results_out = {}
    for target in TARGETS:
        if target not in all_friedman:
            continue
        r = all_friedman[target]
        results_out[target] = {
            'chi2': r['chi2'],
            'p_value': r['p_value'],
            'n_fields': r['n_fields'],
            'cd': r['cd'],
            'avg_ranks': r['avg_ranks'],
            'significant_p005': r['p_value'] < 0.05,
        }
    
    with open(FIG_DIR / 'friedman_results.json', 'w') as f:
        json.dump(results_out, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {FIG_DIR / 'friedman_results.json'}")
    
    return all_friedman


if __name__ == '__main__':
    friedman_nemenyi()
    print("\nDone!")
