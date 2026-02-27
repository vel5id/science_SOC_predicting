#!/usr/bin/env python3
"""
Generate figures requested by reviewer:
1. Friedman + Nemenyi test on per-field LOFO folds
2. S scatter plot (pred vs obs) with R² and ρ annotated
3. K=15 feature-count vs quality curve
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
import matplotlib.ticker as ticker

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

MODELS = {
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
    """Load OOF predictions for a model+target."""
    model_dir = MODELS[model_key]
    fname = f'{target}_oof_predictions.csv'
    fpath = model_dir / fname
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)
    return df


def compute_per_field_rho(df, target):
    """Compute per-field Spearman rho, returning a dict {field: rho}."""
    ycol = TARGET_COLS[target]
    field_rhos = {}
    for field, grp in df.groupby('field_name'):
        if len(grp) < 3:
            continue
        y_true = grp[ycol].values
        y_pred = grp['oof_pred'].values
        if np.std(y_true) < 1e-10 or np.std(y_pred) < 1e-10:
            continue
        rho, _ = stats.spearmanr(y_true, y_pred)
        field_rhos[field] = rho
    return field_rhos


# ========================== FIGURE 1: FRIEDMAN + NEMENYI ==========================
def friedman_nemenyi():
    """Run Friedman test on per-field-fold ρ across 11 ML models."""
    print("=== Friedman + Nemenyi Test ===")

    all_results = {}  # target -> model -> [per-field-rhos]

    for target in TARGETS:
        all_results[target] = {}
        for model_key in MODELS:
            df = load_oof(model_key, target)
            if df is None:
                print(f"  WARN: {model_key}/{target} not found")
                continue
            field_rhos = compute_per_field_rho(df, target)
            all_results[target][model_key] = field_rhos

    # For the Friedman test, use aggregated OOF rho across all 6 targets
    # Alternative: per-target Friedman
    # We'll do per-target and an overall

    friedman_results = {}
    for target in TARGETS:
        data = all_results[target]
        common_fields = None
        for mk, fr in data.items():
            if common_fields is None:
                common_fields = set(fr.keys())
            else:
                common_fields &= set(fr.keys())

        if common_fields is None or len(common_fields) < 5:
            print(f"  Skip {target}: too few common fields ({len(common_fields) if common_fields else 0})")
            continue

        common_fields = sorted(common_fields)
        model_keys = sorted(data.keys())

        # Build matrix: fields x models
        matrix = np.zeros((len(common_fields), len(model_keys)))
        for j, mk in enumerate(model_keys):
            for i, field in enumerate(common_fields):
                matrix[i, j] = data[mk].get(field, np.nan)

        # Friedman test
        stat, p = stats.friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
        print(f"  {TARGET_LABELS[target]}: χ²={stat:.2f}, p={p:.2e}, n_fields={len(common_fields)}, n_models={len(model_keys)}")
        friedman_results[target] = {
            'chi2': stat, 'p': p,
            'n_fields': len(common_fields),
            'n_models': len(model_keys),
            'matrix': matrix,
            'model_keys': model_keys,
            'common_fields': common_fields,
        }

    # ---- Nemenyi post-hoc (average rank based) ----
    # We compute average ranks for each target and create a CD diagram-style plot

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        if target not in friedman_results:
            ax.set_visible(False)
            continue

        res = friedman_results[target]
        matrix = res['matrix']
        model_keys = res['model_keys']
        n_fields = matrix.shape[0]
        n_models = matrix.shape[1]

        # Compute ranks per field (higher rho = rank 1)
        ranks = np.zeros_like(matrix)
        for i in range(n_fields):
            ranks[i] = stats.rankdata(-matrix[i])  # negative because higher is better

        avg_ranks = ranks.mean(axis=0)

        # Nemenyi critical difference
        q_alpha = 3.219  # q_0.05 for 11 groups (approx from tables)
        cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6.0 * n_fields))

        # Sort by avg rank
        order = np.argsort(avg_ranks)

        labels = [MODEL_LABELS[model_keys[o]] for o in order]
        avg_r = avg_ranks[order]

        # Plot horizontal CD diagram
        y_positions = np.arange(len(labels))
        colors = ['#2ca02c' if r <= avg_r[0] + cd else '#d62728' if r > avg_r[2] + cd else '#1f77b4' for r in avg_r]
        ax.barh(y_positions, avg_r, color=colors, edgecolor='black', linewidth=0.5, height=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Mean rank', fontsize=10)
        ax.set_title(f'{TARGET_LABELS[target]}  (χ²={res["chi2"]:.1f}, p={res["p"]:.1e})',
                      fontsize=11, fontweight='bold')
        ax.axvline(avg_r[0] + cd, color='red', linestyle='--', alpha=0.6, label=f'CD={cd:.2f}')
        ax.legend(fontsize=8, loc='lower right')
        ax.invert_yaxis()

    plt.suptitle('Friedman test: mean model ranks across Field-LOFO folds', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = FIG_DIR / 'fig10_friedman_nemenyi.png'
    plt.savefig(fig_path)
    plt.close()
    print(f"  Saved: {fig_path}")

    # Save numeric results
    results_json = {}
    for target in TARGETS:
        if target not in friedman_results:
            continue
        res = friedman_results[target]
        matrix = res['matrix']
        model_keys = res['model_keys']
        ranks = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            ranks[i] = stats.rankdata(-matrix[i])
        avg_ranks = ranks.mean(axis=0)
        results_json[target] = {
            'chi2': float(res['chi2']),
            'p_value': float(res['p']),
            'n_fields': res['n_fields'],
            'avg_ranks': {MODEL_LABELS[model_keys[j]]: float(avg_ranks[j])
                          for j in range(len(model_keys))},
        }
    with open(FIG_DIR / 'friedman_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved: {FIG_DIR / 'friedman_results.json'}")

    return friedman_results


# ========================== FIGURE 2: S SCATTER PLOT ==========================
def sulfur_scatter():
    """Scatter plot of predicted vs observed S with R² and ρ annotated."""
    print("\n=== S Scatter Plot ===")

    df = load_oof('rf', 's')
    if df is None:
        print("  ERROR: RF S OOF not found")
        return

    y_true = df['s'].values
    y_pred = df['oof_pred'].values

    rho, p_rho = stats.spearmanr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    scatter = ax.scatter(y_true, y_pred, alpha=0.4, s=15, c='steelblue', edgecolor='none')

    # Add density coloring
    ax.set_xlabel('Observed S (mg/kg)', fontsize=12)
    ax.set_ylabel('Predicted S (mg/kg)', fontsize=12)

    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')

    # Annotate
    textstr = f'Spearman ρ = {rho:.3f}\n$R^2$ = {r2:.3f}\nn = {len(y_true)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    ax.set_title('Predicted vs. Observed: Sulfur (S)\nRF, Field-LOFO-CV', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    fig_path = FIG_DIR / 'fig11_sulfur_scatter.png'
    plt.savefig(fig_path)
    plt.close()
    print(f"  Saved: {fig_path}")
    print(f"  ρ={rho:.3f}, R²={r2:.3f}")


# ========================== FIGURE 3: K=15 FEATURE CURVE ==========================
def feature_count_curve():
    """Generate 'number of features vs quality' curve for K=15 justification."""
    print("\n=== Feature Count Curve ===")

    # We'll build this by training RF with varying K on a per-property basis
    # using the full dataset and measuring OOF rho
    # But for speed, we approximate by loading the selected features and
    # progressively adding them

    features_file = BASE / 'data' / 'features' / 'selected' / 'best_features.json'
    if not features_file.exists():
        print(f"  ERROR: {features_file} not found")
        return

    with open(features_file) as f:
        best_features = json.load(f)

    master_file = BASE / 'data' / 'features' / 'master_dataset.csv'
    if not master_file.exists():
        print(f"  ERROR: {master_file} not found")
        return

    print("  Loading master dataset...")
    master = pd.read_csv(master_file)
    print(f"  Shape: {master.shape}")

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        ycol = TARGET_COLS[target]

        if ycol not in master.columns:
            print(f"  WARN: {ycol} not in master")
            ax.set_visible(False)
            continue

        # Get feature ranking for this target
        if target not in best_features:
            # Try alternate keys
            alt_keys = {'ph': 'ph', 'soc': 'soc', 'no3': 'no3', 'p': 'p', 'k': 'k', 's': 's'}
            key = alt_keys.get(target, target)
            if key not in best_features:
                print(f"  WARN: {target} not in best_features")
                ax.set_visible(False)
                continue
        else:
            key = target

        feat_list = best_features[key]
        if isinstance(feat_list, dict):
            # Features with importances
            sorted_feats = sorted(feat_list.items(), key=lambda x: -x[1])
            feat_names = [f[0] for f in sorted_feats]
        elif isinstance(feat_list, list):
            if isinstance(feat_list[0], dict):
                feat_names = [f['feature'] for f in feat_list]
            else:
                feat_names = feat_list
        else:
            print(f"  WARN: unexpected format for {target}")
            continue

        # Ensure features exist in master
        available_feats = [f for f in feat_names if f in master.columns]
        if len(available_feats) < 3:
            print(f"  WARN: insufficient features for {target}")
            continue

        y = master[ycol].values
        groups = master['field_name'].values

        K_values = list(range(1, min(len(available_feats) + 1, 26)))
        rho_means = []
        rho_stds = []

        for K in K_values:
            feats = available_feats[:K]
            X = master[feats].values

            # Quick 5-fold group CV instead of full 81-fold LOFO
            gkf = GroupKFold(n_splits=min(10, len(set(groups))))
            rhos = []
            for train_idx, test_idx in gkf.split(X, y, groups):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                rf = RandomForestRegressor(n_estimators=100, max_features='sqrt',
                                           min_samples_leaf=3, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                pred = rf.predict(X_test)
                if np.std(y_test) > 1e-10 and np.std(pred) > 1e-10:
                    rho, _ = stats.spearmanr(y_test, pred)
                    rhos.append(rho)

            rho_means.append(np.mean(rhos))
            rho_stds.append(np.std(rhos))

        # Plot
        rho_means = np.array(rho_means)
        rho_stds = np.array(rho_stds)
        ax.plot(K_values, rho_means, 'o-', color='steelblue', markersize=4, linewidth=1.5)
        ax.fill_between(K_values, rho_means - rho_stds, rho_means + rho_stds,
                         alpha=0.2, color='steelblue')
        ax.axvline(15, color='red', linestyle='--', alpha=0.7, label='K=15')
        ax.set_xlabel('Number of features (K)', fontsize=10)
        ax.set_ylabel('Spearman ρ', fontsize=10)
        ax.set_title(TARGET_LABELS[target], fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle('Number of features vs. quality (RF, 10-fold GroupKFold)',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = FIG_DIR / 'fig_appendix_feature_curve.png'
    plt.savefig(fig_path)
    plt.close()
    print(f"  Saved: {fig_path}")


if __name__ == '__main__':
    friedman_nemenyi()
    sulfur_scatter()
    feature_count_curve()
    print("\n=== All figures generated ===")
