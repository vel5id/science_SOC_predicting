#!/usr/bin/env python3
"""Generate all publication-quality figures for the article."""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle as MplRect
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(BASE, 'ML', 'results')
FIG  = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────
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

TARGET_LABELS = {
    'ph': 'pH (KCl)', 'soc': 'SOC (%)', 'no3': 'NO₃ (mg/kg)',
    'p': 'P₂O₅ (mg/kg)', 'k': 'K₂O (mg/kg)', 's': 'S (mg/kg)',
}
TARGET_ORDER = ['ph', 'soc', 'no3', 'p', 'k', 's']

# ══════════════════════════════════════════════════════════════════
# Figure 1: Study area map (sampling locations on Kazakhstan map)
# ══════════════════════════════════════════════════════════════════
def fig1_study_area():
    print('[Figure 1] Study area map with basemap ...')
    # Use XGBoost OOF file which has centroid_lon/lat
    df = pd.read_csv(os.path.join(RES, 'xgb', 'ph_oof_predictions.csv'), low_memory=False)
    
    # Get unique farms and assign colours
    farms = df['farm'].unique()
    cmap = plt.cm.get_cmap('tab20', len(farms))
    farm_to_c = {f: cmap(i) for i, f in enumerate(farms)}
    
    # ── Study area extent ─────────────────────────────────
    lon_min, lon_max = df['centroid_lon'].min(), df['centroid_lon'].max()
    lat_min, lat_max = df['centroid_lat'].min(), df['centroid_lat'].max()
    pad_lon = (lon_max - lon_min) * 0.08
    pad_lat = (lat_max - lat_min) * 0.10
    ext = [lon_min - pad_lon, lon_max + pad_lon,
           lat_min - pad_lat, lat_max + pad_lat]
    
    # ── Kazakhstan overview extent ────────────────────────
    kz_ext = [46, 88, 40, 56]  # [lon_min, lon_max, lat_min, lat_max]
    
    proj = ccrs.PlateCarree()

    # ── Two-panel figure: overview + detail ───────────────
    fig = plt.figure(figsize=(14, 6))
    
    # --- Panel A: Overview Kazakhstan map ---
    ax_ov = fig.add_axes([0.02, 0.08, 0.38, 0.84], projection=proj)
    ax_ov.set_extent(kz_ext, crs=proj)
    ax_ov.add_feature(cfeature.LAND, facecolor='#f0efe8', edgecolor='none')
    ax_ov.add_feature(cfeature.OCEAN, facecolor='#d4eaf7')
    ax_ov.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor='#444444')
    ax_ov.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax_ov.add_feature(cfeature.LAKES, facecolor='#d4eaf7', edgecolor='#888888', linewidth=0.3)
    ax_ov.add_feature(cfeature.RIVERS, edgecolor='#a0c4e8', linewidth=0.3)
    
    gl = ax_ov.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5,
                          color='grey', linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    
    # Red study-area rectangle
    rect = MplRect((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2],
                   linewidth=2.5, edgecolor='red', facecolor='red', alpha=0.15,
                   transform=proj, zorder=5)
    ax_ov.add_patch(rect)
    ax_ov.plot([ext[0], ext[1], ext[1], ext[0], ext[0]],
               [ext[2], ext[2], ext[3], ext[3], ext[2]],
               color='red', linewidth=2, transform=proj, zorder=6)
    
    # Kazakhstan label
    ax_ov.text(67, 48, 'Kazakhstan', fontsize=12, fontweight='bold',
               ha='center', transform=proj, color='#333333',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Neighbour labels
    for name, lon, lat, sz in [('Russia', 67, 55.3, 8),
                                ('China', 82, 43, 8),
                                ('Uzbekistan', 60, 41.5, 7),
                                ('Kyrgyzstan', 74, 41, 7)]:
        ax_ov.text(lon, lat, name, fontsize=sz, ha='center', transform=proj,
                   color='#777777', fontstyle='italic')
    
    ax_ov.set_title('(a) Location within Kazakhstan', fontsize=11, pad=8)
    
    # --- Panel B: Detailed study area ---
    ax_dt = fig.add_axes([0.46, 0.08, 0.52, 0.84], projection=proj)
    ax_dt.set_extent(ext, crs=proj)
    ax_dt.add_feature(cfeature.LAND, facecolor='#f5f4ef', edgecolor='none')
    ax_dt.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#888888', linestyle=':')
    ax_dt.add_feature(cfeature.RIVERS, edgecolor='#a0c4e8', linewidth=0.3)
    ax_dt.add_feature(cfeature.LAKES, facecolor='#d4eaf7', edgecolor='#888888', linewidth=0.3)
    
    gl2 = ax_dt.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5,
                           color='grey', linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.xlabel_style = {'size': 8}
    gl2.ylabel_style = {'size': 8}
    
    # Scatter sampling points
    for farm in farms:
        sub = df[df['farm'] == farm]
        ax_dt.scatter(sub['centroid_lon'], sub['centroid_lat'],
                      c=[farm_to_c[farm]], s=18, alpha=0.8, edgecolors='white',
                      linewidths=0.3, transform=proj, zorder=4)
    
    ax_dt.text(0.02, 0.02,
               f'N = {len(df)} samples | {len(farms)} farms | '
               f'{df["field_name"].nunique()} fields',
               transform=ax_dt.transAxes, fontsize=8,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
               zorder=10)
    
    ax_dt.set_title('(b) Sampling locations (coloured by farm)', fontsize=11, pad=8)
    
    # ── Save ──
    fig.savefig(os.path.join(FIG, 'fig1_study_area.pdf'))
    fig.savefig(os.path.join(FIG, 'fig1_study_area.png'))
    plt.close(fig)
    print('  ✓ Saved fig1_study_area.pdf/png')


# ══════════════════════════════════════════════════════════════════
# Figure 2: Scatter-plots predicted vs observed (pH, SOC, NO₃)
# ══════════════════════════════════════════════════════════════════
def fig2_scatter_plots():
    print('[Figure 2] Scatter-plots predicted vs observed ...')
    
    # Load best-model OOF: GBDT for pH, ET for SOC, RF for NO₃
    targets_models = [
        ('ph', 'baselines/GBDT', 'GBDT'),
        ('soc', 'baselines/ET', 'ET'),
        ('no3', 'rf', 'RF'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    for i, (tgt, folder, model_name) in enumerate(targets_models):
        path = os.path.join(RES, folder, f'{tgt}_oof_predictions.csv')
        df = pd.read_csv(path, low_memory=False)
        
        actual = df[tgt].values
        pred   = df['oof_pred'].values
        mask   = np.isfinite(actual) & np.isfinite(pred)
        actual, pred = actual[mask], pred[mask]
        
        rho, _ = stats.spearmanr(actual, pred)
        r2 = 1 - np.sum((actual - pred)**2) / np.sum((actual - actual.mean())**2)
        rmse = np.sqrt(np.mean((actual - pred)**2))
        
        ax = axes[i]
        ax.scatter(actual, pred, s=6, alpha=0.35, c='#2b83ba', edgecolors='none')
        
        mn, mx = min(actual.min(), pred.min()), max(actual.max(), pred.max())
        margin = (mx - mn) * 0.05
        ax.plot([mn - margin, mx + margin], [mn - margin, mx + margin],
                'k--', lw=1, alpha=0.6, label='1:1 line')
        ax.set_xlim(mn - margin, mx + margin)
        ax.set_ylim(mn - margin, mx + margin)
        
        ax.set_xlabel(f'Observed {TARGET_LABELS[tgt]}')
        ax.set_ylabel(f'Predicted {TARGET_LABELS[tgt]}')
        ax.set_title(f'{model_name} — {TARGET_LABELS[tgt]}')
        ax.set_aspect('equal')
        
        textstr = f'ρ = {rho:.3f}\nR² = {r2:.3f}\nRMSE = {rmse:.3f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.grid(True, alpha=0.2)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig2_scatter_pred_vs_obs.pdf'))
    fig.savefig(os.path.join(FIG, 'fig2_scatter_pred_vs_obs.png'))
    plt.close(fig)
    print('  ✓ Saved fig2_scatter_pred_vs_obs.pdf/png')


# ══════════════════════════════════════════════════════════════════
# Figure 3: Grouped bar chart — model comparison by Spearman ρ
# ══════════════════════════════════════════════════════════════════
def fig3_model_comparison():
    print('[Figure 3] Model comparison bar chart ...')
    
    comp = pd.read_csv(os.path.join(RES, 'all_models_comparison.csv'))
    
    # Map target names
    tgt_map = {
        'pH (KCl)': 'ph', 'SOC, %': 'soc', 'NO3, mg/kg': 'no3',
        'P2O5, mg/kg': 'p', 'K2O, mg/kg': 'k', 'S, mg/kg': 's',
    }
    comp['tgt'] = comp['Target'].map(tgt_map)
    
    # Select top models
    top_models = ['GBDT', 'RF', 'ET', 'XGBoost', 'CatBoost', 'ResNet']
    comp_top = comp[comp['Model'].isin(top_models)].copy()
    
    # Pivot
    piv = comp_top.pivot(index='tgt', columns='Model', values='Spearman_rho')
    piv = piv.reindex(TARGET_ORDER)
    piv = piv[top_models]  # order models
    
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(TARGET_ORDER))
    n_models = len(top_models)
    width = 0.13
    colours = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4', '#313695']
    top_models = [m for m in top_models if m in piv.columns]  # keep only existing
    
    for j, model in enumerate(top_models):
        vals = piv[model].values
        offset = (j - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=colours[j],
                       edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], fontsize=10)
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Model Comparison: Spearman ρ (Field-LOFO CV)')
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=3, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='grey', linestyle=':', alpha=0.4)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig3_model_comparison.pdf'))
    fig.savefig(os.path.join(FIG, 'fig3_model_comparison.png'))
    plt.close(fig)
    print('  ✓ Saved fig3_model_comparison.pdf/png')


# ══════════════════════════════════════════════════════════════════
# Figure 4: Feature importance (top-15 for pH, SOC, NO₃)
# ══════════════════════════════════════════════════════════════════
def fig4_feature_importance():
    print('[Figure 4] Feature importance ...')
    
    targets_info = [
        ('ph', 'rf', 'RF — pH'),
        ('soc', 'rf', 'RF — SOC'),
        ('no3', 'rf', 'RF — NO₃'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    for i, (tgt, folder, title) in enumerate(targets_info):
        fi = pd.read_csv(os.path.join(RES, folder, f'{tgt}_feature_importance.csv'))
        fi = fi.sort_values('importance', ascending=True).tail(15)
        
        ax = axes[i]
        colours = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi)))
        ax.barh(range(len(fi)), fi['importance'].values, color=colours, edgecolor='white', linewidth=0.3)
        ax.set_yticks(range(len(fi)))
        
        # Shorten feature names for readability
        short = [f.replace('spectral_', 'sp_').replace('topo_', 't_')
                  .replace('climate_', 'cl_').replace('glcm_', 'gl_')
                  .replace('_late_summer', '_lsum').replace('_spring', '_spr')
                  .replace('_autumn', '_aut').replace('_summer', '_sum')
                 for f in fi['feature'].values]
        ax.set_yticklabels(short, fontsize=7.5)
        ax.set_xlabel('MDI Importance')
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig4_feature_importance.pdf'))
    fig.savefig(os.path.join(FIG, 'fig4_feature_importance.png'))
    plt.close(fig)
    print('  ✓ Saved fig4_feature_importance.pdf/png')


# ══════════════════════════════════════════════════════════════════
# Figure 5: Heatmap — grid search for optimal farm split
# ══════════════════════════════════════════════════════════════════
def fig5_heatmap_split():
    print('[Figure 5] Heatmap grid search split ...')
    
    df = pd.read_csv(os.path.join(BASE, 'ML', 'farm_split_results.csv'))
    
    # Pivot: test_farms × val_farms → mean_rho
    piv = df.pivot_table(index='test_farms', columns='val_farms', values='mean_rho')
    piv = piv.sort_index(ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(piv.values, cmap='RdYlGn', aspect='auto',
                    vmin=piv.values.min() - 0.005, vmax=piv.values.max() + 0.005)
    
    # Labels
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns.astype(int))
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index.astype(int))
    ax.set_xlabel('Number of Validation Farms')
    ax.set_ylabel('Number of Test Farms')
    ax.set_title('Grid Search: Mean Spearman ρ\n(56 combinations × 15 seeds × 6 targets)')
    
    # Annotate cells
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            val = piv.values[i, j]
            if np.isfinite(val):
                # Mark the best cell
                text_color = 'white' if val < 0.57 else 'black'
                fontweight = 'bold' if val == piv.values[np.isfinite(piv.values)].max() else 'normal'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=7, color=text_color, fontweight=fontweight)
    
    cbar = fig.colorbar(im, ax=ax, label='Mean Spearman ρ', shrink=0.8)
    
    # Mark optimal
    best_idx = np.unravel_index(np.nanargmax(piv.values), piv.values.shape)
    ax.add_patch(plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                                fill=False, edgecolor='red', linewidth=2.5))
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig5_heatmap_split.pdf'))
    fig.savefig(os.path.join(FIG, 'fig5_heatmap_split.png'))
    plt.close(fig)
    print('  ✓ Saved fig5_heatmap_split.pdf/png')


# ══════════════════════════════════════════════════════════════════
# Figure 6: Heatmap — σ(ρ) for grid search (stability)
# ══════════════════════════════════════════════════════════════════
def fig6_heatmap_std():
    print('[Figure 6] Heatmap σ(ρ) ...')
    
    df = pd.read_csv(os.path.join(BASE, 'ML', 'farm_split_results.csv'))
    piv = df.pivot_table(index='test_farms', columns='val_farms', values='std_rho')
    piv = piv.sort_index(ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(piv.values, cmap='RdYlGn_r', aspect='auto',
                    vmin=piv.values[np.isfinite(piv.values)].min() - 0.005,
                    vmax=piv.values[np.isfinite(piv.values)].max() + 0.005)
    
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns.astype(int))
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index.astype(int))
    ax.set_xlabel('Number of Validation Farms')
    ax.set_ylabel('Number of Test Farms')
    ax.set_title('Grid Search: σ(ρ) — Estimate Stability\n(lower is better)')
    
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            val = piv.values[i, j]
            if np.isfinite(val):
                text_color = 'white' if val > 0.25 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=7, color=text_color)
    
    cbar = fig.colorbar(im, ax=ax, label='σ(ρ)', shrink=0.8)
    
    # Mark optimal (min std among top-5 rho)
    best_rho_idx = df.nlargest(1, 'mean_rho').index[0]
    best_row = df.loc[best_rho_idx]
    for i, idx_val in enumerate(piv.index):
        for j, col_val in enumerate(piv.columns):
            if idx_val == best_row['test_farms'] and col_val == best_row['val_farms']:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor='red', linewidth=2.5))
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig6_heatmap_std.pdf'))
    fig.savefig(os.path.join(FIG, 'fig6_heatmap_std.png'))
    plt.close(fig)
    print('  ✓ Saved fig6_heatmap_std.pdf/png')


# ══════════════════════════════════════════════════════════════════
# Figure 7: Ablation — patch size vs R² for ResNet
# ══════════════════════════════════════════════════════════════════
def fig7_ablation():
    print('[Figure 7] Ablation: patch size vs R² ...')
    
    ab = pd.read_csv(os.path.join(RES, 'ablation_summary_all_targets.csv'))
    
    # For each target, get the best R² per patch size (across feature configs)
    targets_plot = ['PH', 'HU', 'NO3', 'P', 'K', 'S']
    tgt_label = {'PH': 'pH', 'HU': 'SOC', 'NO3': 'NO₃', 'P': 'P₂O₅', 'K': 'K₂O', 'S': 'S'}
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = ['#d73027', '#fc8d59', '#91bfdb', '#4575b4', '#313695', '#999999']
    markers = ['o', 's', 'D', '^', 'v', 'x']
    
    for i, tgt in enumerate(targets_plot):
        sub = ab[ab['Target'] == tgt].copy()
        # Extract patch size as numeric
        sub['patch_num'] = sub['Patch_Size'].str.extract(r'(\d+)').astype(int)
        best_per_patch = sub.groupby('patch_num')['R2'].max().reset_index()
        best_per_patch = best_per_patch.sort_values('patch_num')
        
        ax.plot(best_per_patch['patch_num'], best_per_patch['R2'],
                marker=markers[i], color=colours[i], linewidth=1.5, markersize=7,
                label=tgt_label[tgt])
    
    ax.set_xlabel('Patch Size (pixels)')
    ax.set_ylabel('Best R²')
    ax.set_title('ResNet-18 Ablation: Patch Size Effect on R²')
    ax.set_xticks([16, 32, 64])
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='grey', linestyle=':', alpha=0.4)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig7_ablation_patch.pdf'))
    fig.savefig(os.path.join(FIG, 'fig7_ablation_patch.png'))
    plt.close(fig)
    print('  ✓ Saved fig7_ablation_patch.pdf/png')


# ══════════════════════════════════════════════════════════════════
# Figure 8: Leakage severity — ρ drop from field-LOFO to spatial
# ══════════════════════════════════════════════════════════════════
def fig8_leakage():
    print('[Figure 8] Leakage severity ...')
    
    # Data from the article tables (field-LOFO from all_models_comparison.csv,
    # spatial split 65/6/10 from sections/results.tex tab:tuned_ml)
    targets = ['pH', 'SOC', 'NO₃', 'P₂O₅', 'K₂O', 'S']
    field_lofo_rho = [0.857, 0.735, 0.775, 0.611, 0.624, 0.536]
    spatial_rho    = [0.761, 0.554, 0.575, 0.633, 0.539, 0.436]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(targets))
    width = 0.3
    
    bars1 = ax.bar(x - width/2, field_lofo_rho, width, label='Field-LOFO CV',
                    color='#4575b4', edgecolor='white')
    bars2 = ax.bar(x + width/2, spatial_rho, width, label='Spatial Split (65/6/10)',
                    color='#fc8d59', edgecolor='white')
    
    # Add delta annotations
    for i in range(len(targets)):
        delta = spatial_rho[i] - field_lofo_rho[i]
        mid_y = max(field_lofo_rho[i], spatial_rho[i]) + 0.02
        sign = '+' if delta >= 0 else ''
        ax.text(x[i], mid_y, f'Δ={sign}{delta:.3f}', ha='center', fontsize=7.5,
                color='#d73027' if abs(delta) > 0.15 else '#333333')
    
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=10)
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Spatial Leakage Assessment: ρ Drop from Field-LOFO to Spatial Split')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig8_leakage.pdf'))
    fig.savefig(os.path.join(FIG, 'fig8_leakage.png'))
    plt.close(fig)
    print('  ✓ Saved fig8_leakage.pdf/png')


# ══════════════════════════════════════════════════════════════════
# Figure 9: ConvNeXt ablation — single-season vs multi-season
# ══════════════════════════════════════════════════════════════════
def fig9_convnext_comparison():
    print('[Figure 9] ConvNeXt: single vs multi-season ...')
    
    single = pd.read_csv(os.path.join(RES, 'convnext_ablation_summary_all.csv'))
    multi  = pd.read_csv(os.path.join(RES, 'multiseason_ablation_summary_all.csv'))
    
    targets_plot = ['PH', 'HU', 'NO3', 'P', 'K', 'S']
    tgt_label = {'PH': 'pH', 'HU': 'SOC', 'NO3': 'NO₃', 'P': 'P₂O₅', 'K': 'K₂O', 'S': 'S'}
    
    # Best R² per target for each variant
    single_best = single.groupby('Target')['R2'].max()
    multi_best  = multi.groupby('Target')['R2'].max()
    
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(targets_plot))
    width = 0.35
    
    s_vals = [single_best.get(t, 0) for t in targets_plot]
    m_vals = [multi_best.get(t, 0) for t in targets_plot]
    
    ax.bar(x - width/2, s_vals, width, label='Single-season ConvNeXt',
           color='#91bfdb', edgecolor='white')
    ax.bar(x + width/2, m_vals, width, label='Multi-season ConvNeXt (54ch)',
           color='#d73027', edgecolor='white')
    
    # Improvement annotations
    for i in range(len(targets_plot)):
        if s_vals[i] > 0 and m_vals[i] > 0:
            pct = (m_vals[i] - s_vals[i]) / max(abs(s_vals[i]), 0.01) * 100
            top_y = max(s_vals[i], m_vals[i]) + 0.02
            if top_y > 0:
                sign = '+' if pct >= 0 else ''
                ax.text(x[i], top_y, f'{sign}{pct:.0f}%', ha='center', fontsize=7.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([tgt_label[t] for t in targets_plot])
    ax.set_ylabel('Best R²')
    ax.set_title('ConvNeXt: Single-season vs Multi-season (54-channel)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='grey', linestyle='-', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig9_convnext_seasons.pdf'))
    fig.savefig(os.path.join(FIG, 'fig9_convnext_seasons.png'))
    plt.close(fig)
    print('  ✓ Saved fig9_convnext_seasons.pdf/png')


# ══════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 60)
    print('Generating publication figures ...')
    print('=' * 60)
    
    fig1_study_area()
    fig2_scatter_plots()
    fig3_model_comparison()
    fig4_feature_importance()
    fig5_heatmap_split()
    fig6_heatmap_std()
    fig7_ablation()
    fig8_leakage()
    fig9_convnext_comparison()
    
    print('=' * 60)
    print(f'All figures saved to: {FIG}')
    print('=' * 60)
