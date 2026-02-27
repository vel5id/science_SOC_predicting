import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_final_comparison():
    os.makedirs('ML/results', exist_ok=True)
    
    # Load TUNED ML Baselines
    try:
        df_ml = pd.read_csv('ML/results/tuned_ml_ablation_split.csv')
    except Exception as e:
        print("Error loading tuned ML baselines:", e)
        return
        
    df_ml['Target'] = df_ml['Target'].str.upper()
    df_ml_sorted = df_ml.sort_values(by=['Target', 'R2'], ascending=[True, False])
    best_ml = df_ml_sorted.groupby('Target').head(2).copy()
    best_ml['Model_Name'] = best_ml['Model'] + ' (Tuned)'
    
    # Load CNN models (16x16, 32x32, 64x64) AND (8x8)
    try:
        df_cnn_bases = pd.read_csv('ML/results/ablation_summary_all_targets.csv')
        df_cnn_8x8 = pd.read_csv('ML/results/ablation_summary_8x8.csv')
        df_cnn = pd.concat([df_cnn_bases, df_cnn_8x8], ignore_index=True)
    except Exception as e:
        print("Error loading CNN baselines:", e)
        return
        
    df_cnn['Target'] = df_cnn['Target'].str.upper()
    # Pick optimal CNN per target
    idx = df_cnn.groupby('Target')['R2'].idxmax()
    best_cnn = df_cnn.loc[idx].copy()
    best_cnn['Model_Name'] = 'Best ResNet-18 (2D)'
    
    combined = pd.concat([
        best_ml[['Target', 'Model_Name', 'R2']],
        best_cnn[['Target', 'Model_Name', 'R2']]
    ], ignore_index=True)
    
    combined['Target'] = pd.Categorical(combined['Target'], categories=['PH', 'HU', 'NO3', 'P', 'K', 'S'], ordered=True)
    
    plt.figure(figsize=(15, 8))
    sns.set_theme(style="whitegrid", context="talk")
    
    # Build
    ax = sns.barplot(
        data=combined, 
        x='Target', 
        y='R2', 
        hue='Model_Name',
        palette='Dark2'
    )
    
    plt.title('Final Model Comparison after Hyperparameter Tuning (R² Score)', fontsize=18, pad=20, fontweight='bold')
    plt.xlabel('Soil Property (Target)', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score (Test Set)', fontsize=14, fontweight='bold')
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    # Labels
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height != 0:
            ax.annotate(f'{height:.2f}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom' if height > 0 else 'top', 
                        xytext=(0, 5 if height > 0 else -15), 
                        textcoords='offset points',
                        fontsize=10, rotation=90 if height < 0 else 0)
                        
    plt.ylim(min(-0.3, combined['R2'].min() - 0.1), min(1.0, combined['R2'].max() + 0.15))
    plt.tight_layout()
    
    out_path = 'ML/results/final_comparison_r2_tuned.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {out_path}")

if __name__ == '__main__':
    generate_final_comparison()
