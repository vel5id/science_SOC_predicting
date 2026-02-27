import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_comparison_plot():
    os.makedirs('ML/results', exist_ok=True)
    
    # 1. Load ML Baselines
    try:
        df_ml = pd.read_csv('ML/results/classic_ml_ablation_split.csv')
    except Exception as e:
        print("Error loading ML baselines:", e)
        return
        
    # Pick Top 2 ML models per target by R2
    # First, make sure Target is uppercase for matching
    df_ml['Target'] = df_ml['Target'].str.upper()
    df_ml_sorted = df_ml.sort_values(by=['Target', 'R2'], ascending=[True, False])
    best_ml = df_ml_sorted.groupby('Target').head(2).copy()
    best_ml['Model_Name'] = best_ml['Model']
    
    # 2. Load CNN models
    try:
        df_cnn = pd.read_csv('ML/results/ablation_summary_all_targets.csv')
    except Exception as e:
        print("Error loading CNN baselines:", e)
        return
        
    df_cnn['Target'] = df_cnn['Target'].str.upper()
    # Pick best CNN per target
    idx = df_cnn.groupby('Target')['R2'].idxmax()
    best_cnn = df_cnn.loc[idx].copy()
    # Name it something nice
    best_cnn['Model_Name'] = 'ResNet-18 (2D)'
    
    # Combine the top ML models and the best CNN model
    combined = pd.concat([
        best_ml[['Target', 'Model_Name', 'R2']],
        best_cnn[['Target', 'Model_Name', 'R2']]
    ], ignore_index=True)
    
    # Sort targets biologically/chemically or alphabetically
    combined['Target'] = pd.Categorical(combined['Target'], categories=['PH', 'HU', 'NO3', 'P', 'K', 'S'], ordered=True)
    
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid", context="talk")
    
    # Create the bar plot
    ax = sns.barplot(
        data=combined, 
        x='Target', 
        y='R2', 
        hue='Model_Name',
        palette='tab20'
    )
    
    # Add titles and labels
    plt.title('Unified Model Comparison (R² Score) - Same Spatial Split', fontsize=18, pad=20, fontweight='bold')
    plt.xlabel('Soil Property (Target)', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score (Test Set)', fontsize=14, fontweight='bold')
    
    # Add horizontal line at 0
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
    
    # Move legend outside
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height != 0:
            ax.annotate(f'{height:.2f}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom' if height > 0 else 'top', 
                        xytext=(0, 5 if height > 0 else -15), 
                        textcoords='offset points',
                        fontsize=10, rotation=90 if height < 0 else 0)
                        
    plt.ylim(-0.3, 1.0)
    plt.tight_layout()
    
    out_path = 'ML/results/unified_comparison_r2.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {out_path}")

if __name__ == '__main__':
    generate_comparison_plot()
