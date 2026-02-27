import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_multiseason_comparison():
    os.makedirs('ML/results', exist_ok=True)
    
    # Check if files exist
    cnn_file = 'ML/results/multiseason_ablation_summary_all.csv'
    ml_file = 'ML/results/tuned_ml_ablation_split.csv'
    
    if not os.path.exists(cnn_file) or not os.path.exists(ml_file):
        print("Required CSV files not found yet. Skipping plot.")
        return
        
    cnn_df = pd.read_csv(cnn_file)
    ml_df = pd.read_csv(ml_file)
    
    # 1. Process Multiseason CNN Data
    # Get the best CNN configuration per target
    best_cnn = cnn_df.loc[cnn_df.groupby('Target')['R2'].idxmax()].copy()
    best_cnn['Model'] = "Multiseason CNN (54Ch)"
    
    # 2. Process ML Data
    # Get the best ML model per target (e.g. XGBoost or RF)
    best_ml = ml_df.loc[ml_df.groupby('Target')['R2'].idxmax()].copy()
    
    # Append the actual model name to easily read it on the plot
    best_ml['Model'] = "Best 1D ML (" + best_ml['Model'] + ")"
    
    # Combine the data
    compare_df = pd.concat([
        best_cnn[['Target', 'Model', 'R2']],
        best_ml[['Target', 'Model', 'R2']]
    ])
    
    compare_df['Target'] = compare_df['Target'].str.upper()
    compare_df['Target'] = pd.Categorical(compare_df['Target'], categories=['PH', 'HU', 'NO3', 'P', 'K', 'S'], ordered=True)
    
    # Add an Architecture column for hue
    compare_df['Architecture'] = np.where(compare_df['Model'].str.contains('CNN'), 'Multiseason 3D-CNN', 'Best Tuned 1D ML')
    
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid", context="talk")
    
    ax = sns.barplot(
        data=compare_df, 
        x='Target', 
        y='R2', 
        hue='Architecture',
        palette=['#2ca02c', '#d62728'] # Green for CNN, Red for ML
    )
    
    plt.title('Performance Comparison: Multiseason 3D-CNN vs Best 1D Classical ML', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Soil Property (Target)', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score (Independent Test Split)', fontsize=14, fontweight='bold')
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
    plt.legend(title='Architecture', loc='upper right', fontsize=12)
    
    # Adding values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height) and height != 0:
            ax.annotate(f'{height:.2f}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom' if height > 0 else 'top', 
                        xytext=(0, 5 if height > 0 else -15), 
                        textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        color='black')
                        
    # Set y limit
    plt.ylim(max(-0.5, compare_df['R2'].min() - 0.2), min(1.0, compare_df['R2'].max() + 0.2))
    
    plt.tight_layout()
    out_path = 'ML/results/multiseason_vs_ml_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {out_path}")

if __name__ == "__main__":
    plot_multiseason_comparison()
