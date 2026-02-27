import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_multiseason_ablation_matrix():
    os.makedirs('ML/results', exist_ok=True)
    csv_path = 'ML/results/multiseason_ablation_summary_all.csv'
    
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found yet.")
        return
        
    df = pd.read_csv(csv_path)
    
    # We only have 32x32 patch size in multiseason
    df['Target'] = df['Target'].str.upper()
    df['Target'] = pd.Categorical(df['Target'], categories=['PH', 'HU', 'NO3', 'P', 'K', 'S'], ordered=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()
    
    targets = ['PH', 'HU', 'NO3', 'P', 'K', 'S']
    
    # Map Num_Indices to readable names
    # 0 = Base (39 Channels: 3 seasons optics + radar + dem)
    # 1 = + NDVI (3 seasons)
    # 2 = + BSI
    # 3 = + NDSI
    # 4 = + NDWI
    # 5 = + RECI
    x_labels = ['Base (39Ch)', '+NDVI', '+BSI', '+NDSI', '+NDWI', '+RECI']
    
    for i, target in enumerate(targets):
        ax = axes[i]
        subset = df[df['Target'] == target].sort_values('Num_Indices')
        
        if len(subset) == 0:
            continue
            
        sns.lineplot(
            data=subset, 
            x='Num_Indices', 
            y='R2', 
            marker='o', 
            markersize=10, 
            linewidth=2.5,
            color='#2ca02c',
            ax=ax
        )
        
        ax.set_title(f'Target: {target}', fontsize=16, fontweight='bold')
        ax.set_ylabel('R² Score' if i % 3 == 0 else '')
        ax.set_xlabel('Added Spectral Indices (Across 3 Seasons)' if i >= 3 else '')
        
        ax.set_xticks(range(6))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Annotate points
        for _, row in subset.iterrows():
            ax.annotate(f"{row['R2']:.3f}", 
                        (row['Num_Indices'], row['R2']),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', fontsize=10)
                        
        # Best model line
        best_r2 = subset['R2'].max()
        ax.axhline(best_r2, color='red', linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    out_path = 'ML/results/multiseason_ablation_matrix.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Matrix plot saved to {out_path}")

if __name__ == '__main__':
    plot_multiseason_ablation_matrix()
