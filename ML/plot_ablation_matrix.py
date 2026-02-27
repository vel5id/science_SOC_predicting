import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_ablation_matrix_plot():
    os.makedirs('ML/results', exist_ok=True)
    
    # Load CNN models
    try:
        df_bases = pd.read_csv('ML/results/ablation_summary_all_targets.csv')
        df_8x8 = pd.read_csv('ML/results/ablation_summary_8x8.csv')
        df = pd.concat([df_bases, df_8x8], ignore_index=True)
    except Exception as e:
        print("Error loading CNN baselines:", e)
        return
        
    df['Target'] = df['Target'].str.upper()
    df['Target'] = pd.Categorical(df['Target'], categories=['PH', 'HU', 'NO3', 'P', 'K', 'S'], ordered=True)
    
    # Set the style
    sns.set_theme(style="whitegrid", context="notebook")
    
    # Create a FacetGrid (line plot or point plot)
    g = sns.FacetGrid(
        df, 
        col="Target", 
        col_wrap=3, 
        height=4.5, 
        aspect=1.2,
        sharey=False # Because different targets have drastically different R2 ranges
    )
    
    # Map a pointplot/lineplot to show the trend
    g.map_dataframe(
        sns.pointplot, 
        x="Num_Indices", 
        y="R2", 
        hue="Patch_Size",
        palette="Set1",
        dodge=True,
        markers=["d", "o", "s", "^"],
        linestyles=[":", "-", "--", "-."]
    )
    
    # Adjust titles and labels
    g.set_axis_labels("Number of Spectral Indices Added", "R² Score (Test Set)")
    g.set_titles(col_template="Target: {col_name}", size=14, weight='bold')
    
    # Add a horizontal dashed line at R2=0 for reference
    for ax in g.axes.flatten():
        ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    
    # Add a global legend
    g.add_legend(title="Patch Size", title_fontsize=12, fontsize=11, bbox_to_anchor=(1.02, 0.5), loc='center left')
    
    # Global title
    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle('ResNet-18 Ablation Study: Impact of Spatial Context and Spectral Indices', 
                      fontsize=18, fontweight='bold')
    
    out_path = 'ML/results/cnn_ablation_matrix.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Ablation Matrix plot saved successfully to {out_path}")

if __name__ == '__main__':
    generate_ablation_matrix_plot()
