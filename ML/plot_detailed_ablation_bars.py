import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_detailed_bars():
    os.makedirs('ML/results', exist_ok=True)
    try:
        df_bases = pd.read_csv('ML/results/ablation_summary_all_targets.csv')
        df_8x8 = pd.read_csv('ML/results/ablation_summary_8x8.csv')
        df = pd.concat([df_bases, df_8x8], ignore_index=True)
    except Exception as e:
        print("Error loading CSV:", e)
        return
        
    df['Target'] = df['Target'].str.upper()

    targets = ['PH', 'HU', 'NO3', 'P', 'K', 'S']
    sizes = ['8x8', '16x16', '32x32', '64x64']
    configs = []
    for s in sizes:
        for i in range(6):
            configs.append(f"{s} +{i} idx")

    df['Config'] = df['Patch_Size'] + " +" + df['Num_Indices'].astype(str) + " idx"
    df['Config'] = pd.Categorical(df['Config'], categories=configs, ordered=True)

    fig, axes = plt.subplots(3, 2, figsize=(22, 20))
    axes = axes.flatten()

    for i, t in enumerate(targets):
        ax = axes[i]
        target_data = df[df['Target'] == t].sort_values('Config')
        
        if target_data.empty:
            continue
            
        sns.barplot(
            data=target_data,
            x='Config',
            y='R2',
            hue='Patch_Size',
            palette='Set2',
            dodge=False,
            ax=ax
        )
        
        ax.set_title(f"Target: {t}", fontsize=18, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
        ax.set_ylabel("R² Score" if i % 2 == 0 else "", fontsize=14)
        ax.set_xlabel("")
        ax.axhline(0, color='black', linewidth=1, linestyle='--')
        
        # Adjust Y limits cleanly
        ax.set_ylim(min(-0.3, target_data['R2'].min() - 0.1), min(1.0, target_data['R2'].max() + 0.15))
        
        # Annotate bars
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height != 0:
                ax.annotate(f"{height:.2f}",
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom' if height > 0 else 'top',
                            xytext=(0, 4 if height > 0 else -15),
                            textcoords='offset points',
                            fontsize=10, rotation=0, fontweight='bold')

        # Only keep legend for the first plot
        if i == 0:
            ax.legend(title="Patch Size", loc='upper right', fontsize=12)
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    plt.suptitle('Detailed ResNet-18 Configurations Performance (R² Score by Model)', fontsize=26, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = 'ML/results/detailed_ablation_bars.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Detailed bars plot saved to {out_path}")

if __name__ == '__main__':
    generate_detailed_bars()
