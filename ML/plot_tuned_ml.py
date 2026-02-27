import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_tuned_models():
    os.makedirs('ML/results', exist_ok=True)
    df = pd.read_csv('ML/results/tuned_ml_ablation_split.csv')
    df['Target'] = df['Target'].str.upper()
    df['Target'] = pd.Categorical(df['Target'], categories=['PH', 'HU', 'NO3', 'P', 'K', 'S'], ordered=True)

    plt.figure(figsize=(18, 8))
    sns.set_theme(style="whitegrid", context="talk")

    ax = sns.barplot(
        data=df, 
        x='Target', 
        y='R2', 
        hue='Model',
        palette='tab10'
    )

    plt.title('Performance of All Tuned Classical Models (R² Score)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Soil Property (Target)', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score (Test Split)', fontsize=14, fontweight='bold')
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
    plt.legend(title='Model Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    # Adding values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height) and height != 0:
            ax.annotate(f'{height:.2f}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom' if height > 0 else 'top', 
                        xytext=(0, 5 if height > 0 else -15), 
                        textcoords='offset points',
                        fontsize=9, rotation=90 if height < 0 else 0)

    # Set dynamic y-limits avoiding cutting off annotations
    bottom_lim = max(-1.0, df['R2'].min() - 0.2) # Cap the bottom at -1.0 so horrible models don't ruin the scale completely
    plt.ylim(bottom_lim, min(1.0, df['R2'].max() + 0.2))
    
    plt.tight_layout()
    out_path = 'ML/results/tuned_ml_all_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == '__main__':
    plot_tuned_models()
