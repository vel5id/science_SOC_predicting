"""
evaluate_models.py - Compare all trained ML models on Out-Of-Fold metrics.
Generates comprehensive comparison tables and charts for the article.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "ML/results")
OUT_DIR = RESULTS_DIR  # Save outputs here

TARGETS = ["ph", "soc", "no3", "p", "k", "s"]
TARGET_LABELS = {"ph": "pH (KCl)", "soc": "SOC, %", "no3": "NO3, mg/kg",
                 "p": "P2O5, mg/kg", "k": "K2O, mg/kg", "s": "S, mg/kg"}

def load_metrics(subfolder: str, filename: str) -> dict:
    path = os.path.join(RESULTS_DIR, subfolder, filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def run_comparison():
    # Load all available metrics
    rf_m   = load_metrics("rf", "rf_metrics_summary.json")
    cnn_m  = load_metrics("cnn", "cnn_metrics_summary.json")
    xgb_m  = load_metrics("xgb", "xgb_metrics_summary.json")
    cart_m = load_metrics("cart", "cart_metrics_summary.json")
    cat_m  = load_metrics("catboost", "catboost_metrics_summary.json")
    resnet_m = load_metrics("resnet", "resnet_metrics_summary.json") # New ResNet model
    base_m = load_metrics("baselines", "baselines_metrics_summary.json") # Dict of dicts!

    all_models = ["CatBoost", "RF", "XGBoost", "CART", "1D CNN", "ResNet", "ET", "KNN", "LR", "Ridge", "SGD", "SVR", "GBDT"]
    
    # Restructure data: dict of target -> model -> metrics
    comp_data = {t: {} for t in TARGETS}
    
    for t in TARGETS:
        comp_data[t]["CatBoost"] = cat_m.get(t, {})
        comp_data[t]["RF"] = rf_m.get(t, {})
        comp_data[t]["XGBoost"] = xgb_m.get(t, {})
        comp_data[t]["CART"] = cart_m.get(t, {})
        comp_data[t]["1D CNN"] = cnn_m.get(t, {})
        comp_data[t]["ResNet"] = resnet_m.get(t, {})
        
        # Load baselines
        for b_model in ["ET", "KNN", "LR", "Ridge", "SGD", "SVR", "GBDT"]:
            comp_data[t][b_model] = base_m.get(b_model, {}).get(t, {})

    # Create CSV table
    rows = []
    for t in TARGETS:
        for m_name in all_models:
            mets = comp_data[t].get(m_name, {})
            if mets and "rho" in mets:
                rows.append({
                    "Target": TARGET_LABELS.get(t, t),
                    "Model": m_name,
                    "Spearman_rho": mets.get("rho"),
                    "RMSE": mets.get("rmse"),
                    "R2": mets.get("r2")
                })
                
    df = pd.DataFrame(rows)
    if df.empty:
        print("No metrics found!")
        return
        
    df.to_csv(os.path.join(OUT_DIR, "all_models_comparison.csv"), index=False)
    
    # Pivot for easier terminal display
    pivot_df = df.pivot(index="Model", columns="Target", values="Spearman_rho").round(3)
    print("\n===  ALL MODELS COMPARISON (Spearman ρ, LOFO-CV) ===")
    print(pivot_df.to_string())

    # --- Plot: Grouped Bar chart Spearman rho comparison ---
    plt.figure(figsize=(16, 8))
    x = np.arange(len(TARGETS))
    
    # We have up to 11 models. To fit them we make bars thinner.
    valid_models = [m for m in all_models if m in pivot_df.index]
    num_models = len(valid_models)
    width = 0.8 / num_models
    
    # Nice categorical colors
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(num_models)]
    
    for i, m_name in enumerate(valid_models):
        rhos = [comp_data[t].get(m_name, {}).get("rho", 0) for t in TARGETS]
        rhos = [0 if pd.isna(r) else r for r in rhos] # replace nan with 0 for plotting
        
        offset = (i - num_models/2 + 0.5) * width
        plt.bar(x + offset, rhos, width, label=m_name, color=colors[i], edgecolor='black', alpha=0.9)

    plt.axhline(y=0.7, color='red', linestyle='--', linewidth=1, alpha=0.6, label='High Accuracy Threshold (ρ=0.7)')
    plt.xticks(x, [TARGET_LABELS.get(t, t) for t in TARGETS], fontsize=12)
    plt.ylabel("Spearman ρ (LOFO-CV)", fontsize=12)
    plt.title("Benchmarking 10+ ML Models for Digital Soil Mapping\nSpatial Leave-One-Field-Out Cross-Validation", fontsize=14)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.ylim(-0.2, 1.0) # Lower bound -0.2 to account for some bad models (e.g. LR acting crazy on collinear data)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "all_models_comparison_rho.png"), dpi=300)
    plt.close()
    
    print(f"\nAwesome mega-plot saved to: {os.path.join(OUT_DIR, 'all_models_comparison_rho.png')}")

if __name__ == "__main__":
    run_comparison()
    print("\n[DONE] Mega-evaluation complete.")
