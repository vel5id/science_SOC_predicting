"""
Final 64x64 Evaluation Script for Sulfur CNN
Using parameters from Trial 7 of the interrupted Optuna run.
"""
import torch
from train_optuna_sulfur_cnn_54 import prepare_data, train_one_trial, set_seed

if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare_data()
    
    # Best Params from Trial 7 (64x64)
    best_params = {
        'cnn_dim': 256, 
        'emb_dim': 128, 
        'n_blocks': 2, 
        'lr': 0.000230125, 
        'dropout': 0.3327
    }
    
    print("\n▶ Running final training on 64x64 patches (300 epochs)...")
    v, rho, r2 = train_one_trial(best_params, data, device, n_epochs=300, patience=25, return_preds=True)
    print(f"\nFINAL TEST (64x64): Spearman Rho = {rho:.4f}, R2 = {r2:.4f}")
