"""
train_resnet.py - Train 2D CNN (ResNet) on 13-channel GEE Satellite Patches
Predicts soil properties using Spatial LOFO Cross-Validation.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.models as models
import torchvision.transforms.functional as TF

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ML.data_loader import SpatialDataLoader

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "ML/results/resnet_18ch")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ResNet-18 Architecture Adaptation for 13 Channels ---
class SentinelResNet(nn.Module):
    def __init__(self, in_channels=18, out_features=1, dropout_rate=0.3):
        super().__init__()
        # Load a pretrained ResNet18 (pretrained on RGB 3-channels)
        # We use weights=None to train from scratch since our data is very different
        self.resnet = models.resnet18(weights=None)
        
        # Replace the very first convolutional layer to accept 13 channels instead of 3
        # Original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, 
                                      kernel_size=original_conv1.kernel_size, 
                                      stride=original_conv1.stride, 
                                      padding=original_conv1.padding, 
                                      bias=original_conv1.bias)
        
        # Replace the final fully connected layer for regression
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        return self.resnet(x)

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 60 # Increased epochs because augmentation makes the task harder but generalizes better
PATIENCE = 10 # Increased patience

# --- Data Augmentation Dataset ---
class AugmentedPatchDataset(Dataset):
    def __init__(self, X_tensor, y_tensor, augment=False):
        self.X = X_tensor
        self.y = y_tensor
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            # 1. Random horizontal flip (50% chance)
            if torch.rand(1) < 0.5:
                x = TF.hflip(x)
            
            # 2. Random vertical flip (50% chance)
            if torch.rand(1) < 0.5:
                x = TF.vflip(x)
                
            # 3. Random rotation by 0, 90, 180, or 270 degrees
            rotations = [0, 90, 180, 270]
            rot = np.random.choice(rotations)
            if rot > 0:
                # Need to convert degrees to standard number of 90-deg rotations for generic tensors
                k = rot // 90
                x = torch.rot90(x, k, dims=[1, 2]) # Rotate along H, W dimensions

        return x, y

def train_target(target_col: str, loader: SpatialDataLoader):
    print(f"\n{'='*50}")
    print(f" Training ResNet-18 (2D CNN) for: {target_col.upper()}")
    print(f"{'='*50}")
    
    # Check if patches exist
    patches_dir = os.path.join(_PROJECT_ROOT, "data/patches_18ch")
    if not os.path.exists(patches_dir) or len(os.listdir(patches_dir)) == 0:
        print(f"ERROR: No patches found in {patches_dir}. Please run extract_gee_patches.py first.")
        return None
    
    # Note: data_loader.py __getitem__ now correctly returns (patch, y) when initialized with load_patches=True
    # However, to use PyTorch DataLoader efficiently, we load everything into memory if possible,
    # or use a custom Dataset. Since we only have ~1000 patches, loading to memory is fast (~2GB).
    
    # Custom fast in-memory dataset loader to get ALL valid data for the target
    valid_indices = []
    valid_y = []
    valid_patches = []
    
    print("Loading patches into RAM for faster training...")
    for idx in range(len(loader.df)):
        y_val = loader.y[idx]
        if pd.isna(y_val):
            continue
            
        orig_idx = loader.df.index[idx]
        patch_path = os.path.join(patches_dir, f"patch_idx_{orig_idx}.npy")
        
        if os.path.exists(patch_path):
            try:
                patch = np.load(patch_path) # [13, 64, 64]
                # Optional: Basic normalization per channel (Global Standard Scaling is better, but min-max per patch is okay)
                # For Earth Engine data (Sentinel-2 reflectance), values are usually 0-1, DEM is ~0-3000m
                # Let's standardize DEM (channel 12, index 12 in 0-indexed) globally later or locally
                patch_tensor = torch.tensor(patch, dtype=torch.float32)
                
                valid_patches.append(patch_tensor)
                valid_y.append(y_val)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error loading {patch_path}: {e}")
                
    if not valid_patches:
        print(f"No valid patches found for {target_col}.")
        return None
        
    X_tensor = torch.stack(valid_patches) # [N, 13, 64, 64]
    y_tensor = torch.tensor(valid_y, dtype=torch.float32).unsqueeze(1) # [N, 1]
    fields = loader.fields[valid_indices]
    
    print(f"Loaded {len(valid_patches)} patches. Tensor shape: {X_tensor.shape}")
    # NOTE: Normalization is intentionally deferred to inside the fold loop so that
    # mean/std are computed STRICTLY on the per-fold training subset, preventing
    # data leakage from val/test folds into the normalisation statistics.

    unique_fields = np.unique(fields)
    
    oof_preds = np.zeros(len(y_tensor))
    fold_results = []
    
    # LOFO-CV
    pbar_folds = tqdm(unique_fields, desc=f"Folds [{target_col.upper()}]", leave=True)
    
    for test_field in pbar_folds:
        # TEST set (Out-of-Fold prediction)
        test_mask = (fields == test_field)
        
        # We need a strictly independent VALIDATION set for Early Stopping.
        # Pick one random field from the remaining fields for validation.
        remaining_fields = unique_fields[unique_fields != test_field]
        val_field = np.random.choice(remaining_fields)
        val_mask = (fields == val_field)
        
        # TRAIN set is everything else
        train_mask = ~(test_mask | val_mask)
        
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        # Z-Score Normalization per channel — statistics computed STRICTLY on this
        # fold's train subset to prevent data leakage from val/test into the statistics.
        fold_mean = X_tensor[train_idx].mean(dim=(0, 2, 3), keepdim=True)
        fold_std  = X_tensor[train_idx].std(dim=(0, 2, 3), keepdim=True)
        fold_std[fold_std == 0] = 1.0
        X_norm = (X_tensor - fold_mean) / fold_std

        X_train, y_train = X_norm[train_idx], y_tensor[train_idx]
        X_val, y_val_split = X_norm[val_idx], y_tensor[val_idx]
        X_test, y_test = X_norm[test_idx], y_tensor[test_idx]
        
        # PyTorch Datasets
        # Only apply augmentation to the TRAIN set to prevent spatial data leakage
        train_dataset = AugmentedPatchDataset(X_train, y_train, augment=True)
        val_dataset = AugmentedPatchDataset(X_val, y_val_split, augment=False)
        test_dataset = AugmentedPatchDataset(X_test, y_test, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Init model
        model = SentinelResNet(in_channels=X_tensor.shape[1], out_features=1).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                
                optimizer.zero_grad()
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    preds = model(batch_X)
                    loss = criterion(preds, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                break
                
        # Load best model for OOF prediction
        model.load_state_dict(best_model_state)
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(DEVICE)
                preds = model(batch_X).cpu().numpy()
                fold_preds.extend(preds.flatten())
                
        oof_preds[test_idx] = fold_preds
        pbar_folds.set_postfix({'val_loss': f"{best_val_loss:.3f}", 'ep': best_epoch})
        
    # Calculate global metrics
    y_true = np.array(valid_y)
    rmse = np.sqrt(mean_squared_error(y_true, oof_preds))
    mae = mean_absolute_error(y_true, oof_preds)
    r2 = r2_score(y_true, oof_preds)
    rho, p_val = spearmanr(y_true, oof_preds)
    
    metrics = {"rho": rho, "rmse": rmse, "mae": mae, "r2": r2}
    
    print(f"\n[Results {target_col.upper()}]")
    print(f"  Spearman rho : {rho:.3f} (p={p_val:.2e})")
    print(f"  RMSE         : {rmse:.3f}")
    print(f"  MAE          : {mae:.3f}")
    print(f"  R²           : {r2:.3f}")
    
    # Plot true vs predicted
    plt.figure(figsize=(8,8))
    plt.scatter(y_true, oof_preds, alpha=0.4, edgecolors='k', c='purple')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f"ResNet-18 OOF Predictions: {target_col.upper()}\nSpearman ρ: {rho:.3f}")
    plt.xlabel("True Values")
    plt.ylabel("Predicted")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, f"{target_col}_resnet_scatter.png"), dpi=200)
    plt.close()
    
    return metrics, oof_preds, valid_y

def main():
    print("="*50)
    print(f"  ResNet-18 device : {DEVICE.type.upper()} {' ✅' if DEVICE.type == 'cuda' else ' ❗️'}")
    print("="*50)
    
    targets = ["ph", "soc", "no3", "p", "k", "s"]
    all_metrics = {}
    
    for t in targets:
        loader = SpatialDataLoader(target=t)
        res = train_target(t, loader)
        if res is not None:
            metrics, oofs, trues = res
            all_metrics[t] = metrics
            
            # Save OOFs
            df_oof = pd.DataFrame({"y_true": trues, "y_pred": oofs})
            df_oof.to_csv(os.path.join(RESULTS_DIR, f"{t}_resnet_oof.csv"), index=False)
            
    # Save metrics summary
    if all_metrics:
        with open(os.path.join(RESULTS_DIR, "resnet_18ch_metrics_summary.json"), "w") as f:
            json.dump(all_metrics, f, indent=4)
            
        print("\n" + "="*50)
        print("FINAL ResNet-18 SUMMARY (Spearman ρ, LOFO-CV)")
        print("="*50)
        for t, m in all_metrics.items():
            print(f"  {t.upper():<12} : ρ = {m['rho']:.3f}  RMSE = {m['rmse']:.3f}  R² = {m['r2']:.3f}")
        print(f"\n[DONE] Results saved in {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
