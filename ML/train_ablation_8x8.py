import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import csv
import random
import os
from tqdm import tqdm
import time

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = True

# Safety limits for CPU to prevent overheating
max_cpus = os.cpu_count() or 4
target_threads = max(1, int(max_cpus * 0.8))
torch.set_num_threads(target_threads)
print(f"Limiting PyTorch to {target_threads}/{max_cpus} CPU threads to prevent overheating...")

# ==========================================
# 1. DATASET WITH DYNAMIC FEATURE STACKING
# ==========================================
class AblationPatchDataset(Dataset):
    def __init__(self, patches: torch.Tensor, targets: torch.Tensor, 
                 base_bands_idx: list, active_indices_idx: list, augment: bool = False):
        """
        patches: Tensor of shape [N, C_total, H, W] cached in RAM
        targets: Tensor of shape [N, 1]
        base_bands_idx: List of channel indices for raw bands.
        active_indices_idx: List of channel indices for engineered indices added iteratively.
        """
        self.patches = patches
        self.targets = targets
        # Dynamically select active features
        self.active_channels = base_bands_idx + active_indices_idx
        self.augment = augment

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.patches[idx, self.active_channels, :, :]
        y = self.targets[idx]
        
        # Fast spatial augmentations
        if self.augment:
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[2]) # hflip
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[1]) # vflip
            rot = torch.randint(0, 4, (1,)).item()
            if rot > 0:
                x = torch.rot90(x, k=rot, dims=[1, 2])
                
        return x, y

# ==========================================
# 2. LIGHTWEIGHT CNN/RESNET
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class LightweightResNet(nn.Module):
    def __init__(self, in_channels, out_features=1):
        """Lightweight architecture designed for 16x16 up to 64x64 patches"""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = self._make_layer(32, 32)
        self.layer2 = self._make_layer(32, 64, stride=2)
        self.layer3 = self._make_layer(64, 128, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )

    def _make_layer(self, in_c, out_c, stride=1):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        return ResidualBlock(in_c, out_c, stride, downsample)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ==========================================
# 3. CUSTOM METRICS
# ==========================================
def log_cosh_loss(y_pred, y_true):
    def _log_cosh(x):
        return x + F.softplus(-2.0 * x) - np.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

# ==========================================
# 4. ABLATION LOOP
# ==========================================
def run_ablation_study(df, patch_cache, base_bands_idx, engineered_indices_idx, 
                       target_col='target', farm_col='field_name', device='cuda'):
    """
    df: Pandas DataFrame containing target and farm_id
    patch_cache: Dictionary mapping patch sizes -> Ram-cached torch.Tensor[N, C_total, W, H]
    """
    
    MODELS_DIR = "ML/ml_models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    patch_sizes = [8] # Only 8x8 instead of 16, 32, 64
    num_idx_configs = range(6) # 0 to 5
    
    # Fixed Spatial Split by Farm IDs
    unique_farms = np.array(df[farm_col].unique().tolist())
    np.random.seed(42) # fixed single split
    np.random.shuffle(unique_farms)
    
    # User requirement: Rest Train, Val 10, Test 12 (since total is 81)
    val_count = 10
    test_count = 12
    train_farms = unique_farms[:-(val_count + test_count)]
    val_farms = unique_farms[-(val_count + test_count):-test_count]
    test_farms = unique_farms[-test_count:]
    
    # Get dataframe indices
    train_idx = df[df[farm_col].isin(train_farms)].index.values
    val_idx = df[df[farm_col].isin(val_farms)].index.values
    test_idx = df[df[farm_col].isin(test_farms)].index.values
    
    results = []
    
    for patch_size in patch_sizes:
        print(f"\n{'='*50}\nStarting Ablation for Patch Size: {patch_size}x{patch_size}\n{'='*50}")
        print(f"Split sizes -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Load pre-stacked dataset into RAM entirely (as float32)
        X_full = patch_cache[patch_size].to(torch.float32)

        # Z-Score Normalization per channel — statistics computed STRICTLY on train_idx only
        # to prevent data leakage from val/test into training statistics.
        train_patches = X_full[train_idx]
        dataset_mean = train_patches.mean(dim=(0, 2, 3), keepdim=True)
        dataset_std = train_patches.std(dim=(0, 2, 3), keepdim=True)
        dataset_std[dataset_std == 0] = 1.0
        X_full = (X_full - dataset_mean) / dataset_std
        
        y_full = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)
        groups = df[farm_col].values
        
        # Iterate over feature steps
        for step in range(len(engineered_indices_idx) + 1):
            set_seed(42 + step) # Fix seed per configuration
            active_idx_channels = engineered_indices_idx[:step]
            config_name = f"Bands_{len(base_bands_idx)}+{step}_Indices"
            print(f"  Configuration: {patch_size}x{patch_size} | {config_name}")
            
            # Dynamic feature datasets
            train_dataset = AblationPatchDataset(X_full[train_idx], y_full[train_idx], 
                                                 base_bands_idx, active_idx_channels, augment=True)
            val_dataset   = AblationPatchDataset(X_full[val_idx], y_full[val_idx], 
                                                 base_bands_idx, active_idx_channels, augment=False)
            test_dataset  = AblationPatchDataset(X_full[test_idx], y_full[test_idx], 
                                                 base_bands_idx, active_idx_channels, augment=False)
            
            # Single-threaded DataLoader (num_workers=0) to completely prevent CPU overlapping/freezing
            # pin_memory=False prevents memory exhaustion (OOM Killer)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
            val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
            test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
            
            in_c = len(base_bands_idx) + step
            model = LightweightResNet(in_channels=in_c).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
            criterion = nn.HuberLoss() # Dataset has extreme outliers (e.g. fertilizer hotspots)
            scaler = GradScaler()      # Mixed Precision for fast RTX 5060 Ti training
            
            best_val_loss = float('inf')
            best_weights = None
            patience = 20
            patience_counter = 0
            
            start_time = time.time()
            for epoch in tqdm(range(300), desc=f"  Epochs (Step {step})", leave=False):
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)
                    # Automatic Mixed Precision
                    with autocast():
                        preds = model(batch_X)
                    
                    # Calculate loss in float32 outside autocast for stability
                    loss = criterion(preds.float(), batch_y.float())
                        
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # Validation loop for Early Stopping
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                        preds = model(batch_X)
                        val_loss += criterion(preds, batch_y).item() * batch_X.size(0)
                        
                val_loss /= len(val_dataset)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        tqdm.write(f"    Early stopping at epoch {epoch+1} (Val Loss: {val_loss:.4f})")
                        break # Aggressive early stopping
                        
            # Final Testing on unseen TEST fold
            model.load_state_dict(best_weights)
            model.eval()
            test_preds = []
            test_truths = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    preds = model(batch_X.to(device, non_blocking=True))
                    test_preds.extend(preds.cpu().numpy())
                    test_truths.extend(batch_y.cpu().numpy())
                    
            test_preds = np.array(test_preds).flatten()
            test_truths = np.array(test_truths).flatten()
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Log Metrics Globally (on TEST set)
            rmse = np.sqrt(mean_squared_error(test_truths, test_preds))
            mae = mean_absolute_error(test_truths, test_preds)
            r2 = r2_score(test_truths, test_preds)
            
            y_true_t = torch.tensor(test_truths)
            y_pred_t = torch.tensor(test_preds)
            huber = F.huber_loss(y_pred_t, y_true_t).item()
            log_cosh = log_cosh_loss(y_pred_t, y_true_t).item()
            
            # Save the physical model weights (.pth)
            target_model_dir = os.path.join(MODELS_DIR, target_col)
            os.makedirs(target_model_dir, exist_ok=True)
            model_filename = os.path.join(target_model_dir, f"model_{patch_size}_step{step}.pth")
            torch.save(best_weights, model_filename)
            
            results.append({
                'Patch_Size': f"{patch_size}x{patch_size}",
                'Features': config_name,
                'Num_Indices': step,
                'R2': round(r2, 4),
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'Huber': round(huber, 4),
                'LogCosh': round(log_cosh, 4),
                'Time_s': round(elapsed_time, 1)
            })
            
    # Save CSV
    os.makedirs('ML/results', exist_ok=True)
    csv_path = f'ML/results/ablation_summary_{target_col}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Ablation study complete for {target_col}. Results saved to {csv_path}.")
    return results

if __name__ == '__main__':
    print("Initializing RAM Cache for Ablation Study...")
    df = pd.read_csv("data/features/master_dataset.csv", low_memory=False)
    # The GEE patches were generated based on locations valid for ph, k, p.
    # We will use this exact spatial footprint footprint to load patches into RAM.
    df_valid = df.dropna(subset=['ph', 'k', 'p']).reset_index(drop=True)
    unique_locations = df_valid[['grid_id', 'centroid_lon', 'centroid_lat', 'sampling_date']].drop_duplicates()
    base_df = df_valid.loc[unique_locations.index].copy()
    
    # Pre-allocate Tensors for 16, 32, 64
    num_samples = len(base_df)
    tensor_16 = torch.zeros((num_samples, 18, 16, 16), dtype=torch.float32)
    tensor_64 = torch.zeros((num_samples, 18, 64, 64), dtype=torch.float32)
    
    # Read patches matching idx of base_df
    print("Loading 16x16 and 64x64 patches into memory...")
    for idx_pos, row in tqdm(base_df.reset_index().iterrows(), total=num_samples):
        original_idx = row['index']
        # Load 16x16
        p16_path = f"data/patches_16/patch_idx_{original_idx}.npy"
        if os.path.exists(p16_path):
            arr16 = np.load(p16_path)
            tensor_16[idx_pos] = torch.tensor(arr16)
            
        # Load 64x64
        p64_path = f"data/patches_64/patch_idx_{original_idx}.npy"
        if os.path.exists(p64_path):
            arr64 = np.load(p64_path)
            tensor_64[idx_pos] = torch.tensor(arr64)

    print("Dynamically Center-Cropping 8x8 from 16x16...")
    # Center crop 16x16 -> 8x8: 16//2 - 4 = 4 to 12
    tensor_8 = tensor_16[:, :, 4:12, 4:12]

    patch_cache = {
        8: tensor_8
    }

    # Order of EE bands: B1-B12 (12 bands), NDVI, BSI, NDSI, NDWI, RECI, DEM
    # Indices: 0-11: optical. 12-16: indices. 17: DEM.
    base_bands = list(range(12)) + [17]
    proxy_indices = [12, 13, 14, 15, 16]

    # Run ablation for all main targets
    targets = ['ph', 'k', 's', 'p', 'hu', 'no3']
    all_results = []
    
    for t in targets:
        print(f"\n{'='*60}\n🚀 STARTING ABLATION FOR TARGET: {t.upper()}\n{'='*60}")
        # Make a copy and drop rows where the current target is NaN
        target_df = base_df.copy()
        target_df = target_df.dropna(subset=[t]).reset_index(drop=True)
        
        # Call the ablation study logic
        target_res = run_ablation_study(target_df, patch_cache, base_bands, proxy_indices, target_col=t)
        
        # Add target col internally
        for row in target_res:
            row['Target'] = t.upper()
        all_results.extend(target_res)
        
    global_csv_path = 'ML/results/ablation_summary_8x8.csv'
    with open(global_csv_path, 'w', newline='') as f:
        # Pushing 'Target' to front of keys
        all_keys = ['Target'] + [k for k in all_results[0].keys() if k != 'Target']
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n🎉 FULL EXPERIMENT COMPLETE! Saved 108 model logs to {global_csv_path}")

