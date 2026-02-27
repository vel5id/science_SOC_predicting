"""
Two-Stage Cascaded CNN for Sulfur Prediction.
Stage 1: Train backbone + SOC/NO3 heads (no S gradient).
Stage 2: Freeze backbone, train S head on rich interaction features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import csv
import random
import os
import argparse
from tqdm import tqdm
import time

# ==========================================
# 0. REPRODUCIBILITY
# ==========================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = True
max_cpus = os.cpu_count() or 4
target_threads = max(1, int(max_cpus * 0.8))
torch.set_num_threads(target_threads)

# ==========================================
# 1. DATASET
# ==========================================
class CascadedPatchDataset(Dataset):
    def __init__(self, patches: torch.Tensor, targets_soc: torch.Tensor,
                 targets_no3: torch.Tensor, targets_s: torch.Tensor,
                 base_bands_idx: list, active_indices_idx: list,
                 augment: bool = False):
        self.patches = patches
        self.targets_soc = targets_soc
        self.targets_no3 = targets_no3
        self.targets_s = targets_s
        self.active_channels = base_bands_idx + active_indices_idx
        self.augment = augment

    def __len__(self):
        return len(self.targets_s)

    def __getitem__(self, idx):
        x = self.patches[idx, self.active_channels, :, :]
        y_soc = self.targets_soc[idx]
        y_no3 = self.targets_no3[idx]
        y_s = self.targets_s[idx]

        if self.augment:
            if torch.rand(1) < 0.5: x = torch.flip(x, dims=[2])
            if torch.rand(1) < 0.5: x = torch.flip(x, dims=[1])
            rot = torch.randint(0, 4, (1,)).item()
            if rot > 0: x = torch.rot90(x, k=rot, dims=[1, 2])

        return x, y_soc, y_no3, y_s

# ==========================================
# 2. ARCHITECTURE
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.sig(self.fc2(self.act(self.fc1(scale))))
        return x * scale

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.se = SEBlock(dim)

    def forward(self, x):
        res = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.se(x)
        return res + x

# ==========================================
# 2b. INTERACTION BLOCK + SULFUR MLP
# ==========================================
class InteractionBlock(nn.Module):
    """Computes rich interaction features from SOC/NO3 embeddings and scalar predictions."""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # Output: h_soc(E) + h_no3(E) + h_cross(E) + h_diff(E) + ratio(1) + product(1)
        self.out_dim = 4 * embed_dim + 2

    def forward(self, h_soc, h_no3, pred_soc, pred_no3):
        h_cross = h_soc * h_no3                    # element-wise product (SOC_x_NO3 analog)
        h_diff = h_soc - h_no3                      # difference
        ratio = pred_soc / (pred_no3.abs() + 1e-6)  # scalar ratio (S_NO3_ratio analog)
        product = pred_soc * pred_no3                # scalar product
        return torch.cat([h_soc, h_no3, h_cross, h_diff, ratio, product], dim=1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
    def forward(self, x):
        return x + self.net(x)


class SulfurMLP(nn.Module):
    """Deep MLP for S prediction from frozen features."""
    def __init__(self, in_dim, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, 256)
        self.block1 = ResidualMLPBlock(256, dropout)
        self.block2 = ResidualMLPBlock(256, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        x = self.proj(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


class CascadedPhysicsCNN(nn.Module):
    BACKBONE_DIM = 512
    EMBED_DIM = 64

    def __init__(self, in_channels: int):
        super().__init__()
        stem_dim = 128
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, kernel_size=1),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        )
        self.stages = nn.Sequential(
            ConvNeXtBlock(stem_dim),
            ConvNeXtBlock(stem_dim),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(stem_dim, 256, kernel_size=1),
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(256, self.BACKBONE_DIM, kernel_size=1),
            ConvNeXtBlock(self.BACKBONE_DIM),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Stage 1 heads
        self.soc_embed = nn.Sequential(nn.Linear(self.BACKBONE_DIM, self.EMBED_DIM), nn.GELU())
        self.soc_head = nn.Linear(self.EMBED_DIM, 1)
        self.no3_embed = nn.Sequential(nn.Linear(self.BACKBONE_DIM, self.EMBED_DIM), nn.GELU())
        self.no3_head = nn.Linear(self.EMBED_DIM, 1)

        # Stage 2: Interaction block + Sulfur MLP
        self.interaction = InteractionBlock(self.EMBED_DIM)
        s_in_dim = self.BACKBONE_DIM + self.interaction.out_dim  # 512 + 258 = 770
        self.s_mlp = SulfurMLP(s_in_dim, dropout=0.3)

    def extract_features(self, x):
        """Extract backbone + SOC/NO3 features (used in both stages)."""
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        h_shared = torch.flatten(x, 1)
        h_soc = self.soc_embed(h_shared)
        h_no3 = self.no3_embed(h_shared)
        pred_soc = self.soc_head(h_soc)
        pred_no3 = self.no3_head(h_no3)
        return h_shared, h_soc, h_no3, pred_soc, pred_no3

    def forward_stage1(self, x):
        """Stage 1: only SOC/NO3 predictions."""
        _, _, _, pred_soc, pred_no3 = self.extract_features(x)
        return pred_soc, pred_no3

    def forward_stage2(self, x):
        """Stage 2: S prediction from frozen backbone + interactions."""
        h_shared, h_soc, h_no3, pred_soc, pred_no3 = self.extract_features(x)
        h_interact = self.interaction(h_soc, h_no3, pred_soc, pred_no3)
        h_fused = torch.cat([h_shared, h_interact], dim=1)
        pred_s = self.s_mlp(h_fused)
        return pred_s, pred_soc, pred_no3

    def forward(self, x):
        """Full forward for evaluation."""
        return self.forward_stage2(x)

# ==========================================
# 3. LOSSES
# ==========================================
huber_fn = nn.HuberLoss(delta=1.0)

# ==========================================
# 4. TWO-STAGE TRAINING
# ==========================================
def run_cascaded_training(df, patch_cache, base_bands_idx, engineered_indices, device="cuda", dry_run=False):
    MODELS_DIR = "ML/ml_models_cascaded"
    RESULTS_DIR = "ML/results"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    farm_col = "field_name"
    s1_epochs = 3 if dry_run else 200
    s2_epochs = 3 if dry_run else 300
    patience_limit = 30

    all_idx_channels = [idx for group in engineered_indices for idx in group]
    in_c = len(base_bands_idx) + len(all_idx_channels)

    unique_farms = np.array(df[farm_col].unique().tolist())
    np.random.seed(42)
    np.random.shuffle(unique_farms)

    val_count, test_count = 10, 12
    train_farms = unique_farms[:-(val_count + test_count)]
    val_farms = unique_farms[-(val_count + test_count):-test_count]
    test_farms = unique_farms[-test_count:]

    train_idx = df[df[farm_col].isin(train_farms)].index.values
    val_idx = df[df[farm_col].isin(val_farms)].index.values
    test_idx = df[df[farm_col].isin(test_farms)].index.values

    X_full = patch_cache[32].to(torch.float32)

    # Strict per-channel normalization from train split only
    train_patches = X_full[train_idx]
    ds_mean = train_patches.mean(dim=(0, 2, 3), keepdim=True)
    ds_std = train_patches.std(dim=(0, 2, 3), keepdim=True)
    ds_std[ds_std == 0] = 1.0
    X_full = (X_full - ds_mean) / ds_std

    y_soc_raw = torch.tensor(df["soc"].values, dtype=torch.float32).unsqueeze(1)
    y_no3_raw = torch.tensor(df["no3"].values, dtype=torch.float32).unsqueeze(1)
    y_s_raw = torch.tensor(df["s"].values, dtype=torch.float32).unsqueeze(1)

    # Strict target scaling from train split only
    soc_m, soc_s = y_soc_raw[train_idx].mean(), y_soc_raw[train_idx].std() + 1e-8
    no3_m, no3_s = y_no3_raw[train_idx].mean(), y_no3_raw[train_idx].std() + 1e-8
    s_m, s_s = y_s_raw[train_idx].mean(), y_s_raw[train_idx].std() + 1e-8

    y_soc = (y_soc_raw - soc_m) / soc_s
    y_no3 = (y_no3_raw - no3_m) / no3_s
    y_s = (y_s_raw - s_m) / s_s

    scales = {"soc": (soc_m.item(), soc_s.item()),
              "no3": (no3_m.item(), no3_s.item()),
              "s":   (s_m.item(), s_s.item())}

    train_ds = CascadedPatchDataset(X_full[train_idx], y_soc[train_idx], y_no3[train_idx], y_s[train_idx], base_bands_idx, all_idx_channels, augment=True)
    val_ds   = CascadedPatchDataset(X_full[val_idx],   y_soc[val_idx],   y_no3[val_idx],   y_s[val_idx],   base_bands_idx, all_idx_channels, augment=False)
    test_ds  = CascadedPatchDataset(X_full[test_idx],  y_soc[test_idx],  y_no3[test_idx],  y_s[test_idx],  base_bands_idx, all_idx_channels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0)

    set_seed(42)
    model = CascadedPhysicsCNN(in_channels=in_c).to(device)
    scaler = GradScaler('cuda' if device == 'cuda' else 'cpu')

    # =====================
    # STAGE 1: SOC / NO3
    # =====================
    print("\n" + "="*50)
    print("  STAGE 1: Training Backbone + SOC/NO3 Heads")
    print("="*50)
    # Only optimize backbone + SOC/NO3 params (exclude s_mlp, interaction)
    s1_params = list(model.stem.parameters()) + list(model.stages.parameters()) \
              + list(model.soc_embed.parameters()) + list(model.soc_head.parameters()) \
              + list(model.no3_embed.parameters()) + list(model.no3_head.parameters())
    opt1 = torch.optim.AdamW(s1_params, lr=5e-4, weight_decay=1e-4)

    best_v1, best_w1, pat1 = float('inf'), None, 0
    t0 = time.time()

    for epoch in tqdm(range(s1_epochs), desc="Stage 1"):
        model.train()
        for bx, b_soc, b_no3, _ in train_loader:
            bx, b_soc, b_no3 = bx.to(device), b_soc.to(device), b_no3.to(device)
            opt1.zero_grad(set_to_none=True)
            with autocast('cuda' if device == 'cuda' else 'cpu'):
                p_soc, p_no3 = model.forward_stage1(bx)
                loss = 0.5 * huber_fn(p_soc, b_soc) + 0.5 * huber_fn(p_no3, b_no3)
            scaler.scale(loss).backward()
            scaler.step(opt1)
            scaler.update()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bx, b_soc, b_no3, _ in val_loader:
                bx, b_soc, b_no3 = bx.to(device), b_soc.to(device), b_no3.to(device)
                p_soc, p_no3 = model.forward_stage1(bx)
                v_loss += (0.5 * huber_fn(p_soc, b_soc) + 0.5 * huber_fn(p_no3, b_no3)).item() * bx.size(0)
        v_loss /= len(val_ds)

        if v_loss < best_v1:
            best_v1 = v_loss
            best_w1 = {k: v.clone() for k, v in model.state_dict().items()}
            pat1 = 0
        else:
            pat1 += 1
            if pat1 >= patience_limit:
                print(f"  Stage 1 early stop at epoch {epoch+1}")
                break

    s1_time = time.time() - t0
    model.load_state_dict(best_w1)
    print(f"  Stage 1 done in {s1_time:.1f}s | Best val loss: {best_v1:.5f}")

    # Evaluate Stage 1 on test
    model.eval()
    s1_preds_soc, s1_preds_no3, s1_trues_soc, s1_trues_no3 = [], [], [], []
    with torch.no_grad():
        for bx, b_soc, b_no3, _ in test_loader:
            p_soc, p_no3 = model.forward_stage1(bx.to(device))
            s1_preds_soc.extend((p_soc.cpu().numpy().flatten() * scales["soc"][1]) + scales["soc"][0])
            s1_preds_no3.extend((p_no3.cpu().numpy().flatten() * scales["no3"][1]) + scales["no3"][0])
            s1_trues_soc.extend((b_soc.numpy().flatten() * scales["soc"][1]) + scales["soc"][0])
            s1_trues_no3.extend((b_no3.numpy().flatten() * scales["no3"][1]) + scales["no3"][0])

    rho_soc, _ = spearmanr(s1_trues_soc, s1_preds_soc)
    rho_no3, _ = spearmanr(s1_trues_no3, s1_preds_no3)
    print(f"  Stage 1 Test → SOC ρ={rho_soc:.4f} | NO3 ρ={rho_no3:.4f}")

    # =====================
    # STAGE 2: S (FROZEN)
    # =====================
    print("\n" + "="*50)
    print("  STAGE 2: Training S Head (Backbone FROZEN)")
    print("="*50)
    # Freeze backbone + SOC/NO3 heads
    for p in model.stem.parameters(): p.requires_grad = False
    for p in model.stages.parameters(): p.requires_grad = False
    for p in model.pool.parameters(): p.requires_grad = False
    for p in model.soc_embed.parameters(): p.requires_grad = False
    for p in model.soc_head.parameters(): p.requires_grad = False
    for p in model.no3_embed.parameters(): p.requires_grad = False
    for p in model.no3_head.parameters(): p.requires_grad = False

    # Only train interaction + s_mlp
    s2_params = list(model.interaction.parameters()) + list(model.s_mlp.parameters())
    opt2 = torch.optim.AdamW(s2_params, lr=5e-4, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=s2_epochs)

    best_v2, best_w2, pat2 = float('inf'), None, 0
    t0 = time.time()

    for epoch in tqdm(range(s2_epochs), desc="Stage 2"):
        model.train()  # BN stays in eval due to freeze, but s_mlp dropout is active
        for bx, _, _, b_s in train_loader:
            bx, b_s = bx.to(device), b_s.to(device)
            opt2.zero_grad(set_to_none=True)
            with autocast('cuda' if device == 'cuda' else 'cpu'):
                p_s, _, _ = model.forward_stage2(bx)
                loss = huber_fn(p_s, b_s)
            scaler.scale(loss).backward()
            scaler.step(opt2)
            scaler.update()
        scheduler.step()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bx, _, _, b_s in val_loader:
                bx, b_s = bx.to(device), b_s.to(device)
                p_s, _, _ = model.forward_stage2(bx)
                v_loss += huber_fn(p_s, b_s).item() * bx.size(0)
        v_loss /= len(val_ds)

        if v_loss < best_v2:
            best_v2 = v_loss
            best_w2 = {k: v.clone() for k, v in model.state_dict().items()}
            pat2 = 0
        else:
            pat2 += 1
            if pat2 >= patience_limit:
                print(f"  Stage 2 early stop at epoch {epoch+1}")
                break

    s2_time = time.time() - t0
    model.load_state_dict(best_w2)
    print(f"  Stage 2 done in {s2_time:.1f}s | Best val loss: {best_v2:.5f}")

    # =====================
    # FINAL EVALUATION
    # =====================
    print("\n" + "="*50)
    print("  FINAL TEST RESULTS")
    print("="*50)
    model.eval()
    preds = {"soc": [], "no3": [], "s": []}
    truths = {"soc": [], "no3": [], "s": []}

    with torch.no_grad():
        for bx, b_soc, b_no3, b_s in test_loader:
            p_s, p_soc, p_no3 = model.forward_stage2(bx.to(device))
            preds["soc"].extend((p_soc.cpu().numpy().flatten() * scales["soc"][1]) + scales["soc"][0])
            preds["no3"].extend((p_no3.cpu().numpy().flatten() * scales["no3"][1]) + scales["no3"][0])
            preds["s"].extend((p_s.cpu().numpy().flatten() * scales["s"][1]) + scales["s"][0])
            truths["soc"].extend((b_soc.numpy().flatten() * scales["soc"][1]) + scales["soc"][0])
            truths["no3"].extend((b_no3.numpy().flatten() * scales["no3"][1]) + scales["no3"][0])
            truths["s"].extend((b_s.numpy().flatten() * scales["s"][1]) + scales["s"][0])

    results = {}
    for tgt in ["soc", "no3", "s"]:
        t = np.array(truths[tgt])
        p = np.array(preds[tgt])
        rmse = np.sqrt(mean_squared_error(t, p))
        mae = mean_absolute_error(t, p)
        r2 = r2_score(t, p)
        rho, _ = spearmanr(t, p)
        results[tgt] = {"R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4), "Spearman_rho": round(rho, 4)}
        print(f"  {tgt.upper():>4s}: R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  ρ={rho:.4f}")

    total_time = s1_time + s2_time
    torch.save(best_w2, os.path.join(MODELS_DIR, "cascaded_physics_cnn_v2.pth"))

    rows = [{"Target": tgt.upper(), "Stage1_s": round(s1_time, 1), "Stage2_s": round(s2_time, 1), **met} for tgt, met in results.items()]
    with open(os.path.join(RESULTS_DIR, "cascaded_cnn_v2_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return results

# ==========================================
# 5. MAIN ENTRY
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv("data/features/master_dataset.csv", low_memory=False)
    df_valid = df.dropna(subset=["ph", "k", "p"]).reset_index(drop=True)
    unique_locations = df_valid[["grid_id", "centroid_lon", "centroid_lat", "sampling_date"]].drop_duplicates()
    base_df = df_valid.loc[unique_locations.index].copy()
    base_df = base_df.dropna(subset=["soc", "no3", "s"]).reset_index(drop=True)
    
    num_samples = len(base_df)
    tensor_64 = torch.zeros((num_samples, 54, 64, 64), dtype=torch.float32)
    
    loaded = 0
    for idx_pos, row in tqdm(base_df.iterrows(), total=num_samples, desc="Loading patches"):
        p64_path = f"data/patches_multiseason_64/patch_idx_{row.name}.npy"
        if not os.path.exists(p64_path):
            p64_path = f"data/patches_multiseason_64/patch_idx_{idx_pos}.npy"
        if os.path.exists(p64_path):
            arr = np.load(p64_path)
            if arr.shape[0] == 54:
                tensor_64[idx_pos] = torch.tensor(arr)
                loaded += 1

    patch_cache = {32: tensor_64[:, :, 16:48, 16:48]}

    base_bands_idx = list(range(0, 12)) + list(range(17, 29)) + list(range(34, 46)) + [51, 52, 53]
    engineered_indices = [[12, 29, 46], [13, 30, 47], [14, 31, 48], [15, 32, 49], [16, 33, 50]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_cascaded_training(base_df, patch_cache, base_bands_idx, engineered_indices, device=device, dry_run=args.dry_run)