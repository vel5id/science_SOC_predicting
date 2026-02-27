"""
Refined Hybrid Sulfur CNN: 54-Channel Optuna Tuning
Includes Gated Fusion, Full Spectral Range, and Physics-Guided Tabular inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import pandas as pd
import numpy as np
import optuna
import os
import random
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

# --- Setup ---
os.makedirs("ML/results/optuna_54", exist_ok=True)
os.makedirs("ML/ml_models_54", exist_ok=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Modules ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(dropout)
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
        x = self.drop(x)
        return res + x

class GatedFusion(nn.Module):
    def __init__(self, cnn_dim, tab_dim, fusion_dim):
        super().__init__()
        self.cnn_proj = nn.Sequential(nn.Linear(cnn_dim, fusion_dim), nn.GELU())
        self.tab_proj = nn.Sequential(nn.Linear(tab_dim, fusion_dim), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, 1), nn.Sigmoid())
        self.final = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.GELU())

    def forward(self, h_cnn, h_tab):
        h_c = self.cnn_proj(h_cnn)
        h_t = self.tab_proj(h_tab)
        alpha = self.gate(torch.cat([h_c, h_t], dim=1))
        h_fused = alpha * h_c + (1 - alpha) * h_t
        return self.final(h_fused)

class PhysicsHybridSulfurCNN54(nn.Module):
    def __init__(self, in_channels=54, cnn_dim=256, emb_dim=64, n_blocks=2, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, cnn_dim, kernel_size=1),
            nn.BatchNorm2d(cnn_dim),
            nn.GELU()
        )
        self.blocks = nn.Sequential(*[ConvNeXtBlock(cnn_dim, dropout) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Tabular heads for auxiliary supervision
        self.soc_embed = nn.Sequential(nn.Linear(cnn_dim, emb_dim), nn.GELU())
        self.no3_embed = nn.Sequential(nn.Linear(cnn_dim, emb_dim), nn.GELU())
        
        self.soc_head = nn.Linear(emb_dim, 1)
        self.no3_head = nn.Linear(emb_dim, 1)

        # Fusion: CNN backbone + [Tab SOC, Tab NO3, Latent Cross]
        # h_tab_input is [soc_val, no3_val, cross_val] -> size 3 * emb_dim
        # Actually in original refactor it was tab_norm from dataset
        # Let's use 3 * emb_dim as fused tab size if we use embeddings
        self.fusion = GatedFusion(cnn_dim, 3 * emb_dim, max(128, cnn_dim))
        
        self.s_head = nn.Sequential(
            nn.Linear(max(128, cnn_dim), 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, tab_raw):
        # x: [B, 54, 32, 32]
        # tab_raw: [B, 2] (Normalized SOC, NO3)
        h = self.stem(x)
        h = self.blocks(h)
        h_cnn = torch.flatten(self.pool(h), 1)

        h_soc = self.soc_embed(h_cnn)
        h_no3 = self.no3_embed(h_cnn)
        
        pred_soc = self.soc_head(h_soc)
        pred_no3 = self.no3_head(h_no3)

        # Physics bridge: use truth-aligned embeddings and latent cross
        # Here we can either use latent embeddings or pure tabular inputs.
        # Let's use the latent ones for the cross-logic as originally intended.
        h_cross = h_soc * h_no3
        
        # Concat the latent SOC/NO3/Cross to fuse with CNN
        h_tab_latents = torch.cat([h_soc, h_no3, h_cross], dim=1)
        
        h_fused = self.fusion(h_cnn, h_tab_latents)
        pred_s = self.s_head(h_fused)

        return pred_s, pred_soc, pred_no3

# --- Data ---
class SulfurHybridDataset54(Dataset):
    def __init__(self, patches: torch.Tensor, tabular: torch.Tensor,
                  targets_s: torch.Tensor, targets_soc: torch.Tensor,
                  targets_no3: torch.Tensor, augment: bool = False):
        self.patches = patches
        self.tabular = tabular
        self.targets_s = targets_s
        self.targets_soc = targets_soc
        self.targets_no3 = targets_no3
        self.augment = augment

    def __len__(self): return len(self.targets_s)

    def __getitem__(self, idx):
        x = self.patches[idx] 
        t = self.tabular[idx]
        y_s = self.targets_s[idx]
        y_soc = self.targets_soc[idx]
        y_no3 = self.targets_no3[idx]

        if self.augment:
            if random.random() < 0.5: x = torch.flip(x, [1])
            if random.random() < 0.5: x = torch.flip(x, [2])
            rot = random.randint(0, 3)
            if rot > 0: x = torch.rot90(x, rot, [1, 2])
        return x, t, y_s, y_soc, y_no3

def prepare_data():
    df = pd.read_csv("data/features/master_dataset.csv", low_memory=False)
    df_valid = df.dropna(subset=["ph", "k", "p"]).reset_index(drop=True)
    unique_locs = df_valid[["grid_id", "centroid_lon", "centroid_lat", "sampling_date"]].drop_duplicates()
    base_df = df_valid.loc[unique_locs.index].copy()
    base_df = base_df.dropna(subset=["soc", "no3", "s"]).reset_index(drop=True)

    n = len(base_df)
    X_full = torch.zeros((n, 54, 64, 64), dtype=torch.float32)
    tab_raw = torch.zeros((n, 2), dtype=torch.float32)
    y_s = torch.zeros((n, 1), dtype=torch.float32)
    
    patches_dir = "data/patches_multiseason_64"
    for i, row in tqdm(base_df.iterrows(), total=n, desc="Loading patches"):
        p = os.path.join(patches_dir, f"patch_idx_{row.name}.npy")
        if not os.path.exists(p): # fallback
             p = os.path.join(patches_dir, f"patch_idx_{i}.npy")
        
        arr = np.load(p)
        # Use full 64x64 resolution
        X_full[i] = torch.tensor(arr)
        tab_raw[i, 0] = row["soc"]
        tab_raw[i, 1] = row["no3"]
        y_s[i, 0] = row["s"]

    # Target stats
    y_soc = tab_raw[:, 0:1]
    y_no3 = tab_raw[:, 1:2]
    
    farm_col = "field_name"
    unique_farms = np.array(base_df[farm_col].unique().tolist())
    np.random.seed(42)
    np.random.shuffle(unique_farms)
    
    v_c, t_c = 10, 12
    tr_farms = unique_farms[:-(v_c + t_c)]
    va_farms = unique_farms[-(v_c+t_c):-t_c]
    te_farms = unique_farms[-t_c:]
    
    tr_idx = base_df[base_df[farm_col].isin(tr_farms)].index.values
    va_idx = base_df[base_df[farm_col].isin(va_farms)].index.values
    te_idx = base_df[base_df[farm_col].isin(te_farms)].index.values
    
    return X_full, tab_raw, y_s, y_soc, y_no3, tr_idx, va_idx, te_idx

def train_one_trial(params, data, device, n_epochs=80, patience=15, return_preds=False):
    X_full, tab_raw, y_s, y_soc, y_no3, tr_idx, va_idx, te_idx = data
    
    # Normalization (Strict Isolation)
    tr_patches = X_full[tr_idx]
    ds_m = tr_patches.mean(dim=(0, 2, 3), keepdim=True)
    ds_s = tr_patches.std(dim=(0, 2, 3), keepdim=True)
    ds_s[ds_s == 0] = 1.0
    X_f_norm = (X_full - ds_m) / ds_s
    
    soc_tr = tab_raw[tr_idx, 0]
    no3_tr = tab_raw[tr_idx, 1]
    s_tr   = y_s[tr_idx, 0]
    
    soc_m, soc_s = soc_tr.mean(), soc_tr.std() + 1e-8
    no3_m, no3_s = no3_tr.mean(), no3_tr.std() + 1e-8
    s_m, s_s     = s_tr.mean(), s_tr.std() + 1e-8
    
    tab_norm = torch.zeros_like(tab_raw)
    tab_norm[:, 0] = (tab_raw[:, 0] - soc_m) / soc_s
    tab_norm[:, 1] = (tab_raw[:, 1] - no3_m) / no3_s
    
    y_s_n   = (y_s - s_m) / s_s
    y_soc_n = (y_soc - soc_m) / soc_s
    y_no3_n = (y_no3 - no3_m) / no3_s
    
    def mk(idx, aug):
        return SulfurHybridDataset54(X_f_norm[idx], tab_norm[idx], y_s_n[idx], y_soc_n[idx], y_no3_n[idx], augment=aug)

    tr_dl = DataLoader(mk(tr_idx, True),  batch_size=64, shuffle=True)
    va_dl = DataLoader(mk(va_idx, False), batch_size=128, shuffle=False)
    
    model = PhysicsHybridSulfurCNN54(
        cnn_dim=params["cnn_dim"], emb_dim=params["emb_dim"], 
        n_blocks=params["n_blocks"], dropout=params["dropout"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    huber = nn.HuberLoss(delta=1.0)
    scaler = GradScaler()
    
    best_v = float('inf')
    best_w = None
    p_cnt = 0
    
    for _ in range(n_epochs):
        model.train()
        for bx, bt, by, b_soc, b_no3 in tr_dl:
            bx, bt, by, b_soc, b_no3 = bx.to(device), bt.to(device), by.to(device), b_soc.to(device), b_no3.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
                ps, p_soc, p_no3 = model(bx, bt)
                loss_s = huber(ps, by)
                loss_aux = huber(p_soc, b_soc) + huber(p_no3, b_no3)
                loss = loss_s + 0.3 * loss_aux
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        model.eval()
        v_l = 0.0
        with torch.no_grad():
            for bx, bt, by, _, _ in va_dl:
                bx, bt, by = bx.to(device), bt.to(device), by.to(device)
                ps, _, _ = model(bx, bt)
                v_l += huber(ps, by).item() * bx.size(0)
        v_l /= len(va_idx)
        
        if v_l < best_v:
            best_v = v_l
            best_w = {k: v.clone() for k, v in model.state_dict().items()}
            p_cnt = 0
        else:
            p_cnt += 1
            if p_cnt >= patience: break
            
    if not return_preds: return best_v
    
    # Evaluation
    model.load_state_dict(best_w)
    model.eval()
    te_dl = DataLoader(mk(te_idx, False), batch_size=128)
    preds, truths = [], []
    with torch.no_grad():
        for bx, bt, by, _, _ in te_dl:
            ps, _, _ = model(bx.to(device), bt.to(device))
            preds.extend((ps.cpu().numpy().flatten() * s_s.item()) + s_m.item())
            truths.extend((by.cpu().numpy().flatten() * s_s.item()) + s_m.item())
    
    p, t = np.array(preds), np.array(truths)
    rho, _ = spearmanr(t, p)
    r2 = r2_score(t, p)
    return best_v, rho, r2

def objective(trial):
    params = {
        "cnn_dim": trial.suggest_categorical("cnn_dim", [128, 256, 384]),
        "emb_dim": trial.suggest_categorical("emb_dim", [64, 128]),
        "n_blocks": trial.suggest_int("n_blocks", 1, 3),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4)
    }
    return train_one_trial(params, data, device)

if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare_data()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    
    print("Best Trial:", study.best_trial.params)
    v, rho, r2 = train_one_trial(study.best_trial.params, data, device, n_epochs=300, patience=25, return_preds=True)
    print(f"\nFINAL TEST: Spearman Rho = {rho:.4f}, R2 = {r2:.4f}")
