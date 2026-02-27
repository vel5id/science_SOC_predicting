"""
Optuna-Tuned Physics-Guided Hybrid CNN for Sulfur Prediction.
[REFINED VERSION: Fixed Data Leakage & Gated Fusion Balance]
"""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import json
import random
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 0. SETUP
# ──────────────────────────────────────────────────────────────
def set_seed(seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed);    random.seed(seed)
    torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = True
max_cpus = os.cpu_count() or 4
torch.set_num_threads(max(1, int(max_cpus * 0.8)))

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "ML/results/optuna_sulfur")
MODELS_DIR  = os.path.join(_PROJECT_ROOT, "ML/ml_models_optuna_sulfur")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SHAP_CNN_CHANNELS = [1, 2, 10, 13, 18, 27, 53]
N_CNN_CH = len(SHAP_CNN_CHANNELS)

# ──────────────────────────────────────────────────────────────
# 1. DATASET
# ──────────────────────────────────────────────────────────────
class SulfurHybridDataset(Dataset):
    def __init__(self, patches: torch.Tensor, tabular: torch.Tensor,
                 targets_s: torch.Tensor, targets_soc: torch.Tensor,
                 targets_no3: torch.Tensor, augment: bool = False):
        self.patches = patches
        self.tabular = tabular
        self.targets_s = targets_s
        self.targets_soc = targets_soc
        self.targets_no3 = targets_no3
        self.augment = augment

    def __len__(self):
        return len(self.targets_s)

    def __getitem__(self, idx):
        x = self.patches[idx, SHAP_CNN_CHANNELS, :, :]
        t = self.tabular[idx]
        y_s = self.targets_s[idx]
        y_soc = self.targets_soc[idx]
        y_no3 = self.targets_no3[idx]

        if self.augment:
            if torch.rand(1) < 0.5: x = torch.flip(x, dims=[2])
            if torch.rand(1) < 0.5: x = torch.flip(x, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0: x = torch.rot90(x, k=k, dims=[1, 2])

        return x, t, y_s, y_soc, y_no3

# ──────────────────────────────────────────────────────────────
# 2. ARCHITECTURE
# ──────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        mid = max(1, ch // r)
        self.fc = nn.Sequential(nn.Conv2d(ch, mid, 1), nn.ReLU(),
                                 nn.Conv2d(mid, ch, 1), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(F.adaptive_avg_pool2d(x, 1))

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

class PhysicsHybridSulfurCNN(nn.Module):
    def __init__(self, cnn_dim: int, emb_dim: int, dropout: float, n_blocks: int):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.emb_dim = emb_dim

        self.stem = nn.Sequential(
            nn.Conv2d(N_CNN_CH, cnn_dim, kernel_size=1),
            nn.BatchNorm2d(cnn_dim),
            nn.GELU(),
        )
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ConvNeXtBlock(cnn_dim))
        blocks += [nn.AvgPool2d(2, 2)]
        if n_blocks > 1:
            blocks += [ConvNeXtBlock(cnn_dim), nn.AvgPool2d(2, 2)]
        self.cnn_stages = nn.Sequential(*blocks)
        self.cnn_pool   = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_proj   = nn.Sequential(
            nn.Linear(cnn_dim, cnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.soc_embed = nn.Sequential(nn.Linear(1, emb_dim), nn.GELU())
        self.no3_embed = nn.Sequential(nn.Linear(1, emb_dim), nn.GELU())

        self.soc_head = nn.Linear(emb_dim, 1)
        self.no3_head = nn.Linear(emb_dim, 1)

        self.fusion = GatedFusion(cnn_dim, 3 * emb_dim, max(64, cnn_dim))
        
        self.s_head = nn.Sequential(
            nn.Linear(max(64, cnn_dim), 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, patch, tabular):
        h = self.stem(patch)
        h = self.cnn_stages(h)
        h = self.cnn_pool(h)
        h_cnn = self.cnn_proj(torch.flatten(h, 1))

        soc_val = tabular[:, 0:1]
        no3_val = tabular[:, 1:2]
        h_soc = self.soc_embed(soc_val)
        h_no3 = self.no3_embed(no3_val)

        pred_soc = self.soc_head(h_soc)
        pred_no3 = self.no3_head(h_no3)

        h_cross = h_soc * h_no3

        h_tab = torch.cat([h_soc, h_no3, h_cross], dim=1)
        h_fused = self.fusion(h_cnn, h_tab)
        
        pred_s = self.s_head(h_fused)

        return pred_soc, pred_no3, pred_s

# ──────────────────────────────────────────────────────────────
# 3. DATA LOADING (Raw Data, No Normalization Leakage)
# ──────────────────────────────────────────────────────────────
def prepare_data():
    print("Loading master dataset...")
    df = pd.read_csv(os.path.join(_PROJECT_ROOT, "data/features/master_dataset.csv"), low_memory=False)

    df_valid = df.dropna(subset=["ph", "k", "p"]).reset_index(drop=True)
    unique_locs = df_valid[["grid_id", "centroid_lon", "centroid_lat", "sampling_date"]].drop_duplicates()
    base_df = df_valid.loc[unique_locs.index].copy()
    base_df = base_df.dropna(subset=["soc", "no3", "s"]).reset_index(drop=True)
    n = len(base_df)
    
    tensor_64 = torch.zeros((n, 54, 64, 64), dtype=torch.float32)
    patches_dir = os.path.join(_PROJECT_ROOT, "data/patches_multiseason_64")
    
    for i, row in tqdm(base_df.iterrows(), total=n, desc="Loading patches"):
        p = os.path.join(patches_dir, f"patch_idx_{i}.npy")
        if os.path.exists(p):
            arr = np.load(p)
            if arr.shape[0] == 54:
                tensor_64[i] = torch.tensor(arr)

    X_full = tensor_64[:, :, 16:48, 16:48]

    # Return RAW tabular values. Normalization happens strictly per-split later.
    soc_arr = base_df["soc"].fillna(base_df["soc"].median()).values.astype(np.float32)
    no3_arr = base_df["no3"].fillna(base_df["no3"].median()).values.astype(np.float32)
    tab_raw = torch.tensor(np.column_stack([soc_arr, no3_arr]), dtype=torch.float32)

    y_s = torch.tensor(base_df["s"].values, dtype=torch.float32).unsqueeze(1)
    y_soc = torch.tensor(soc_arr, dtype=torch.float32).unsqueeze(1)
    y_no3 = torch.tensor(no3_arr, dtype=torch.float32).unsqueeze(1)

    farm_col = "field_name"
    unique_farms = np.array(base_df[farm_col].unique().tolist())
    np.random.seed(42)
    np.random.shuffle(unique_farms)
    val_count, test_count = 10, 12
    train_farms = unique_farms[:-(val_count + test_count)]
    val_farms   = unique_farms[-(val_count + test_count):-test_count]
    test_farms  = unique_farms[-test_count:]

    train_idx = base_df[base_df[farm_col].isin(train_farms)].index.values
    val_idx   = base_df[base_df[farm_col].isin(val_farms)].index.values
    test_idx  = base_df[base_df[farm_col].isin(test_farms)].index.values

    return (X_full, tab_raw, y_s, y_soc, y_no3, train_idx, val_idx, test_idx)

# ──────────────────────────────────────────────────────────────
# 4. SINGLE-FOLD TRAINING
# ──────────────────────────────────────────────────────────────
def train_one_trial(params, data, device, n_epochs=150, patience=20, return_preds=False):
    X_full, tab_raw, y_s, y_soc, y_no3, train_idx, val_idx, test_idx = data

    cnn_dim  = params["cnn_dim"]
    emb_dim  = params["emb_dim"]
    dropout  = params["dropout"]
    n_blocks = params["n_blocks"]
    lr       = params["lr"]
    wd       = params["weight_decay"]
    aux_w    = params["aux_loss_weight"]

    set_seed(42)
    model = PhysicsHybridSulfurCNN(cnn_dim, emb_dim, dropout, n_blocks).to(device)

    # STRICT DATA LEAKAGE FIX: Calculate mean/std ONLY on train_idx
    train_patches = X_full[train_idx]
    ds_mean = train_patches.mean(dim=(0, 2, 3), keepdim=True)
    ds_std  = train_patches.std(dim=(0, 2, 3), keepdim=True)
    ds_std[ds_std == 0] = 1.0
    X_full_norm = (X_full - ds_mean) / ds_std

    soc_train = tab_raw[train_idx, 0]
    no3_train = tab_raw[train_idx, 1]
    soc_m, soc_s = soc_train.mean(), soc_train.std() + 1e-8
    no3_m, no3_s = no3_train.mean(), no3_train.std() + 1e-8
    
    tab_norm = torch.empty_like(tab_raw)
    tab_norm[:, 0] = (tab_raw[:, 0] - soc_m) / soc_s
    tab_norm[:, 1] = (tab_raw[:, 1] - no3_m) / no3_s

    y_soc_norm = (y_soc - soc_m) / soc_s
    y_no3_norm = (y_no3 - no3_m) / no3_s

    def mk(idx, aug):
        return SulfurHybridDataset(X_full_norm[idx], tab_norm[idx], y_s[idx], 
                                    y_soc_norm[idx], y_no3_norm[idx], augment=aug)

    # Increased batch sizes for 80GB RAM handling
    tr_dl = DataLoader(mk(train_idx, True),  batch_size=64, shuffle=True,  num_workers=0)
    va_dl = DataLoader(mk(val_idx,   False), batch_size=128, shuffle=False, num_workers=0)

    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr/100)
    huber  = nn.HuberLoss()
    scaler = GradScaler('cuda' if device == 'cuda' else 'cpu')

    best_val, best_w, pat_cnt = float("inf"), None, 0

    for epoch in range(n_epochs):
        model.train()
        for bx, bt, by, by_soc, by_no3 in tr_dl:
            bx, bt, by, by_soc, by_no3 = bx.to(device), bt.to(device), by.to(device), by_soc.to(device), by_no3.to(device)

            opt.zero_grad(set_to_none=True)
            with autocast('cuda' if device == 'cuda' else 'cpu'):
                ps, pn, pu = model(bx, bt)
                loss_s   = huber(pu.float(), by.float())
                loss_aux = huber(ps.float(), by_soc.float()) + huber(pn.float(), by_no3.float())
                loss = loss_s + aux_w * loss_aux

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, bt, by, _, _ in va_dl:
                bx, bt, by = bx.to(device), bt.to(device), by.to(device)
                _, _, pu = model(bx, bt)
                val_loss += huber(pu, by).item() * bx.size(0)
        val_loss /= len(val_idx)

        if val_loss < best_val:
            best_val = val_loss
            best_w   = {k: v.clone() for k, v in model.state_dict().items()}
            pat_cnt  = 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                break

    if return_preds:
        model.load_state_dict(best_w)
        model.eval()
        ts_dl = DataLoader(mk(test_idx, False), batch_size=128, shuffle=False, num_workers=0)
        preds, truths = [], []
        with torch.no_grad():
            for bx, bt, by, _, _ in ts_dl:
                _, _, pu = model(bx.to(device), bt.to(device))
                preds.extend(pu.cpu().numpy().flatten())
                truths.extend(by.numpy().flatten())
        return best_val, np.array(preds), np.array(truths), best_w

    return best_val

# ──────────────────────────────────────────────────────────────
# 5. OPTUNA OBJECTIVE
# ──────────────────────────────────────────────────────────────
def build_objective(data, device):
    def objective(trial):
        params = {
            "cnn_dim":        trial.suggest_categorical("cnn_dim",  [64, 128, 256]),
            # FIXED: emb_dim scaled up to avoid tabular data drowning
            "emb_dim":        trial.suggest_categorical("emb_dim",  [64, 128, 256]),
            "dropout":        trial.suggest_float("dropout",        0.1, 0.5,  step=0.05),
            "n_blocks":       trial.suggest_int("n_blocks",         1, 3),
            "lr":             trial.suggest_float("lr",             1e-5, 5e-3, log=True),
            "weight_decay":   trial.suggest_float("weight_decay",   1e-5, 1e-2, log=True),
            "aux_loss_weight":trial.suggest_float("aux_loss_weight",0.0, 0.5,  step=0.05),
        }
        try:
            val_loss = train_one_trial(params, data, device, n_epochs=80, patience=15)
        except Exception as e:
            print(f"  Trial failed: {e}")
            return float("inf")
        return val_loss
    return objective

# ──────────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials",  type=int, default=40)
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare_data()

    if args.dry_run:
        params = {"cnn_dim": 64, "emb_dim": 64, "dropout": 0.3, "n_blocks": 1, 
                  "lr": 1e-3, "weight_decay": 1e-4, "aux_loss_weight": 0.1}
        val_loss, preds, truths, _ = train_one_trial(params, data, device, n_epochs=5, patience=5, return_preds=True)
        sys.exit(0)

    db_path = os.path.join(RESULTS_DIR, "sulfur_optuna_study.db")
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42), 
                                study_name="sulfur_cnn_physics_guided", storage=f"sqlite:///{db_path}", load_if_exists=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(build_objective(data, device), n_trials=args.n_trials, show_progress_bar=True)

    best_params = study.best_params
    val_loss, preds, truths, best_weights = train_one_trial(best_params, data, device, n_epochs=300, patience=25, return_preds=True)

    rmse = np.sqrt(mean_squared_error(truths, preds))
    mae  = mean_absolute_error(truths, preds)
    r2   = r2_score(truths, preds)
    rho, p_val = spearmanr(truths, preds)

    print(f"\n  FINAL TEST RESULTS — S (Sulfur)\n{'='*50}")
    print(f"  Spearman ρ : {rho:.4f}\n  R²         : {r2:.4f}")
    print(f"  RMSE       : {rmse:.4f}\n  MAE        : {mae:.4f}")

    torch.save(best_weights, os.path.join(MODELS_DIR, "best_physics_sulfur_cnn.pth"))