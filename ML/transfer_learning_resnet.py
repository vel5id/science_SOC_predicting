"""
transfer_learning_resnet.py
----------------------------
ResNet-18: From Scratch vs ImageNet Transfer Learning
Cross-strategy comparison: Field-LOFO (81 folds) and Farm-LOFO (20 folds)

For TL initialisation: load pretrained ResNet-18 (ImageNet RGB weights),
then replace conv1 with an 18-channel conv whose weights are obtained by
averaging the 3 RGB output channels and tiling to 18 channels (standard
"channel expansion" technique for multispectral imagery).

Outputs:
  ML/results/transfer_learning/tl_results_field_lofo.csv
  ML/results/transfer_learning/tl_results_farm_lofo.csv
  ML/results/transfer_learning/tl_summary.json
"""
import os, sys, json, copy, time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms.functional as TF

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PATCHES_DIR  = ROOT / "data" / "patches_18ch"
DATA_CSV     = ROOT / "data" / "features" / "master_dataset.csv"
OUT_DIR      = ROOT / "ML" / "results" / "transfer_learning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Configuration ─────────────────────────────────────────────────────────────
TARGETS       = ["ph", "soc", "no3", "p", "k", "s"]
FARM_COL      = "farm"
FIELD_COL     = "field_name"
IN_CHANNELS   = 18

BATCH_SIZE    = 32
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
EPOCHS        = 60
PATIENCE      = 10


# ─── Model ─────────────────────────────────────────────────────────────────────
def build_resnet(pretrained: bool, in_channels: int = IN_CHANNELS) -> nn.Module:
    """
    Build ResNet-18 for regression with IN_CHANNELS input channels.

    pretrained=True  → load ImageNet weights, adapt conv1 by mean-tiling RGB→N
    pretrained=False → random initialisation (from scratch)
    """
    if pretrained:
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        old_w = base.conv1.weight.data          # [64, 3, 7, 7]
        mean_w = old_w.mean(dim=1, keepdim=True) # [64, 1, 7, 7]
        new_w = mean_w.repeat(1, in_channels, 1, 1)  # [64, 18, 7, 7]
        new_w = new_w / (in_channels / 3.0)     # scale to preserve activation magnitude
    else:
        base = models.resnet18(weights=None)
        new_w = None

    base.conv1 = nn.Conv2d(
        in_channels, 64,
        kernel_size=7, stride=2, padding=3, bias=False
    )
    if new_w is not None:
        base.conv1.weight.data = new_w

    num_ftrs = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(128, 1)
    )
    return base


# ─── Augmented Dataset ─────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, augment: bool = False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.augment:
            if torch.rand(1) < 0.5:
                x = TF.hflip(x)
            if torch.rand(1) < 0.5:
                x = TF.vflip(x)
            k = np.random.randint(0, 4)
            if k:
                x = torch.rot90(x, k, dims=[1, 2])
        return x, y


# ─── Data loading ──────────────────────────────────────────────────────────────
def load_patches(df: pd.DataFrame, target: str):
    """
    Load patches matching df rows that have a non-NaN target and a patch file.
    Returns X_tensor, y_array, farm_array, field_array, valid_df_indices.
    """
    patches, ys, farms, fields, kept_idx = [], [], [], [], []
    for loc_i, (df_idx, row) in enumerate(df.iterrows()):
        if pd.isna(row[target]):
            continue
        pf = PATCHES_DIR / f"patch_idx_{df_idx}.npy"
        if not pf.exists():
            continue
        try:
            arr = np.load(pf).astype(np.float32)   # [C, H, W]
            patches.append(torch.from_numpy(arr))
            ys.append(float(row[target]))
            farms.append(str(row[FARM_COL]))
            fields.append(str(row[FIELD_COL]))
            kept_idx.append(loc_i)
        except Exception as e:
            print(f"[WARN] Cannot load {pf}: {e}")

    if not patches:
        raise RuntimeError(f"No valid patches found for target '{target}'")

    X = torch.stack(patches)            # [N, C, H, W]
    y = np.array(ys, dtype=np.float32)
    return X, y, np.array(farms), np.array(fields), np.array(kept_idx)


# ─── Training one fold ─────────────────────────────────────────────────────────
def _train_fold(X_all, train_idx, val_idx, test_idx, y_all, pretrained: bool):
    """Train one fold, return OOF predictions for test_idx."""
    # Per-fold channel-wise normalisation (train statistics only)
    fold_mean = X_all[train_idx].mean(dim=(0, 2, 3), keepdim=True)
    fold_std  = X_all[train_idx].std(dim=(0, 2, 3), keepdim=True)
    fold_std[fold_std == 0] = 1.0
    X_norm = (X_all - fold_mean) / fold_std

    y_t = torch.tensor(y_all, dtype=torch.float32).unsqueeze(1)

    tr_ds  = PatchDataset(X_norm[train_idx], y_t[train_idx], augment=True)
    val_ds = PatchDataset(X_norm[val_idx],   y_t[val_idx],   augment=False)
    te_ds  = PatchDataset(X_norm[test_idx],  y_t[test_idx],  augment=False)

    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    te_dl  = DataLoader(te_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = build_resnet(pretrained=pretrained).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    best_val, best_state, patience_cnt = float("inf"), None, 0

    epoch_bar = tqdm(range(EPOCHS), desc="      epoch", leave=False)
    for epoch in epoch_bar:
        model.train()
        for bX, by in tr_dl:
            bX, by = bX.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bX, by in val_dl:
                bX, by = bX.to(DEVICE), by.to(DEVICE)
                val_loss += criterion(model(bX), by).item() * bX.size(0)
        val_loss /= max(len(val_dl.dataset), 1)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
        epoch_bar.set_postfix(val=f"{val_loss:.4f}", best=f"{best_val:.4f}", pat=f"{patience_cnt}/{PATIENCE}")
        if patience_cnt >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for bX, _ in te_dl:
            preds.extend(model(bX.to(DEVICE)).cpu().numpy().flatten())
    return np.array(preds)


# ─── Field-LOFO ────────────────────────────────────────────────────────────────
def run_field_lofo(target: str, pretrained: bool, label: str) -> dict:
    print(f"\n  [Field-LOFO | {label}] target={target.upper()}")
    df = pd.read_csv(DATA_CSV)
    X, y, farms, fields, _ = load_patches(df, target)

    unique_fields = np.unique(fields)
    oof = np.zeros(len(y))

    for test_field in tqdm(unique_fields, desc=f"  Field-LOFO {target} {label}", leave=False):
        test_idx  = np.where(fields == test_field)[0]
        other     = unique_fields[unique_fields != test_field]
        if len(other) == 0:
            continue
        val_field = np.random.RandomState(SEED).choice(other)
        val_idx   = np.where(fields == val_field)[0]
        train_idx = np.where((fields != test_field) & (fields != val_field))[0]

        if len(train_idx) < BATCH_SIZE:
            continue

        fold_preds = _train_fold(X, train_idx, val_idx, test_idx, y, pretrained)
        oof[test_idx] = fold_preds

    rho, _  = spearmanr(y, oof)
    r2      = r2_score(y, oof)
    rmse    = np.sqrt(mean_squared_error(y, oof))
    mae     = mean_absolute_error(y, oof)
    n       = len(y)

    print(f"    rho={rho:.4f}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  N={n}")
    return {"target": target, "mode": label, "strategy": "field_lofo",
            "rho": round(rho, 4), "r2": round(r2, 4),
            "rmse": round(rmse, 4), "mae": round(mae, 4), "n": n}


# ─── Farm-LOFO ─────────────────────────────────────────────────────────────────
def run_farm_lofo(target: str, pretrained: bool, label: str) -> dict:
    print(f"\n  [Farm-LOFO  | {label}] target={target.upper()}")
    df = pd.read_csv(DATA_CSV)
    X, y, farms, fields, _ = load_patches(df, target)

    unique_farms = np.unique(farms)
    oof = np.zeros(len(y))

    for test_farm in tqdm(unique_farms, desc=f"  Farm-LOFO  {target} {label}", leave=False):
        test_idx  = np.where(farms == test_farm)[0]
        other     = unique_farms[unique_farms != test_farm]
        if len(other) == 0:
            continue
        val_farm  = np.random.RandomState(SEED).choice(other)
        val_idx   = np.where(farms == val_farm)[0]
        train_idx = np.where((farms != test_farm) & (farms != val_farm))[0]

        if len(train_idx) < BATCH_SIZE:
            continue

        fold_preds = _train_fold(X, train_idx, val_idx, test_idx, y, pretrained)
        oof[test_idx] = fold_preds

    rho, _  = spearmanr(y, oof)
    r2      = r2_score(y, oof)
    rmse    = np.sqrt(mean_squared_error(y, oof))
    mae     = mean_absolute_error(y, oof)
    n       = len(y)

    print(f"    rho={rho:.4f}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  N={n}")
    return {"target": target, "mode": label, "strategy": "farm_lofo",
            "rho": round(rho, 4), "r2": round(r2, 4),
            "rmse": round(rmse, 4), "mae": round(mae, 4), "n": n}


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    all_rows = []

    for target in tqdm(TARGETS, desc="Targets"):
        print(f"\n{'='*60}")
        print(f"  TARGET: {target.upper()}")
        print(f"{'='*60}")

        # ── Field-LOFO: Scratch ──────────────────────────────────────────────
        row = run_field_lofo(target, pretrained=False, label="scratch")
        all_rows.append(row)

        # ── Field-LOFO: Transfer Learning ───────────────────────────────────
        row = run_field_lofo(target, pretrained=True, label="tl_imagenet")
        all_rows.append(row)

        # ── Farm-LOFO: Scratch ───────────────────────────────────────────────
        row = run_farm_lofo(target, pretrained=False, label="scratch")
        all_rows.append(row)

        # ── Farm-LOFO: Transfer Learning ────────────────────────────────────
        row = run_farm_lofo(target, pretrained=True, label="tl_imagenet")
        all_rows.append(row)

    df_all = pd.DataFrame(all_rows)

    # Save split by strategy for convenience
    df_field = df_all[df_all["strategy"] == "field_lofo"].copy()
    df_farm  = df_all[df_all["strategy"] == "farm_lofo"].copy()

    df_field.to_csv(OUT_DIR / "tl_results_field_lofo.csv", index=False)
    df_farm.to_csv(OUT_DIR  / "tl_results_farm_lofo.csv",  index=False)
    df_all.to_csv(OUT_DIR   / "tl_results_all.csv",         index=False)

    # Summary JSON
    summary = {}
    for _, row in df_all.iterrows():
        key = f"{row['target']}_{row['strategy']}_{row['mode']}"
        summary[key] = {"rho": row["rho"], "r2": row["r2"],
                        "rmse": row["rmse"], "mae": row["mae"], "n": int(row["n"])}

    with open(OUT_DIR / "tl_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = (time.time() - t0) / 60
    print(f"\n[DONE] Total time: {elapsed:.1f} min")
    print(f"Results saved to: {OUT_DIR}")
    print(df_farm.to_string(index=False))


if __name__ == "__main__":
    main()
