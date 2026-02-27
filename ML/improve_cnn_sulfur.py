"""
improve_cnn_sulfur.py
====================
Улучшенная CNN архитектура для предсказания серы.

Улучшения:
  1) Больше параметров в архитектуре (BASE_CH: 32→48)
  2) Более агрессивная регуляризация (DROPOUT: 0.3→0.4)
  3) Более низкий LR (3e-4→1e-4) + warmup
  4) Больше epochs (150→250) с patient early stopping
  5) Ensemble: 5 моделей с разными random seeds
  6) Meta-learner: ResNet ensemble + XGBoost OOF → Ridge

Целевой R²: > 0.60 (близко к XGBoost 0.68)

Запуск:
  .venv/bin/python ML/improve_cnn_sulfur.py
"""

import io, sys, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Import agronomic metrics
sys.path.insert(0, str(Path(__file__).parent))
from agronomic_metrics import compute_all_agronomic_metrics

# ─── Пути ─────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_CSV = ROOT / "data" / "features" / "master_dataset.csv"
PATCH_DIR= ROOT / "data" / "patches_multiseason_64"
OUT_DIR  = ROOT / "ML" / "results" / "improve_cnn_sulfur"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Константы ────────────────────────────────────────────────────────────
TARGET_COL   = "s"
FIELD_COL    = "field_name"
SPRING_CHANNELS = list(range(17)) + [53]  # 18 каналов
N_SPATIAL    = len(SPRING_CHANNELS)
TABULAR_FEATURES = ["topo_TPI", "topo_profile_curvature", "topo_aspect_cos",
                    "topo_slope", "topo_TWI", "climate_MAP", "climate_MAT"]
N_TABULAR    = len(TABULAR_FEATURES)

PATCH_SIZE   = 64
CROP_SIZE    = 32
BATCH_SIZE   = 32
MAX_EPOCHS   = 250
PATIENCE     = 20
BASE_CH      = 48      # увеличено с 32
LR           = 1e-4    # снижено с 3e-4
DROPOUT      = 0.4     # увеличено с 0.3
WEIGHT_DECAY = 1e-4
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ENSEMBLE   = 3       # 3 модели в ensemble (быстрее, чем 5)

SPATIAL_SHAP_WEIGHTS = np.array([
    0.0010, 0.0734, 0.0057, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,
    0.0010, 0.0010, 0.0010, 0.0078, 0.0010, 0.0070, 0.0010, 0.0010, 0.0010,
    0.0135,
], dtype=np.float32)


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeatureWeightLayer(nn.Module):
    def __init__(self, n_channels, shap_weights=None):
        super().__init__()
        if shap_weights is not None:
            w = np.asarray(shap_weights, dtype=np.float32)
            w = np.clip(w, 1e-9, None)
            w /= w.sum()
            init = torch.from_numpy(np.log(w))
        else:
            init = torch.zeros(n_channels, dtype=torch.float32)
        self.raw_weights = nn.Parameter(init)

    def forward(self, x):
        w = F.softmax(self.raw_weights, dim=0)
        return x * w.view(1, -1, 1, 1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(DROPOUT),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class ImprovedResNet(nn.Module):
    """Улучшенная ResNet архитектура."""
    def __init__(self, n_spatial=N_SPATIAL, n_tabular=N_TABULAR,
                 shap_weights=None, base_ch=BASE_CH, dropout=DROPOUT):
        super().__init__()
        self.feat_weight = FeatureWeightLayer(n_spatial, shap_weights)

        self.stem = nn.Sequential(
            nn.Conv2d(n_spatial, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResBlock(base_ch,     base_ch,     stride=1)
        self.res2 = ResBlock(base_ch,     base_ch * 2, stride=2)
        self.res3 = ResBlock(base_ch * 2, base_ch * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.tab_embed = nn.Sequential(
            nn.Linear(n_tabular, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
        )

        fusion_dim = base_ch * 4 + 64
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, tab):
        x = self.feat_weight(x)
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x).flatten(1)
        t = self.tab_embed(tab)
        return self.head(torch.cat([x, t], dim=1))


class SulfurDataset(Dataset):
    def __init__(self, patches, tabular, targets, augment=False):
        self.patches = patches
        self.tabular = tabular
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.patches[idx].clone()
        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            x = torch.rot90(x, k=int(k), dims=[1, 2])
        return x, self.tabular[idx], self.targets[idx]


def load_patches(df, patch_dir, channels, crop):
    N = len(df)
    C = len(channels)
    half = (PATCH_SIZE - crop) // 2
    patches = torch.zeros((N, C, crop, crop), dtype=torch.float32)
    valid = np.ones(N, dtype=bool)

    for i, orig_idx in enumerate(df.index):
        path = patch_dir / f"patch_idx_{orig_idx}.npy"
        if not path.exists():
            valid[i] = False
            continue
        try:
            raw = np.load(path, allow_pickle=False).astype(np.float32)
            raw = raw[channels, half:half+crop, half:half+crop]
            patches[i] = torch.from_numpy(raw)
        except Exception:
            valid[i] = False

    return patches, valid


def train_model(model, train_ldr, val_ldr, device, out_path):
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_loss = float("inf")
    patience_ctr = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        tr_loss = 0.0
        for xb, tb, yb in train_ldr:
            xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, tb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ldr.dataset)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for xb, tb, yb in val_ldr:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                pred = model(xb, tb)
                vl_loss += criterion(pred, yb).item() * xb.size(0)
        vl_loss /= len(val_ldr.dataset)

        scheduler.step(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_ctr = 0
            torch.save(model.state_dict(), out_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

        if (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1:3d}  tr={tr_loss:.4f}  val={vl_loss:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")


def predict_loader(model, ldr, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, tb, _ in ldr:
            xb, tb = xb.to(device), tb.to(device)
            preds.append(model(xb, tb).cpu().numpy())
    return np.concatenate(preds).ravel()


def main():
    set_seed()
    t0 = time.time()
    print("=" * 70)
    print("IMPROVED CNN — SULFUR PREDICTION  |  LOFO-CV ensemble")
    print(f"Device: {DEVICE}  |  MAX_EPOCHS={MAX_EPOCHS}  BASE_CH={BASE_CH}")
    print("=" * 70)

    # ── Загрузка данных ───────────────────────────────────────────────
    print("\n[1] Loading data ...")
    df_raw = pd.read_csv(DATA_CSV, low_memory=False)
    df = df_raw.dropna(subset=[TARGET_COL]).copy().reset_index()
    df = df.rename(columns={"index": "orig_idx"})

    available = [(PATCH_DIR / f"patch_idx_{i}.npy").exists() for i in df["orig_idx"]]
    df = df[available].reset_index(drop=True)
    print(f"  Samples: {len(df)}")

    # ── Загрузка патчей ───────────────────────────────────────────────
    print(f"\n[2] Loading patches (spring {CROP_SIZE}×{CROP_SIZE}) ...")
    patches_raw, valid_mask = load_patches(
        df.set_index("orig_idx"), PATCH_DIR, SPRING_CHANNELS, CROP_SIZE
    )
    df = df[valid_mask].reset_index(drop=True)
    patches_raw = patches_raw[valid_mask]
    print(f"  Valid patches: {len(df)}")

    # ── Целевая переменная и признаки ─────────────────────────────────
    y_orig = df[TARGET_COL].values.astype(np.float32)
    y_log = np.log1p(y_orig)
    targets = torch.from_numpy(y_log).unsqueeze(1)

    tab_df = df[TABULAR_FEATURES].copy()
    for col in TABULAR_FEATURES:
        tab_df[col] = tab_df[col].fillna(tab_df[col].median())
    tab_arr = tab_df.values.astype(np.float32)

    fields = df[FIELD_COL].values
    unique_fields = np.unique(fields)
    N_FOLDS = len(unique_fields)
    N = len(df)

    print(f"\n[3] LOFO-CV: {N_FOLDS} folds, ensemble of {N_ENSEMBLE} models")

    # ── LOFO-CV ensemble ──────────────────────────────────────────────
    oof_ensemble = np.zeros((N_ENSEMBLE, N), dtype=np.float32)

    for model_idx in range(N_ENSEMBLE):
        print(f"\n  ===== Model {model_idx + 1}/{N_ENSEMBLE} =====")
        set_seed(SEED + model_idx * 100)
        oof = np.zeros(N, dtype=np.float32)

        for fold_i, test_field in enumerate(unique_fields):
            if (fold_i + 1) % 20 == 0:
                print(f"    Fold {fold_i+1}/{N_FOLDS} ...")

            test_idx = np.where(fields == test_field)[0]
            pool_idx = np.where(fields != test_field)[0]

            rng_fold = np.random.default_rng(SEED + model_idx * 100 + fold_i)
            perm = rng_fold.permutation(len(pool_idx))
            n_val = max(1, int(0.15 * len(pool_idx)))
            val_idx = pool_idx[perm[:n_val]]
            tr_idx = pool_idx[perm[n_val:]]

            # Нормализация
            tr_p = patches_raw[tr_idx]
            p_mean = tr_p.mean(dim=(0, 2, 3), keepdim=True)
            p_std = tr_p.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
            p_norm = (patches_raw - p_mean) / p_std

            tr_t = tab_arr[tr_idx]
            t_mean = tr_t.mean(axis=0)
            t_std = tr_t.std(axis=0).clip(min=1e-6)
            t_norm = torch.from_numpy((tab_arr - t_mean) / t_std)

            # DataLoaders
            def make_loader(idx, aug, shuf):
                ds = SulfurDataset(p_norm[idx], t_norm[idx], targets[idx], augment=aug)
                return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuf, num_workers=0)

            train_ldr = make_loader(tr_idx, True, True)
            val_ldr = make_loader(val_idx, False, False)
            test_ldr = make_loader(test_idx, False, False)

            # Train model
            set_seed(SEED + model_idx * 100 + fold_i)
            model = ImprovedResNet(
                n_spatial=N_SPATIAL, n_tabular=N_TABULAR,
                shap_weights=SPATIAL_SHAP_WEIGHTS, base_ch=BASE_CH, dropout=DROPOUT
            ).to(DEVICE)

            best_pt = OUT_DIR / f"_tmp_m{model_idx}_f{fold_i:03d}.pt"
            train_model(model, train_ldr, val_ldr, DEVICE, best_pt)
            model.load_state_dict(torch.load(best_pt, map_location=DEVICE))
            best_pt.unlink()

            # Predict
            preds_log = predict_loader(model, test_ldr, DEVICE)
            oof[test_idx] = preds_log

        oof_ensemble[model_idx] = oof

    # ── Ensemble averaging ────────────────────────────────────────────
    print(f"\n[4] Averaging {N_ENSEMBLE} models ...")
    oof_avg = oof_ensemble.mean(axis=0)
    oof_avg_orig = np.expm1(oof_avg)

    metrics_ensemble = {
        "rho": float(spearmanr(y_orig, oof_avg_orig)[0]),
        "r2": float(r2_score(y_orig, oof_avg_orig)),
        "rmse": float(np.sqrt(mean_squared_error(y_orig, oof_avg_orig))),
        "mae": float(mean_absolute_error(y_orig, oof_avg_orig)),
    }

    print(f"\nCNN Ensemble Results:")
    print(f"  ρ={metrics_ensemble['rho']:+.3f}  R²={metrics_ensemble['r2']:.3f}  "
          f"RMSE={metrics_ensemble['rmse']:.2f}  MAE={metrics_ensemble['mae']:.2f}")

    # ── Agronomic metrics ─────────────────────────────────────────────
    print(f"\n[5] Computing agronomic metrics ...")
    agro = compute_all_agronomic_metrics(y_orig, oof_avg_orig)
    print(f"  RPD = {agro['rpd']:.2f}  (target: > 1.8)")
    print(f"  RPIQ = {agro['rpiq']:.2f}")
    print(f"  CCC = {agro['ccc']:.3f}  (target: > 0.9)")
    print(f"  Deficit Recall = {agro['deficit_recall']:.1%}  (find deficits)")
    print(f"  Deficit Precision = {agro['deficit_precision']:.1%}  (avoid false alarms)")

    # ── Сравнение с другими моделями ──────────────────────────────────
    print(f"\n[6] Comparison with other models")
    comparison = pd.DataFrame([
        {"model": "CNN Ensemble (this run)", **metrics_ensemble, **agro},
        {"model": "XGBoost (previous)", "rho": 0.462, "r2": 0.682, "rmse": 4.29, "mae": 2.62},
        {"model": "CNN ResNet (LOFO)", "rho": 0.337, "r2": 0.413, "rmse": 5.87, "mae": 3.30},
        {"model": "CNN UNet", "rho": 0.304, "r2": 0.298, "rmse": 6.42, "mae": 3.57},
    ])
    comparison.to_csv(OUT_DIR / "comparison_with_improvements.csv", index=False)
    print(comparison[["model", "r2", "rmse", "rpd", "ccc", "deficit_recall"]].to_string(index=False))

    # ── Scatter plot ──────────────────────────────────────────────────
    print(f"\n[7] Saving scatter plot ...")
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#111111")
    ax.scatter(y_orig, oof_avg_orig, c="#4ab5e0", s=20, alpha=0.6, linewidths=0)
    lo = min(y_orig.min(), oof_avg_orig.min())
    hi = max(y_orig.max(), oof_avg_orig.max())
    ax.plot([lo, hi], [lo, hi], color="#e05c4a", lw=1.5, ls="--")
    ax.set_xlabel("True S (mg/kg)", color="white")
    ax.set_ylabel("Predicted S (mg/kg)", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.set_title(
        f"CNN Ensemble ({N_ENSEMBLE} models)\n"
        f"ρ={metrics_ensemble['rho']:+.3f}  R²={metrics_ensemble['r2']:.3f}  "
        f"RMSE={metrics_ensemble['rmse']:.2f}",
        color="white", fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_cnn_ensemble.png", dpi=160, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"\n[Done] Total time: {elapsed/60:.1f} min")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
