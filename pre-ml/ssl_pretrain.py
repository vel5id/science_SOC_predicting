"""
pre-ml/ssl_pretrain.py
======================
Self-supervised pretraining (Masked Autoencoder) на табличных признаках.

Идея:
  У нас 1085 образцов с богатыми признаками (100 TS + 78 spring spectral + 8 topo
  + 4 climate = ~190 признаков), но задача предсказания серы плохо обобщается.
  SSL-pretraining помогает:
    1. Обучаем автоэнкодер на ВСЕХ признаках (без лейблов) → модель учит
       структуру данных почв/вегетации, не переобучаясь на конкретный таргет.
    2. Fine-tuning энкодера на S → encoder уже знает полезные паттерны.

Архитектура (SCARF-style Masked Autoencoder):
  - Encoder:  [n_feat] → Dense(256) → BN → GELU → Dense(128) → BN → GELU → Dense(64)
  - Decoder:  [64] → Dense(128) → GELU → Dense(256) → GELU → Dense(n_feat)
  - Pretrain: маскируем случайные 35% признаков → предсказываем их (MSE)
  - Fine-tune: encoder (опционально замороженный) + head Linear(64→1)

Сохраняет:
  pre-ml/checkpoints/encoder_pretrained.pt   — веса энкодера
  pre-ml/checkpoints/pretrain_loss.png        — кривая обучения
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ── Конфигурация ───────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "preml" / "sulfur_enriched_dataset.csv"
GROUPS_PATH = ROOT / "data" / "preml" / "feature_groups.json"
CKPT_DIR  = Path(__file__).parent / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MASK_RATIO   = 0.35    # 35% признаков маскируется при pretraining
HIDDEN_DIMS  = [256, 128, 64]  # размеры скрытых слоёв энкодера
EPOCHS_PRE   = 200             # эпохи pretraining
LR_PRE       = 1e-3
BATCH_SIZE   = 64
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# Архитектура
# ══════════════════════════════════════════════════════════════════════════════
class MLP_Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TabularEncoder(nn.Module):
    """Энкодер: сжимает n_feat признаков → embedding_dim."""
    def __init__(self, n_feat: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        dims = [n_feat] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(MLP_Block(dims[i], dims[i + 1], dropout))
        self.encoder = nn.Sequential(*layers)
        self.embedding_dim = hidden_dims[-1]

    def forward(self, x):
        return self.encoder(x)


class TabularDecoder(nn.Module):
    """Декодер: восстанавливает признаки из embedding."""
    def __init__(self, embedding_dim: int, hidden_dims: list, n_feat: int,
                 dropout: float = 0.1):
        super().__init__()
        dims = [embedding_dim] + list(reversed(hidden_dims[:-1])) + [n_feat]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(MLP_Block(dims[i], dims[i + 1], dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))  # финальный без активации
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class MaskedAutoencoder(nn.Module):
    """MAE = Encoder + Decoder с маскировкой при pretraining."""
    def __init__(self, n_feat: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        self.encoder = TabularEncoder(n_feat, hidden_dims, dropout)
        self.decoder = TabularDecoder(hidden_dims[-1], hidden_dims, n_feat, dropout)

    def forward(self, x, mask=None):
        """
        x:    [B, n_feat] — входные признаки
        mask: [B, n_feat] bool — True = маскированный (нулевой) признак
        Returns: (x_hat, z) — восстановленные признаки, embedding
        """
        x_masked = x.clone()
        if mask is not None:
            x_masked[mask] = 0.0
        z = self.encoder(x_masked)
        x_hat = self.decoder(z)
        return x_hat, z


# ══════════════════════════════════════════════════════════════════════════════
# Pretraining loop
# ══════════════════════════════════════════════════════════════════════════════
def pretrain(model: MaskedAutoencoder,
             X: torch.Tensor,
             epochs: int,
             lr: float,
             batch_size: int,
             mask_ratio: float) -> list:
    """Pretraining: маскируем random 35% признаков и восстанавливаем их."""

    dataset = TensorDataset(X)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss  = nn.MSELoss()

    model.train()
    losses = []
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)

            # Генерация маски: случайно маскируем mask_ratio признаков
            mask = torch.rand_like(batch) < mask_ratio  # [B, n_feat] bool

            x_hat, _ = model(batch, mask)

            # Loss только по маскированным признакам
            loss = mse_loss(x_hat[mask], batch[mask])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(X)
        losses.append(epoch_loss)
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs}  loss={epoch_loss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    return losses


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Загрузка данных ────────────────────────────────────────────────────────
    print("Loading dataset ...")
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Run build_sulfur_dataset.py first.")
        return

    df = pd.read_csv(DATA_PATH, low_memory=False)
    with open(GROUPS_PATH) as f:
        groups = json.load(f)

    # Признаки для pretraining = safe_features + OOF
    feature_cols = (groups["safe_features"]
                    + groups["oof_predictions"])
    feature_cols = [c for c in feature_cols if c in df.columns]

    print(f"Feature dim: {len(feature_cols)}")
    print(f"Samples: {len(df)}")

    # Нормализация
    X_raw = df[feature_cols].fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Сохраняем scaler и feature list для последующего использования
    import pickle
    with open(CKPT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(CKPT_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    # ── Создание модели ────────────────────────────────────────────────────────
    n_feat = X_tensor.shape[1]
    model  = MaskedAutoencoder(n_feat, HIDDEN_DIMS, dropout=0.1).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: MAE | n_feat={n_feat} | params={n_params:,}")
    print(f"  Encoder: {n_feat} → 256 → 128 → 64")
    print(f"  Decoder: 64 → 128 → 256 → {n_feat}")
    print(f"  Mask ratio: {MASK_RATIO:.0%}")

    # ── Pretraining ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Pretraining (MAE) for {EPOCHS_PRE} epochs ...")
    print(f"{'='*60}")
    losses = pretrain(model, X_tensor, EPOCHS_PRE, LR_PRE, BATCH_SIZE, MASK_RATIO)

    # ── Сохранение весов энкодера ──────────────────────────────────────────────
    torch.save(model.encoder.state_dict(), CKPT_DIR / "encoder_pretrained.pt")
    torch.save(model.state_dict(), CKPT_DIR / "mae_pretrained.pt")
    print(f"\nSaved encoder → {CKPT_DIR / 'encoder_pretrained.pt'}")

    # ── Финальная реконструкция (качество кодирования) ────────────────────────
    model.eval()
    with torch.no_grad():
        x_all = X_tensor.to(DEVICE)
        mask  = torch.rand_like(x_all) < MASK_RATIO
        x_hat, z = model(x_all, mask)
        recon_mse = nn.MSELoss()(x_hat[mask], x_all[mask]).item()
    print(f"Final reconstruction MSE (masked features): {recon_mse:.6f}")
    print(f"Embedding shape: {z.shape}  (each sample → 64-dim vector)")

    # ── График кривой обучения ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="#2b83ba", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked Reconstruction MSE")
    ax.set_title("SSL Pretraining: Masked Autoencoder Loss Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CKPT_DIR / "pretrain_loss.png", dpi=200)
    plt.close(fig)
    print(f"Saved loss curve → {CKPT_DIR / 'pretrain_loss.png'}")


if __name__ == "__main__":
    main()
