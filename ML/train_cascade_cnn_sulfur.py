"""
train_cascade_cnn_sulfur.py
===========================
Cascade CNN для предсказания серы (S) на основе top-15 SHAP-признаков.

Сравниваются две архитектуры:
  A) SulfurResNet  — ResNet с progressive downsampling (spatial cascade)
  B) SulfurUNet    — U-Net с skip-connections (encoder-decoder + spatial fusion)

Общие принципы обеих архитектур:
  - Только темпорально чистые признаки: spring S2 (каналы 0-16) + DEM (53)
  - log1p(S) при обучении, expm1 для метрик (оригинальный масштаб)
  - FeatureWeightLayer инициализирован SHAP-значениями (настраиваем на будущее)
  - Tabular-ветвь: topo_TPI, topo_slope, topo_TWI, topo_profile_curvature,
                   topo_aspect_cos, climate_MAP, climate_MAT
  - Пространственный train/val/test split по фермам (59/10/12)
  - HuberLoss, AdamW, early stopping

API управления весами признаков (FeatureWeightLayer):
  layer.set_weights(shap_array)  — задать новые веса из SHAP
  layer.freeze()                 — заморозить (transfer learning)
  layer.unfreeze()               — разморозить для fine-tuning
  layer.get_weights()            — получить текущие веса (numpy)

Выход:
  ML/results/cascade_cnn_sulfur/
    resnet_best.pt               — лучшие веса ResNet
    unet_best.pt                 — лучшие веса UNet
    resnet_metrics.json          — val/test метрики ResNet
    unet_metrics.json            — val/test метрики UNet
    comparison.csv               — сравнение обеих архитектур
    channel_weights_resnet.csv   — финальные веса каналов ResNet
    channel_weights_unet.csv     — финальные веса каналов UNet
    scatter_resnet.png
    scatter_unet.png

Запуск:
  .venv/bin/python ML/train_cascade_cnn_sulfur.py
"""

import io, sys, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─── Пути ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_CSV = ROOT / "data" / "features" / "master_dataset.csv"
PATCH_DIR= ROOT / "data" / "patches_multiseason_64"
OUT_DIR  = ROOT / "ML" / "results" / "cascade_cnn_sulfur"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Константы ────────────────────────────────────────────────────────────────
# Каналы в 54-канальном патче:
#   0-11  : Spring S2 bands (B1-B12)
#   12    : Spring NDVI
#   13    : Spring BSI
#   14    : Spring NDSI
#   15    : Spring NDWI
#   16    : Spring RECI
#   17-33 : Summer  (LEAKY — не используем)
#   34-50 : Autumn  (LEAKY — не используем)
#   51-52 : Radar VV/VH 2020 (не используем для единообразия)
#   53    : DEM
SPRING_PATCH_CHANNELS: List[int] = list(range(17)) + [53]   # 18 каналов
N_SPATIAL = len(SPRING_PATCH_CHANNELS)                        # 18

SPATIAL_CHANNEL_NAMES: List[str] = [
    "s2_B1", "s2_B2", "s2_B3", "s2_B4", "s2_B5", "s2_B6",
    "s2_B7", "s2_B8", "s2_B8A", "s2_B9", "s2_B11", "s2_B12",
    "s2_NDVI", "s2_BSI", "s2_NDSI", "s2_NDWI", "s2_RECI",
    "DEM",
]

# SHAP-веса для 18 пространственных каналов (из shap_mean_abs_values.csv, log1p-версия)
# Неизвестным каналам назначается фоновое значение (среднее по измеренным минусовым)
SPATIAL_SHAP_WEIGHTS: np.ndarray = np.array([
    0.0010,  # 0  s2_B1  (нет в топ-15)
    0.0734,  # 1  s2_B2  ← #1, доминирует
    0.0057,  # 2  s2_B3  ← #12
    0.0010,  # 3  s2_B4
    0.0010,  # 4  s2_B5
    0.0010,  # 5  s2_B6
    0.0010,  # 6  s2_B7
    0.0010,  # 7  s2_B8
    0.0010,  # 8  s2_B8A
    0.0010,  # 9  s2_B9
    0.0010,  # 10 s2_B11
    0.0078,  # 11 s2_B12 ← #8
    0.0010,  # 12 s2_NDVI
    0.0070,  # 13 s2_BSI ← #9
    0.0010,  # 14 s2_NDSI
    0.0010,  # 15 s2_NDWI
    0.0010,  # 16 s2_RECI
    0.0135,  # 17 DEM    ← #3
], dtype=np.float32)

# Табличные признаки (темпорально чистые, нет в патчах)
TABULAR_FEATURES: List[str] = [
    "topo_TPI",               # SHAP #4  (0.0113)
    "topo_profile_curvature", # SHAP #11 (0.0060)
    "topo_aspect_cos",        # SHAP #14 (0.0048)
    "topo_slope",             # производная DEM
    "topo_TWI",               # топографический индекс влажности
    "climate_MAP",            # SHAP #5  (0.0106) — среднегодовые осадки
    "climate_MAT",            # среднегодовая температура
]
N_TABULAR = len(TABULAR_FEATURES)   # 7

# Обучение
TARGET_COL   = "s"
FARM_COL     = "farm"        # информационный (не для LOFO)
FIELD_COL    = "field_name"  # LOFO-CV по полям (81 поле = как в RF)
VAL_FRAC     = 0.15          # доля train-образцов для val (early-stop) внутри фолда
PATCH_SIZE   = 64
CROP_SIZE    = 32        # centre-crop до 32×32
BATCH_SIZE   = 32
MAX_EPOCHS   = 150   # 81 фолдов × 2 arch: ограничиваем для разумного времени
PATIENCE     = 10    # достаточно для ранней остановки на field-level LOFO
BASE_CH      = 32    # уменьшено с 64: 1071 образцов не требует больших архитектур
LR           = 3e-4
WEIGHT_DECAY = 1e-4
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Воспроизводимость ────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
#  1. FEATURE WEIGHT LAYER  (настраиваемый на будущее)
# ══════════════════════════════════════════════════════════════════════════════

class FeatureWeightLayer(nn.Module):
    """Обучаемые веса важности каналов, инициализированные из SHAP.

    Применяет мягкое взвешивание каналов перед первым слоем CNN:
        output = x * softmax(raw_weights)   shape: [1, C, 1, 1]

    API:
        layer.set_weights(shap_arr)  — обновить из нового SHAP-вектора
        layer.freeze()               — заморозить для transfer learning
        layer.unfreeze()             — разморозить для fine-tuning
        layer.get_weights()          — получить текущие веса (numpy, sum=1)
    """

    def __init__(self, n_channels: int,
                 shap_weights: Optional[np.ndarray] = None) -> None:
        super().__init__()
        if shap_weights is not None:
            w = np.asarray(shap_weights, dtype=np.float32)
            w = np.clip(w, 1e-9, None)
            w /= w.sum()
            # raw_weights — логиты; softmax → [0,1], сумма=1
            init = torch.from_numpy(np.log(w))
        else:
            init = torch.zeros(n_channels, dtype=torch.float32)
        self.raw_weights = nn.Parameter(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.raw_weights, dim=0)          # [C]
        return x * w.view(1, -1, 1, 1)                  # broadcast

    @torch.no_grad()
    def set_weights(self, shap_weights: np.ndarray) -> None:
        """Установить новые веса из SHAP-массива (можно вызвать в любой момент)."""
        w = np.asarray(shap_weights, dtype=np.float32)
        w = np.clip(w, 1e-9, None)
        w /= w.sum()
        self.raw_weights.data.copy_(torch.from_numpy(np.log(w)))

    def freeze(self) -> None:
        """Заморозить веса (transfer learning)."""
        self.raw_weights.requires_grad_(False)

    def unfreeze(self) -> None:
        """Разморозить веса (fine-tuning)."""
        self.raw_weights.requires_grad_(True)

    @torch.no_grad()
    def get_weights(self) -> np.ndarray:
        """Вернуть нормированные веса (sum=1) в numpy."""
        return F.softmax(self.raw_weights, dim=0).cpu().numpy()

    def weights_dataframe(self, channel_names: List[str]) -> pd.DataFrame:
        """Таблица: channel_name → weight (отсортирована по убыванию)."""
        w = self.get_weights()
        df = pd.DataFrame({"channel": channel_names, "weight": w})
        return df.sort_values("weight", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  2. АРХИТЕКТУРА A: SulfurResNet
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Стандартный残差-блок с опциональным stride для downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class SulfurResNet(nn.Module):
    """
    Residual Cascade CNN для предсказания S (скаляр на поле).

    Pipeline:
      [B, 18, 32, 32] → FeatureWeightLayer
      → Stem Conv (18→64)
      → ResBlock(64→64)
      → ResBlock(64→128, stride=2) → 16×16
      → ResBlock(128→256, stride=2) → 8×8
      → AdaptiveAvgPool → [B, 256]
      → Tabular embed (7→64)
      → Concat [B, 320] → Head → [B, 1]
    """

    def __init__(self, n_spatial: int = N_SPATIAL,
                 n_tabular: int = N_TABULAR,
                 shap_weights: Optional[np.ndarray] = None,
                 base_ch: int = 64,
                 dropout: float = 0.3) -> None:
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

        # Tabular branch
        self.tab_embed = nn.Sequential(
            nn.Linear(n_tabular, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
        )

        fusion_dim = base_ch * 4 + 64   # 256 + 64 = 320
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor,
                tab: torch.Tensor) -> torch.Tensor:
        x = self.feat_weight(x)
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x).flatten(1)           # [B, 256]
        t = self.tab_embed(tab)               # [B, 64]
        return self.head(torch.cat([x, t], dim=1))  # [B, 1]


# ══════════════════════════════════════════════════════════════════════════════
#  3. АРХИТЕКТУРА B: SulfurUNet
# ══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """Два последовательных Conv-BN-ReLU (стандартный строительный блок U-Net)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SulfurUNet(nn.Module):
    """
    U-Net для предсказания S (скаляр).

    Encoder сжимает пространство 32→16→8→4.
    Decoder разворачивает обратно 4→8→16→32 с skip-connections.
    Финальная карта [B, C, 32, 32] усредняется пулингом → скаляр.

    Skip-connections передают пространственный контекст,
    decoder обучается восстанавливать тонкие пространственные паттерны.

    Pipeline:
      [B, 18, 32, 32] → FeatureWeightLayer
      Encoder:
        enc1: DoubleConv(18→64)   → [B,64,32,32]
        enc2: Pool + DoubleConv → [B,128,16,16]
        enc3: Pool + DoubleConv → [B,256,8,8]
        btn:  Pool + DoubleConv → [B,512,4,4]
      Decoder:
        dec3: Up + Cat(enc3) → DoubleConv(768→256) → [B,256,8,8]
        dec2: Up + Cat(enc2) → DoubleConv(384→128) → [B,128,16,16]
        dec1: Up + Cat(enc1) → DoubleConv(192→64)  → [B,64,32,32]
      Head:
        AdaptiveAvgPool(1) → [B,64]
        Tabular embed (7→64)
        Concat [B,128] → Linear → [B,1]
    """

    def __init__(self, n_spatial: int = N_SPATIAL,
                 n_tabular: int = N_TABULAR,
                 shap_weights: Optional[np.ndarray] = None,
                 base_ch: int = 64,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.feat_weight = FeatureWeightLayer(n_spatial, shap_weights)

        # Encoder
        self.enc1 = DoubleConv(n_spatial, base_ch)          # 32→32
        self.enc2 = DoubleConv(base_ch,   base_ch * 2)      # 16→16
        self.enc3 = DoubleConv(base_ch*2, base_ch * 4)      # 8→8
        self.btn  = DoubleConv(base_ch*4, base_ch * 8)      # 4→4
        self.pool = nn.MaxPool2d(2)

        # Decoder (Upsample bilinear + skip concat + DoubleConv)
        self.up3  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DoubleConv(base_ch*8 + base_ch*4, base_ch * 4)   # 512+256→256
        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(base_ch*4 + base_ch*2, base_ch * 2)   # 256+128→128
        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(base_ch*2 + base_ch,   base_ch)        # 128+64→64

        # Global pooling + табличная ветвь
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.tab_embed = nn.Sequential(
            nn.Linear(n_tabular, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
        )

        self.head = nn.Sequential(
            nn.Linear(base_ch + 64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor,
                tab: torch.Tensor) -> torch.Tensor:
        x = self.feat_weight(x)

        # Encoder
        e1 = self.enc1(x)                    # [B,64,32,32]
        e2 = self.enc2(self.pool(e1))        # [B,128,16,16]
        e3 = self.enc3(self.pool(e2))        # [B,256,8,8]
        bt = self.btn(self.pool(e3))         # [B,512,4,4]

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(bt), e3], dim=1))   # [B,256,8,8]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))   # [B,128,16,16]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))   # [B,64,32,32]

        # Head
        cnn_feat = self.gap(d1).flatten(1)   # [B, 64]
        tab_feat = self.tab_embed(tab)        # [B, 64]
        return self.head(torch.cat([cnn_feat, tab_feat], dim=1))  # [B,1]


# ══════════════════════════════════════════════════════════════════════════════
#  4. DATASET
# ══════════════════════════════════════════════════════════════════════════════

def load_patches(df: pd.DataFrame,
                 patch_dir: Path,
                 channels: List[int],
                 crop: int) -> Tuple[torch.Tensor, np.ndarray]:
    """Загружает патчи для всех строк df. Возвращает тензор [N, C, crop, crop]
    и маску valid_mask[N] (False = файл не найден → строка будет удалена)."""
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
            raw = np.load(path, allow_pickle=False).astype(np.float32)  # [54, 64, 64]
            raw = raw[channels, half:half+crop, half:half+crop]           # [C, crop, crop]
            patches[i] = torch.from_numpy(raw)
        except Exception:
            valid[i] = False

    return patches, valid


class SulfurPatchDataset(Dataset):
    """Dataset для гибридной CNN: возвращает (patch, tabular, target)."""

    def __init__(self,
                 patches: torch.Tensor,   # [N, C, H, W] — нормированные
                 tabular: torch.Tensor,   # [N, T]
                 targets: torch.Tensor,   # [N, 1] — log1p(S)
                 augment: bool = False) -> None:
        self.patches = patches
        self.tabular = tabular
        self.targets = targets
        self.augment = augment

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        x = self.patches[idx].clone()
        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[2])   # horizontal flip
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[1])   # vertical flip
            k = torch.randint(0, 4, (1,)).item()
            x = torch.rot90(x, k=int(k), dims=[1, 2])
        return x, self.tabular[idx], self.targets[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  5. TRAIN / EVAL UTILS
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                out_path: Path,
                arch_name: str) -> Dict:
    """Обучение с early stopping. Возвращает историю обучения."""
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_loss = float("inf")
    patience_ctr  = 0
    history       = {"train_loss": [], "val_loss": []}

    for epoch in range(MAX_EPOCHS):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for xb, tb, yb in train_loader:
            xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, tb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for xb, tb, yb in val_loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                pred = model(xb, tb)
                vl_loss += criterion(pred, yb).item() * xb.size(0)
        vl_loss /= len(val_loader.dataset)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        scheduler.step(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), out_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"    [{arch_name}] Early stop at epoch {epoch+1}  "
                      f"best_val_loss={best_val_loss:.4f}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"    [{arch_name}] epoch {epoch+1:3d}  "
                  f"tr={tr_loss:.4f}  val={vl_loss:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    return history


def predict_loader(model: nn.Module,
                   loader: DataLoader,
                   device: torch.device) -> np.ndarray:
    """Возвращает предсказания в log-пространстве (ravel)."""
    model.eval()
    preds_log = []
    with torch.no_grad():
        for xb, tb, _ in loader:
            xb, tb = xb.to(device), tb.to(device)
            preds_log.append(model(xb, tb).cpu().numpy())
    return np.concatenate(preds_log).ravel()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Метрики в оригинальном масштабе (mg/kg)."""
    rho_val, _ = spearmanr(y_true, y_pred)
    return {
        "rho":  float(rho_val),
        "r2":   float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
    }


def scatter_plot(y_true: np.ndarray, y_pred: np.ndarray,
                 metrics: Dict, arch: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#111111")
    ax.scatter(y_true, y_pred, c="#4ab5e0", s=20, alpha=0.6, linewidths=0)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="#e05c4a", lw=1.5, ls="--")
    ax.set_xlabel("True S (mg/kg)", color="white")
    ax.set_ylabel("Predicted S (mg/kg)", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.set_title(
        f"{arch} — Sulfur (S, mg/kg)\n"
        f"ρ={metrics['rho']:+.3f}  R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.2f}",
        color="white", fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def make_loaders(patches_n: torch.Tensor,
                 tab_n: torch.Tensor,
                 targets: torch.Tensor,
                 train_idx: np.ndarray,
                 val_idx: np.ndarray,
                 test_idx: np.ndarray,
                 batch: int = BATCH_SIZE):
    """Вспомогательная: создаёт три DataLoader для одного LOFO-фолда."""
    def _dl(idx, aug, shuf):
        ds = SulfurPatchDataset(patches_n[idx], tab_n[idx], targets[idx], augment=aug)
        return DataLoader(ds, batch_size=batch, shuffle=shuf, num_workers=0)
    return _dl(train_idx, True, True), _dl(val_idx, False, False), _dl(test_idx, False, False)


def main() -> None:
    set_seed(SEED)
    print("=" * 65)
    print("CASCADE CNN — SULFUR (S)  |  LOFO-CV по полям (81)  |  log1p")
    print(f"Device: {DEVICE}  |  val_frac per fold: {VAL_FRAC}")
    print("=" * 65)

    # ── 1. Загрузка мастер-датасета ───────────────────────────────────
    print("\n[1] Loading master_dataset.csv ...")
    df_raw = pd.read_csv(DATA_CSV, low_memory=False)
    df = df_raw.dropna(subset=[TARGET_COL]).copy().reset_index()
    df = df.rename(columns={"index": "orig_idx"})
    print(f"  Samples with S: {len(df)}")

    available = [(PATCH_DIR / f"patch_idx_{i}.npy").exists() for i in df["orig_idx"]]
    df = df[available].reset_index(drop=True)
    print(f"  Samples with patches: {len(df)}")

    # ── 2. Загрузка патчей ────────────────────────────────────────────
    print(f"\n[2] Loading {len(df)} patches (spring ch0-16 + DEM53, crop {CROP_SIZE}×{CROP_SIZE}) ...")
    patches_raw, valid_mask = load_patches(
        df.set_index("orig_idx"), PATCH_DIR, SPRING_PATCH_CHANNELS, CROP_SIZE
    )
    df          = df[valid_mask].reset_index(drop=True)
    patches_raw = patches_raw[valid_mask]
    print(f"  Valid patches: {len(df)}  shape: {tuple(patches_raw.shape)}")

    # ── 3. Целевая переменная (log1p) ─────────────────────────────────
    y_orig  = df[TARGET_COL].values.astype(np.float32)
    y_log   = np.log1p(y_orig)
    targets = torch.from_numpy(y_log).unsqueeze(1)   # [N, 1]

    # ── 4. Табличные признаки ─────────────────────────────────────────
    print(f"\n[3] Tabular features ({N_TABULAR}): {TABULAR_FEATURES}")
    tab_df = df[TABULAR_FEATURES].copy()
    for col in TABULAR_FEATURES:
        tab_df[col] = tab_df[col].fillna(tab_df[col].median())
    tab_arr = tab_df.values.astype(np.float32)

    # ── 5. LOFO-CV по полям (field_name, 81 поле — как RF) ───────────
    fields        = df[FIELD_COL].values
    unique_fields = np.unique(fields)
    N_FOLDS       = len(unique_fields)
    print(f"\n[4] LOFO-CV: {N_FOLDS} folds (one field per fold, ~{int(len(df)/N_FOLDS)} samples/field)")

    arch_pairs = [("ResNet", SulfurResNet), ("UNet", SulfurUNet)]

    # Накопители OOF-предсказаний (заполняются по фолдам)
    oof = {
        arch: {"preds": np.zeros(len(df), dtype=np.float32),
               "true":  y_orig.copy()}
        for arch, _ in arch_pairs
    }
    # Считаем n_params один раз (не зависит от данных)
    last_n_params = {}
    for arch_name, ModelClass in arch_pairs:
        _tmp = ModelClass(N_SPATIAL, N_TABULAR, base_ch=BASE_CH).to("cpu")
        last_n_params[arch_name] = sum(p.numel() for p in _tmp.parameters() if p.requires_grad)
        del _tmp

    for fold_i, test_field in enumerate(unique_fields):
        print(f"\n  Fold {fold_i+1:02d}/{N_FOLDS}  |  test field: «{test_field}»", end="")

        # Тест: текущее поле
        test_idx   = np.where(fields == test_field)[0]
        # Остальные образцы — train/val pool
        pool_idx   = np.where(fields != test_field)[0]

        # Val: случайные VAL_FRAC от pool (разный seed на каждый фолд)
        rng_fold   = np.random.default_rng(SEED + fold_i)
        perm       = rng_fold.permutation(len(pool_idx))
        n_val      = max(1, int(VAL_FRAC * len(pool_idx)))
        val_idx    = pool_idx[perm[:n_val]]
        tr_idx     = pool_idx[perm[n_val:]]

        print(f"  train={len(tr_idx)}  val={len(val_idx)}  test={len(test_idx)}")

        # Нормализация только по train данным этого фолда
        tr_p   = patches_raw[tr_idx]
        p_mean = tr_p.mean(dim=(0, 2, 3), keepdim=True)
        p_std  = tr_p.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
        p_norm = (patches_raw - p_mean) / p_std

        tr_t    = tab_arr[tr_idx]
        t_mean  = tr_t.mean(axis=0)
        t_std   = tr_t.std(axis=0).clip(min=1e-6)
        t_norm  = torch.from_numpy((tab_arr - t_mean) / t_std)

        train_ldr, val_ldr, test_ldr = make_loaders(
            p_norm, t_norm, targets, tr_idx, val_idx, test_idx)

        for arch_name, ModelClass in arch_pairs:
            set_seed(SEED + fold_i)
            model = ModelClass(
                n_spatial=N_SPATIAL, n_tabular=N_TABULAR,
                shap_weights=SPATIAL_SHAP_WEIGHTS, base_ch=BASE_CH,
            ).to(DEVICE)

            best_pt = OUT_DIR / f"_tmp_{arch_name.lower()}_f{fold_i:03d}.pt"
            train_model(model, train_ldr, val_ldr, DEVICE, best_pt, arch_name)
            model.load_state_dict(torch.load(best_pt, map_location=DEVICE))
            best_pt.unlink()

            # OOF predict
            preds_log = predict_loader(model, test_ldr, DEVICE)
            oof[arch_name]["preds"][test_idx] = np.expm1(preds_log)


    # ── 6. Финальные метрики OOF ──────────────────────────────────────
    print(f"\n{'='*65}")
    print("LOFO-CV OOF RESULTS (all folds combined)")
    print(f"{'='*65}")

    rows = []
    final_metrics = {}
    for arch_name, _ in arch_pairs:
        y_true = oof[arch_name]["true"]
        y_pred = oof[arch_name]["preds"]
        m = compute_metrics(y_true, y_pred)
        final_metrics[arch_name] = m
        verdict = "✅ R²≥0.5" if m["r2"] >= 0.5 else "⚠️  R²<0.5"
        print(f"  {arch_name:8s}  ρ={m['rho']:+.3f}  R²={m['r2']:.3f}  "
              f"RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  {verdict}")
        rows.append({"architecture": arch_name, **m,
                     "n_params": last_n_params[arch_name]})

    # ── 7. Обучаем финальные модели на ВСЕХ данных ────────────────────
    print(f"\n[5] Training final models on all data (random 10% val for early-stop) ...")
    rng_final = np.random.default_rng(SEED + 999)
    all_idx   = np.arange(len(df))
    rng_final.shuffle(all_idx)
    val_n     = max(1, int(0.10 * len(df)))
    val_idx_f  = all_idx[:val_n]
    tr_idx_f   = all_idx[val_n:]

    p_mean_f = patches_raw[tr_idx_f].mean(dim=(0, 2, 3), keepdim=True)
    p_std_f  = patches_raw[tr_idx_f].std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
    p_norm_f = (patches_raw - p_mean_f) / p_std_f

    t_mean_f = tab_arr[tr_idx_f].mean(axis=0)
    t_std_f  = tab_arr[tr_idx_f].std(axis=0).clip(min=1e-6)
    t_norm_f = torch.from_numpy((tab_arr - t_mean_f) / t_std_f)

    train_ldr_f, val_ldr_f, _ = make_loaders(
        p_norm_f, t_norm_f, targets, tr_idx_f, val_idx_f, val_idx_f)

    for arch_name, ModelClass in arch_pairs:
        set_seed(SEED)
        model = ModelClass(
            n_spatial=N_SPATIAL, n_tabular=N_TABULAR,
            shap_weights=SPATIAL_SHAP_WEIGHTS, base_ch=BASE_CH,
        ).to(DEVICE)
        best_pt = OUT_DIR / f"{arch_name.lower()}_final.pt"
        train_model(model, train_ldr_f, val_ldr_f, DEVICE, best_pt, arch_name)
        model.load_state_dict(torch.load(best_pt, map_location=DEVICE))
        print(f"  [{arch_name}] final model saved → {best_pt.name}")

        # Веса финальной модели
        wdf = model.feat_weight.weights_dataframe(SPATIAL_CHANNEL_NAMES)
        wdf.to_csv(OUT_DIR / f"channel_weights_{arch_name.lower()}_final.csv", index=False)
        print(f"  [{arch_name}] top channel: {wdf.iloc[0]['channel']}  "
              f"w={wdf.iloc[0]['weight']:.4f}")

    # ── 8. Scatter-plots по OOF ───────────────────────────────────────
    print("\n[6] Saving OOF scatter plots ...")
    for arch_name, _ in arch_pairs:
        m = final_metrics[arch_name]
        scatter_plot(
            oof[arch_name]["true"], oof[arch_name]["preds"],
            m, arch_name,
            OUT_DIR / f"scatter_oof_{arch_name.lower()}.png",
        )

    # ── 9. Сравнительная таблица ──────────────────────────────────────
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(OUT_DIR / "comparison_lofo.csv", index=False)

    print("\n" + "=" * 65)
    print("FINAL SUMMARY (LOFO-CV OOF)")
    print("=" * 65)
    print(cmp_df[["architecture", "rho", "r2", "rmse", "mae"]].to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")
    print("\nAPI FeatureWeightLayer:")
    print("  model.feat_weight.set_weights(new_shap_array)  — обновить SHAP-веса")
    print("  model.feat_weight.freeze()                     — заморозить")
    print("  model.feat_weight.unfreeze()                   — разморозить")
    print("  model.feat_weight.get_weights()                — текущие веса (sum=1)")


if __name__ == "__main__":
    main()
