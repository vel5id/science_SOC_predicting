"""
pre-ml/train_sulfur.py
======================
Обучение и оценка моделей предсказания серы (S) с использованием
SSL-pretrained энкодера и обогащённых признаков.

Сравниваемые подходы:
  1. RF_baseline   — Random Forest на safe_features + OOF
  2. XGB_baseline  — XGBoost на safe_features + OOF
  3. MLP_vanilla   — MLP без pretraining
  4. MLP_ssl       — MLP с замороженным SSL-pretrained энкодером
  5. MLP_finetune  — MLP с доучиваемым SSL-pretrained энкодером

Валидация:
  - Field-LOFO CV (186 фолдов) — основная
  - Farm-LOFO CV (20 хозяйств) — строгая

Результаты: pre-ml/results/sulfur_comparison.csv
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Импортируем архитектуру из ssl_pretrain
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ssl_pretrain import TabularEncoder, MaskedAutoencoder

warnings.filterwarnings("ignore")

# ── Пути ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "data" / "preml" / "sulfur_enriched_dataset.csv"
GROUPS_PATH = ROOT / "data" / "preml" / "feature_groups.json"
CKPT_DIR   = Path(__file__).parent / "checkpoints"
RES_DIR    = Path(__file__).parent / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

# ── Гиперпараметры ─────────────────────────────────────────────────────────────
HIDDEN_DIMS    = [256, 128, 64]   # должно совпадать с ssl_pretrain.py
EPOCHS_FT      = 150              # эпохи fine-tuning
LR_FT          = 5e-4
BATCH_SIZE     = 32
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Regression head поверх энкодера
# ══════════════════════════════════════════════════════════════════════════════
class RegressionHead(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        return self.head(z).squeeze(-1)


class SSLRegressor(nn.Module):
    """Encoder (pretrained) + Regression head."""
    def __init__(self, encoder: TabularEncoder, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        self.head    = RegressionHead(encoder.embedding_dim)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


class VanillaMLP(nn.Module):
    """MLP без pretraining (для сравнения)."""
    def __init__(self, n_feat: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        dims = [n_feat] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Утилиты метрик
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, tag=""):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t, y_p = y_true[valid], y_pred[valid]
    rho, _   = spearmanr(y_t, y_p)
    r2       = r2_score(y_t, y_p)
    rmse     = np.sqrt(mean_squared_error(y_t, y_p))
    mae      = np.mean(np.abs(y_t - y_p))
    return dict(tag=tag, rho=round(float(rho), 4), r2=round(float(r2), 4),
                rmse=round(float(rmse), 4), mae=round(float(mae), 4),
                n=int(valid.sum()))


# ══════════════════════════════════════════════════════════════════════════════
# Field-LOFO CV (используем field_name как группу)
# ══════════════════════════════════════════════════════════════════════════════
def field_lofo_predict(model_fn, X: np.ndarray, y: np.ndarray,
                       groups: np.ndarray) -> np.ndarray:
    """
    model_fn(X_train, y_train, X_test) → y_pred_test
    Возвращает OOF-предсказания того же размера, что и y.
    """
    unique_groups = np.unique(groups)
    oof = np.full(len(y), np.nan)
    for g in unique_groups:
        tr = groups != g
        te = groups == g
        if tr.sum() < 10:
            continue
        oof[te] = model_fn(X[tr], y[tr], X[te])
    return oof


def farm_lofo_predict(model_fn, X: np.ndarray, y: np.ndarray,
                      farms: np.ndarray) -> np.ndarray:
    """Farm-level LOFO CV."""
    unique_farms = np.unique(farms)
    oof = np.full(len(y), np.nan)
    for farm in unique_farms:
        tr = farms != farm
        te = farms == farm
        if tr.sum() < 10:
            continue
        oof[te] = model_fn(X[tr], y[tr], X[te])
    return oof


# ══════════════════════════════════════════════════════════════════════════════
# Model factories
# ══════════════════════════════════════════════════════════════════════════════
def rf_predict(X_train, y_train, X_test):
    rf = RandomForestRegressor(n_estimators=400, max_depth=None,
                                min_samples_leaf=2, max_features=0.5,
                                random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf.predict(X_test)


def xgb_predict(X_train, y_train, X_test):
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, verbosity=0, tree_method="hist",
        early_stopping_rounds=40,
        eval_metric="rmse",
    )
    # Разобьём train на train/early_stop (последние 15%)
    n_val = max(1, int(len(X_train) * 0.15))
    Xtr, Xvl = X_train[:-n_val], X_train[-n_val:]
    ytr, yvl = y_train[:-n_val], y_train[-n_val:]
    model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)
    return model.predict(X_test)


def make_mlp_predict(n_feat, pretrained_encoder=None, freeze=True):
    """Фабрика: возвращает функцию предсказания для MLP."""
    def predict(X_train, y_train, X_test):
        scaler_inner = StandardScaler()
        Xtr = torch.tensor(scaler_inner.fit_transform(X_train), dtype=torch.float32)
        Xte = torch.tensor(scaler_inner.transform(X_test),  dtype=torch.float32)
        ytr = torch.tensor(y_train, dtype=torch.float32)

        if pretrained_encoder is not None:
            # Загружаем веса энкодера (новая копия, чтобы не мутировать)
            enc = TabularEncoder(n_feat, HIDDEN_DIMS, dropout=0.1)
            enc.load_state_dict(pretrained_encoder)
            model = SSLRegressor(enc, freeze_encoder=freeze).to(DEVICE)
        else:
            model = VanillaMLP(n_feat, HIDDEN_DIMS + [32], dropout=0.1).to(DEVICE)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_FT, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FT)
        criterion = nn.MSELoss()

        ds = TensorDataset(Xtr, ytr)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        for _ in range(EPOCHS_FT):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(Xte.to(DEVICE)).cpu().numpy()
        return preds

    return predict


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Загрузка данных ────────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        print("Dataset not found. Run build_sulfur_dataset.py first.")
        return

    print("Loading dataset ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    with open(GROUPS_PATH) as f:
        groups_dict = json.load(f)

    feature_cols = [c for c in (groups_dict["safe_features"]
                                + groups_dict["oof_predictions"])
                    if c in df.columns]
    print(f"  Samples: {len(df)} | Features: {len(feature_cols)}")

    X  = df[feature_cols].fillna(0).values.astype(np.float32)
    y  = df["target_S"].values.astype(np.float32)
    fields  = df["field_name"].values
    farms   = df["farm"].values

    # Глобальная нормализация (для RF/XGB)
    scaler_global = StandardScaler()
    X_scaled = scaler_global.fit_transform(X)

    n_feat = X.shape[1]
    print(f"  Feature dim: {n_feat}")

    # ── Загрузка pretrained энкодера (если есть) ───────────────────────────────
    enc_path = CKPT_DIR / "encoder_pretrained.pt"
    pretrained_sd = None
    if enc_path.exists():
        pretrained_sd = torch.load(enc_path, map_location="cpu")
        print(f"Loaded pretrained encoder from {enc_path}")
    else:
        print(f"WARNING: pretrained encoder not found at {enc_path}")
        print("  Run ssl_pretrain.py first for SSL comparison.")

    # ── Определение экспериментов ──────────────────────────────────────────────
    experiments = [
        ("RF_baseline",  rf_predict),
        ("XGB_baseline", xgb_predict),
        ("MLP_vanilla",  make_mlp_predict(n_feat, pretrained_encoder=None)),
    ]
    if pretrained_sd is not None:
        experiments += [
            ("MLP_ssl_frozen",   make_mlp_predict(n_feat, pretrained_sd, freeze=True)),
            ("MLP_ssl_finetune", make_mlp_predict(n_feat, pretrained_sd, freeze=False)),
        ]

    # ── Быстрые методы для field-LOFO ─────────────────────────────────────────
    fast_experiments = [(n, fn) for n, fn in experiments
                        if n in ("RF_baseline", "XGB_baseline")]
    mlp_experiments  = [(n, fn) for n, fn in experiments
                        if n not in ("RF_baseline", "XGB_baseline")]

    # ── Field-LOFO CV (только RF/XGB, для MLP слишком долго) ──────────────────
    print(f"\n{'='*60}")
    print("Field-LOFO CV (186 folds) — RF & XGB only ...")
    print(f"{'='*60}")

    results_field = []
    oof_preds_all = {}
    for name, fn in fast_experiments:
        print(f"\n  [{name}] ...")
        oof = field_lofo_predict(fn, X_scaled, y, fields)
        oof_preds_all[name] = oof
        m = compute_metrics(y, oof, tag=name)
        m["cv"] = "field-LOFO"
        results_field.append(m)
        print(f"    rho={m['rho']:.4f}  R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}")

    # ── Farm-LOFO CV (все 5 моделей, 20 фолдов) ────────────────────────────────
    print(f"\n{'='*60}")
    print("Farm-LOFO CV (20 farms) — all 5 models ...")
    print(f"{'='*60}")

    results_farm = []
    oof_farm_all = {}
    for name, fn in experiments:
        print(f"\n  [{name}] ...")
        oof = farm_lofo_predict(fn, X_scaled, y, farms)
        oof_farm_all[name] = oof
        m = compute_metrics(y, oof, tag=name)
        m["cv"] = "farm-LOFO"
        results_farm.append(m)
        print(f"    rho={m['rho']:.4f}  R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}")

    # ── Сводная таблица ────────────────────────────────────────────────────────
    all_results = pd.DataFrame(results_field + results_farm)
    out_path = RES_DIR / "sulfur_comparison.csv"
    all_results.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print("SUMMARY — Field-LOFO CV (RF/XGB):")
    print(all_results[all_results["cv"] == "field-LOFO"]
          [["tag", "rho", "r2", "rmse", "mae"]].to_string(index=False))
    print("\nSUMMARY — Farm-LOFO CV (all models):")
    print(all_results[all_results["cv"] == "farm-LOFO"]
          [["tag", "rho", "r2", "rmse", "mae"]].to_string(index=False))
    print(f"\nSaved → {out_path}")

    # ── Scatter-plot: farm-LOFO top-2 по rho ──────────────────────────────────
    farm_df   = all_results[all_results["cv"] == "farm-LOFO"]
    best_farm = farm_df.nlargest(2, "rho")["tag"].tolist()
    fig, axes = plt.subplots(1, len(best_farm), figsize=(6 * len(best_farm), 5))
    if len(best_farm) == 1:
        axes = [axes]
    for ax, name in zip(axes, best_farm):
        oof = oof_farm_all.get(name, np.full(len(y), np.nan))
        valid = np.isfinite(oof) & np.isfinite(y)
        rho, _ = spearmanr(y[valid], oof[valid])
        ax.scatter(y[valid], oof[valid], s=8, alpha=0.4, c="#2b83ba")
        mn, mx = min(y[valid].min(), oof[valid].min()), max(y[valid].max(), oof[valid].max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.6)
        ax.set_xlabel("Observed S (mg/kg)")
        ax.set_ylabel("Predicted S (mg/kg)")
        ax.set_title(f"{name} farm-LOFO  ρ={rho:.3f}")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(RES_DIR / "sulfur_scatter.png", dpi=200)
    plt.close(fig)
    print(f"Saved scatter → {RES_DIR / 'sulfur_scatter.png'}")


if __name__ == "__main__":
    main()
