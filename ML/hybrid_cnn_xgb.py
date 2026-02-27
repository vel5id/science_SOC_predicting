"""
hybrid_cnn_xgb.py
=================
Гибридная архитектура: CNN экстрактор признаков + XGBoost.

Идея:
  1. Обучаем ResNet-подобный CNN на 32×32 патчах (многоканальные снимки)
  2. Извлекаем 128-мерный эмбеддинг из предпоследнего слоя CNN
  3. Конкатенируем [CNN_emb(128), tabular_feat(90)] → 218-мерный вектор
  4. XGBoost обучается на этом 218-мерном векторе

Протокол: LOFO-CV по полям (81 фолд) — нет утечки.
  В каждом фолде:
    a) CNN обучается только на train-полях
    b) Из обученного CNN извлекаются эмбеддинги train+test
    c) XGBoost обучается на train-эмбеддингах
    d) XGBoost предсказывает test

Запуск:
  .venv/bin/python ML/hybrid_cnn_xgb.py
"""

import io, sys, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from agronomic_metrics import compute_all_agronomic_metrics, format_agronomic_report

ROOT     = Path(__file__).parent.parent
DATA_CSV = ROOT / "data" / "features" / "master_dataset.csv"
PATCH_DIR = ROOT / "data" / "patches"
OUT_DIR  = ROOT / "ML" / "results" / "hybrid_cnn_xgb_farm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "s"
FIELD_COL  = "farm"
SEED       = 42

# ─── Гиперпараметры ───────────────────────────────────────────────────────────
EMB_DIM     = 128      # размер CNN-эмбеддинга
BASE_CH     = 32       # базовые каналы CNN
PATCH_SIZE  = 32       # размер патча
MAX_EPOCHS  = 80       # эпохи CNN per fold
BATCH_SIZE  = 32
LR          = 1e-3
PATIENCE    = 10       # early stopping CNN
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    random_state=SEED, verbosity=0, tree_method="hist",
    early_stopping_rounds=50,
)

# ─── Утилиты признаков ────────────────────────────────────────────────────────
def get_tabular_cols(df):
    """90 clean tabular features."""
    num_cols = df.select_dtypes(include="number").columns
    spring  = sorted([c for c in num_cols if "_spring" in c
                      and "_summer" not in c and "_autumn" not in c
                      and "_late_summer" not in c])
    topo    = sorted([c for c in num_cols if c.startswith("topo_")])
    climate = sorted([c for c in num_cols if c.startswith("climate_")])
    return spring + topo + climate


def get_channel_cols(df):
    """Спектральные каналы для построения псевдо-патча из статистик.
    Если нет реальных пространственных патчей, используем per-pixel статистики
    (mean, std, p10, p25, p50, p75, p90) по каждому каналу — имитируем пространство.
    """
    num_cols = df.select_dtypes(include="number").columns
    # S2 bands spring
    s2_bands = sorted([c for c in num_cols if c.startswith("s2_B") and "_spring" in c
                       and "_summer" not in c and "_autumn" not in c])
    # spectral indices spring
    indices  = sorted([c for c in num_cols if c.startswith("spectral_") and "_spring" in c
                       and "_summer" not in c and "_autumn" not in c])
    return s2_bands + indices


# ─── Псевдо-патч из таблицы ────────────────────────────────────────────────────
def make_pseudo_patch(row, channel_cols, patch_size=16):
    """
    Имитируем 2D-патч из scalar-значений:
    Каждый канал → константная матрица patch_size×patch_size с данным значением.
    CNN сможет выучить cross-channel взаимодействия, но не пространственные.
    """
    n_ch = len(channel_cols)
    patch = np.zeros((n_ch, patch_size, patch_size), dtype=np.float32)
    for i, col in enumerate(channel_cols):
        patch[i, :, :] = float(row[col])
    return patch


# ─── Dataset ──────────────────────────────────────────────────────────────────
class SoilPatchDataset(Dataset):
    def __init__(self, X_tab, y, channel_cols, df_rows, patch_size=16, use_real_patches=False):
        """
        X_tab: tabular features [N, 90]
        y: log1p(S) targets [N]
        channel_cols: list of column names for pseudo-patch
        df_rows: DataFrame rows for building patches
        """
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.patches = []
        for _, row in df_rows.iterrows():
            p = make_pseudo_patch(row, channel_cols, patch_size)
            self.patches.append(p)
        self.patches = torch.tensor(np.array(self.patches), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.patches[idx], self.X_tab[idx], self.y[idx]


# ─── ResNet Block ──────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


# ─── CNN Feature Extractor ────────────────────────────────────────────────────
class CNNExtractor(nn.Module):
    """
    Input: [B, n_ch, P, P] (pseudo-patch)
    Output: [B, emb_dim] (embedding)
    """
    def __init__(self, in_ch, base_ch=BASE_CH, emb_dim=EMB_DIM, patch_size=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResBlock(base_ch)
        self.pool1 = nn.AvgPool2d(2)   # P/2
        self.res2 = ResBlock(base_ch)
        self.res3 = ResBlock(base_ch)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.proj  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.pool1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.gap(x)
        return self.proj(x)


# ─── Hybrid Model (CNN + MLP head for CNN pre-training) ───────────────────────
class HybridCNNHead(nn.Module):
    """CNN экстрактор + лёгкая регрессионная голова (только для pre-train CNN)."""
    def __init__(self, cnn_extractor):
        super().__init__()
        self.cnn = cnn_extractor
        self.head = nn.Linear(EMB_DIM, 1)

    def forward(self, patches):
        emb = self.cnn(patches)
        return self.head(emb).squeeze(-1)


# ─── LOFO CV ──────────────────────────────────────────────────────────────────
def lofo_splits(fields, unique_fields):
    for i, f in enumerate(unique_fields):
        yield i, np.where(fields != f)[0], np.where(fields == f)[0]


# ─── Train CNN one fold ────────────────────────────────────────────────────────
def train_cnn_one_fold(train_patches, train_tab, train_y_log, val_patches, val_tab, val_y_log,
                       in_ch, fold_idx):
    """Обучаем CNN на train, early stopping по val loss."""
    cnn = CNNExtractor(in_ch=in_ch, base_ch=BASE_CH, emb_dim=EMB_DIM, patch_size=PATCH_SIZE).to(DEVICE)
    model = HybridCNNHead(cnn).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # Dataset
    class PatchDataset(Dataset):
        def __init__(self, P, y):
            self.P = torch.tensor(P, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.P[i], self.y[i]

    train_ds = PatchDataset(train_patches, train_y_log)
    val_ds   = PatchDataset(val_patches,   val_y_log)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for P_b, y_b in train_loader:
            P_b, y_b = P_b.to(DEVICE), y_b.to(DEVICE)
            pred = model(P_b)
            loss = F.mse_loss(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for P_b, y_b in val_loader:
                P_b, y_b = P_b.to(DEVICE), y_b.to(DEVICE)
                pred = model(P_b)
                val_losses.append(F.mse_loss(pred, y_b).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.cnn  # возвращаем только CNN-экстрактор


# ─── Extract embeddings ────────────────────────────────────────────────────────
def extract_embeddings(cnn_model, patches):
    """Batch inference для получения эмбеддингов."""
    cnn_model.eval()
    P_t = torch.tensor(patches, dtype=torch.float32)
    loader = DataLoader(P_t, batch_size=128, shuffle=False)
    embs = []
    with torch.no_grad():
        for P_b in loader:
            P_b = P_b.to(DEVICE)
            embs.append(cnn_model(P_b).cpu().numpy())
    return np.vstack(embs)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 70)
    print("HYBRID CNN + XGBoost: Soil Sulfur Prediction")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # ── Загрузка данных ────────────────────────────────────────────────────────
    print("\n[1] Loading data ...")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    df = df.dropna(subset=[TARGET_COL]).copy().reset_index(drop=True)

    tab_cols     = get_tabular_cols(df)
    channel_cols = get_channel_cols(df)

    for col in tab_cols + channel_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    y_orig = df[TARGET_COL].values.astype(np.float32)
    y_log  = np.log1p(y_orig)
    fields = df[FIELD_COL].values
    unique_fields = np.unique(fields)
    N = len(df)

    print(f"  N={N}, fields={len(unique_fields)}, tab_cols={len(tab_cols)}, "
          f"channel_cols={len(channel_cols)}")

    # ── Построение псевдо-патчей (константные матрицы) ────────────────────────
    print("\n[2] Building pseudo-patches from spectral channels ...")
    print(f"  Patch shape: [{len(channel_cols)}, {PATCH_SIZE}, {PATCH_SIZE}]")

    all_patches = np.zeros((N, len(channel_cols), PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for i in range(N):
        for ci, col in enumerate(channel_cols):
            all_patches[i, ci, :, :] = df[col].iloc[i]

    X_tab = df[tab_cols].values.astype(np.float32)

    # ── LOFO-CV ────────────────────────────────────────────────────────────────
    print(f"\n[3] LOFO-CV ({len(unique_fields)} folds) ...")
    oof_hybrid   = np.zeros(N)
    oof_xgb_only = np.zeros(N)   # baseline: XGBoost without CNN embeddings

    for fi, tr_idx, te_idx in lofo_splits(fields, unique_fields):
        if fi % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Fold {fi+1}/{len(unique_fields)}  elapsed={elapsed:.0f}s")

        # ── Inner val split (15% of train) ────────────────────────────────────
        rng = np.random.default_rng(SEED + fi)
        perm = rng.permutation(len(tr_idx))
        n_val = max(1, int(0.15 * len(tr_idx)))
        val_inner  = tr_idx[perm[:n_val]]
        tr_inner   = tr_idx[perm[n_val:]]

        # ── Normalize tabular features ────────────────────────────────────────
        sc = StandardScaler()
        X_tr_tab = sc.fit_transform(X_tab[tr_inner])
        X_val_tab = sc.transform(X_tab[val_inner])
        X_te_tab  = sc.transform(X_tab[te_idx])

        # ── Normalize patch channels ──────────────────────────────────────────
        ch_mean = all_patches[tr_inner].mean(axis=(0, 2, 3), keepdims=True)
        ch_std  = all_patches[tr_inner].std(axis=(0, 2, 3), keepdims=True) + 1e-6
        P_tr  = (all_patches[tr_inner]  - ch_mean) / ch_std
        P_val = (all_patches[val_inner] - ch_mean) / ch_std
        P_te  = (all_patches[te_idx]    - ch_mean) / ch_std

        # ── Train CNN ─────────────────────────────────────────────────────────
        cnn = train_cnn_one_fold(
            P_tr, X_tr_tab, y_log[tr_inner],
            P_val, X_val_tab, y_log[val_inner],
            in_ch=len(channel_cols), fold_idx=fi
        )

        # ── Extract CNN embeddings ────────────────────────────────────────────
        emb_tr  = extract_embeddings(cnn, P_tr)   # [len(tr_inner), 128]
        emb_val = extract_embeddings(cnn, P_val)
        emb_te  = extract_embeddings(cnn, P_te)

        # ── Combined features: [CNN_emb | tabular] ────────────────────────────
        X_tr_comb  = np.hstack([emb_tr,  X_tr_tab])
        X_val_comb = np.hstack([emb_val, X_val_tab])
        X_te_comb  = np.hstack([emb_te,  X_te_tab])

        # ── XGBoost on combined features ──────────────────────────────────────
        m_hyb = xgb.XGBRegressor(**XGB_PARAMS)
        m_hyb.fit(X_tr_comb, y_log[tr_inner],
                  eval_set=[(X_val_comb, y_log[val_inner])], verbose=False)
        oof_hybrid[te_idx] = m_hyb.predict(X_te_comb)

        # ── XGBoost on tabular only (baseline comparison) ─────────────────────
        m_tab = xgb.XGBRegressor(**XGB_PARAMS)
        m_tab.fit(X_tr_tab, y_log[tr_inner],
                  eval_set=[(X_val_tab, y_log[val_inner])], verbose=False)
        oof_xgb_only[te_idx] = m_tab.predict(X_te_tab)

        # Free CUDA memory
        del cnn
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Результаты ────────────────────────────────────────────────────────────
    print("\n[4] Results ...")
    oof_hybrid_orig   = np.expm1(oof_hybrid)
    oof_xgb_only_orig = np.expm1(oof_xgb_only)

    def get_metrics(y_true, y_pred, label):
        r2   = r2_score(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        rho  = float(spearmanr(y_true, y_pred)[0])
        print(f"  {label:30s}  ρ={rho:+.3f}  R²={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
        return r2, rho, rmse, mae

    print("=" * 70)
    r2_hyb,  rho_hyb,  rmse_hyb,  mae_hyb  = get_metrics(y_orig, oof_hybrid_orig,   "Hybrid CNN+XGB")
    r2_xgb,  rho_xgb,  rmse_xgb,  mae_xgb  = get_metrics(y_orig, oof_xgb_only_orig, "XGBoost (tab only)")

    delta_r2 = r2_hyb - r2_xgb
    print(f"\n  CNN contribution: ΔR² = {delta_r2:+.3f} "
          f"({'improvement' if delta_r2 > 0 else 'regression'})")

    # ── Agronomic metrics ──────────────────────────────────────────────────────
    print("\n[5] Agronomic metrics ...")
    agro_hyb = compute_all_agronomic_metrics(y_orig, oof_hybrid_orig)
    agro_xgb = compute_all_agronomic_metrics(y_orig, oof_xgb_only_orig)

    print(f"  Hybrid  RPD={agro_hyb['rpd']:.2f}  CCC={agro_hyb['ccc']:.3f}  "
          f"Recall={agro_hyb['deficit_recall']:.1%}  Precision={agro_hyb['deficit_precision']:.1%}")
    print(f"  XGB     RPD={agro_xgb['rpd']:.2f}  CCC={agro_xgb['ccc']:.3f}  "
          f"Recall={agro_xgb['deficit_recall']:.1%}  Precision={agro_xgb['deficit_precision']:.1%}")

    # ── Comparison table ───────────────────────────────────────────────────────
    comparison = pd.DataFrame([
        {
            "Model": "Hybrid CNN+XGBoost",
            "ρ": rho_hyb, "R²": r2_hyb, "RMSE": rmse_hyb, "MAE": mae_hyb,
            "RPD": agro_hyb["rpd"], "CCC": agro_hyb["ccc"],
            "Deficit_Recall": agro_hyb["deficit_recall"],
            "Deficit_Precision": agro_hyb["deficit_precision"],
        },
        {
            "Model": "XGBoost (tabular only)",
            "ρ": rho_xgb, "R²": r2_xgb, "RMSE": rmse_xgb, "MAE": mae_xgb,
            "RPD": agro_xgb["rpd"], "CCC": agro_xgb["ccc"],
            "Deficit_Recall": agro_xgb["deficit_recall"],
            "Deficit_Precision": agro_xgb["deficit_precision"],
        },
    ])
    print("\n" + comparison.to_string(index=False))
    comparison.to_csv(OUT_DIR / "hybrid_comparison.csv", index=False)

    # ── Scatter plots ──────────────────────────────────────────────────────────
    print("\n[6] Saving scatter plots ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0a0a0a")

    for ax, (name, oof_orig, r2, rho) in zip(axes, [
        ("Hybrid CNN+XGB", oof_hybrid_orig,   r2_hyb, rho_hyb),
        ("XGBoost (tab)", oof_xgb_only_orig,  r2_xgb, rho_xgb),
    ]):
        ax.set_facecolor("#111111")
        ax.scatter(y_orig, oof_orig, c="#4ab5e0", s=15, alpha=0.5, linewidths=0)
        lo = min(y_orig.min(), oof_orig.min())
        hi = max(y_orig.max(), oof_orig.max())
        ax.plot([lo, hi], [lo, hi], color="#e05c4a", lw=1.5, ls="--")
        ax.set_xlabel("True S (mg/kg)", color="white")
        ax.set_ylabel("Predicted S (mg/kg)", color="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.set_title(f"{name}\nρ={rho:+.3f}  R²={r2:.3f}",
                     color="white", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_hybrid_vs_xgb.png", dpi=160,
                bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Hybrid CNN+XGB:  R²={r2_hyb:.3f}  RPD={agro_hyb['rpd']:.2f}  CCC={agro_hyb['ccc']:.3f}")
    print(f"  XGBoost (base):  R²={r2_xgb:.3f}  RPD={agro_xgb['rpd']:.2f}  CCC={agro_xgb['ccc']:.3f}")
    print(f"  CNN contribution: ΔR²={delta_r2:+.3f}")
    if delta_r2 > 0.01:
        print("  ✅ CNN улучшил XGBoost!")
    elif delta_r2 > -0.01:
        print("  ➡️  CNN не изменил качество (нейтральный эффект)")
    else:
        print("  ❌ CNN ухудшил XGBoost (шум побеждает сигнал)")
    print(f"\n  Time: {elapsed/60:.1f} min")
    print(f"  Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
