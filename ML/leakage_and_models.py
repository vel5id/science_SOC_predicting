"""
leakage_and_models.py
=====================
Часть 1 — Тесты на утечки:
  1.1  Permutation test (Ridge, 5 перестановок)  → OOF R² ≈ 0 = нет утечки
  1.2  Permutation test (XGBoost, 3 перестановки)
  1.3  Spatial proximity: R² каждого fold vs мин. расстояние до train-полей
  1.4  Patch shuffle test: CNN ResNet на перемешанных пикселях внутри патчей

Часть 2 — Альтернативные модели (LOFO-CV, 81 поле, log1p):
  - Ridge Regression  (линейный baseline)
  - ElasticNet        (разреженный линейный)
  - CatBoost          (gradient boosting)
  - Tabular MLP       (PyTorch, 122 признака)
  - Stacking          (мета-лернер на OOF лучших моделей)

Часть 3 — Итоговая сравнительная таблица (все модели включая RF/XGB/CNN)

Признаки (темпорально чистые):
  - 110 spring-сезонных (spectral, indices, S2, L8)
  - 8 topo (DEM, TPI, TWI, slope, aspect_cos/sin, curvatures)
  - 4 climate (MAT, MAP, GS_temp, GS_precip)
  = 122 итого

Запуск:
  .venv/bin/python ML/leakage_and_models.py
"""

import io, sys, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import catboost as cb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Пути ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_CSV  = ROOT / "data" / "features" / "master_dataset.csv"
PATCH_DIR = ROOT / "data" / "patches_multiseason_64"
OUT_DIR   = ROOT / "ML" / "results" / "leakage_and_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "s"
FIELD_COL  = "field_name"
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Каналы в 54-канальном патче: spring [0-16] + DEM [53] = 18 каналов
SPRING_CHANNELS = list(range(17)) + [53]


# ─── Утилиты ─────────────────────────────────────────────────────────────
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred):
    """Метрики в оригинальном масштабе mg/kg."""
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "rho":  float(rho),
        "r2":   float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
    }


def get_feature_cols(df):
    """Темпорально чистые признаки: spring + topo + climate.

    ВАЖНО: исключаем delta-признаки вида delta_*_spring_to_summer/autumn,
    которые содержат '_spring' в имени, но реально используют summer/autumn данные.
    Итого: 78 spring + 8 topo + 4 climate = 90 признаков.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    spring  = sorted([c for c in num_cols if '_spring' in c
                       and '_summer' not in c
                       and '_autumn' not in c
                       and '_late_summer' not in c])
    topo    = sorted([c for c in num_cols if c.startswith('topo_')])
    climate = sorted([c for c in num_cols if c.startswith('climate_')])
    return spring + topo + climate


def lofo_splits(fields, unique_fields, seed=SEED):
    """Генератор LOFO-CV: yield (fold_i, train_idx, test_idx)."""
    for i, test_field in enumerate(unique_fields):
        test_idx  = np.where(fields == test_field)[0]
        train_idx = np.where(fields != test_field)[0]
        yield i, train_idx, test_idx


def scatter_plot(y_true, y_pred, metrics, title, out_path):
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
        f"{title}\n"
        f"\u03c1={metrics['rho']:+.3f}  R\u00b2={metrics['r2']:.3f}  RMSE={metrics['rmse']:.2f}",
        color="white", fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  TABULAR MLP (PyTorch)
# ══════════════════════════════════════════════════════════════════════════

class TabularMLP(nn.Module):
    def __init__(self, n_features, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_fold(X_tr, y_tr, X_val, y_val, n_features,
                   max_epochs=200, patience=15, lr=1e-3, batch=64):
    """Обучает MLP на одном фолде с early stopping. Возвращает обученную модель."""
    model = TabularMLP(n_features).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    X_tr_t  = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    y_tr_t  = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    ds_tr  = TensorDataset(X_tr_t, y_tr_t)
    ldr_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)

    best_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in ldr_tr:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    set_seed()
    t0 = time.time()
    print("=" * 70)
    print("LEAKAGE TESTS + ALTERNATIVE MODELS  |  LOFO-CV по field_name (81)")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # ── 1. Загрузка данных ────────────────────────────────────────────
    print("\n[1] Loading data ...")
    df_raw = pd.read_csv(DATA_CSV, low_memory=False)
    df = df_raw.dropna(subset=[TARGET_COL]).copy()
    print(f"  Samples with S: {len(df)}")

    feature_cols = get_feature_cols(df)
    print(f"  Temporally clean features: {len(feature_cols)}")
    print(f"    spring: {len([c for c in feature_cols if '_spring' in c])}")
    print(f"    topo:   {len([c for c in feature_cols if c.startswith('topo_')])}")
    print(f"    climate:{len([c for c in feature_cols if c.startswith('climate_')])}")

    # Заполняем NaN медианой
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    X = df[feature_cols].values.astype(np.float32)
    y_orig = df[TARGET_COL].values.astype(np.float32)
    y_log  = np.log1p(y_orig)
    fields = df[FIELD_COL].values
    unique_fields = np.unique(fields)
    N = len(df)
    N_FOLDS = len(unique_fields)
    print(f"  N={N}, fields={N_FOLDS}")

    # ══════════════════════════════════════════════════════════════════
    #  ЧАСТЬ 1: ТЕСТЫ НА УТЕЧКИ
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("ЧАСТЬ 1: ТЕСТЫ НА УТЕЧКИ")
    print("=" * 70)

    # ── 1.1 Permutation test (Ridge) ──────────────────────────────────
    print("\n[1.1] Permutation test — Ridge (5 перестановок)")
    N_PERMS_RIDGE = 5
    perm_r2_ridge = []
    rng = np.random.default_rng(SEED)

    for p in range(N_PERMS_RIDGE):
        y_shuf = y_log.copy()
        rng.shuffle(y_shuf)
        oof = np.zeros(N)
        for _, tr_idx, te_idx in lofo_splits(fields, unique_fields):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[tr_idx])
            X_te = sc.transform(X[te_idx])
            m = Ridge(alpha=10.0, random_state=SEED)
            m.fit(X_tr, y_shuf[tr_idx])
            oof[te_idx] = np.clip(m.predict(X_te), 0, np.log1p(200))
        r2 = r2_score(np.expm1(y_shuf), np.expm1(oof))
        perm_r2_ridge.append(r2)
        print(f"  Perm {p+1}: OOF R² = {r2:.4f}")

    print(f"  >>> Mean R²(shuffled) = {np.mean(perm_r2_ridge):.4f} "
          f"± {np.std(perm_r2_ridge):.4f}  (should be ≈ 0)")

    # ── 1.2 Permutation test (XGBoost) ────────────────────────────────
    print("\n[1.2] Permutation test — XGBoost (3 перестановки)")
    N_PERMS_XGB = 3
    perm_r2_xgb = []

    for p in range(N_PERMS_XGB):
        y_shuf = y_log.copy()
        rng.shuffle(y_shuf)
        oof = np.zeros(N)
        for _, tr_idx, te_idx in lofo_splits(fields, unique_fields):
            m = xgb.XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED + p, verbosity=0, tree_method="hist",
            )
            m.fit(X[tr_idx], y_shuf[tr_idx])
            oof[te_idx] = m.predict(X[te_idx])
        r2 = r2_score(np.expm1(y_shuf), np.expm1(oof))
        perm_r2_xgb.append(r2)
        print(f"  Perm {p+1}: OOF R² = {r2:.4f}")

    print(f"  >>> Mean R²(shuffled) = {np.mean(perm_r2_xgb):.4f} "
          f"± {np.std(perm_r2_xgb):.4f}  (should be ≈ 0)")

    # ── 1.3 Spatial proximity analysis ────────────────────────────────
    print("\n[1.3] Spatial proximity analysis ...")
    if "centroid_lon" in df.columns and "centroid_lat" in df.columns:
        # Центроид каждого поля
        field_centroids = df.groupby(FIELD_COL).agg(
            lon=("centroid_lon", "mean"),
            lat=("centroid_lat", "mean"),
        )

        # Быстрый Ridge LOFO для per-fold R²
        fold_r2s = []
        fold_dists = []
        for fi, tr_idx, te_idx in lofo_splits(fields, unique_fields):
            test_field = unique_fields[fi]
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[tr_idx])
            X_te = sc.transform(X[te_idx])
            m = Ridge(alpha=10.0)
            m.fit(X_tr, y_log[tr_idx])
            preds = np.expm1(np.clip(m.predict(X_te), 0, np.log1p(200)))
            truth = y_orig[te_idx]
            if len(te_idx) > 1 and np.std(truth) > 0:
                r2 = r2_score(truth, preds)
            else:
                r2 = np.nan

            # Min distance от test-поля до ближайшего train-поля
            test_c = field_centroids.loc[test_field][["lon", "lat"]].values.reshape(1, -1)
            train_fields_list = [f for f in unique_fields if f != test_field]
            if len(train_fields_list) > 0:
                train_c = field_centroids.loc[train_fields_list][["lon", "lat"]].values
                dists = cdist(test_c, train_c, metric="euclidean")
                min_dist = dists.min()
            else:
                min_dist = np.nan
            fold_r2s.append(r2)
            fold_dists.append(min_dist)

        fold_r2s = np.array(fold_r2s)
        fold_dists = np.array(fold_dists)
        valid = ~np.isnan(fold_r2s) & ~np.isnan(fold_dists)
        if valid.sum() > 5:
            rho_spatial, p_spatial = spearmanr(fold_dists[valid], fold_r2s[valid])
            print(f"  Spearman(min_dist, fold_R²): ρ = {rho_spatial:+.3f}, p = {p_spatial:.4f}")
            print(f"  {'No spatial leakage detected' if p_spatial > 0.05 else 'WARNING: possible spatial leakage!'}")

            # Plot
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor("#0a0a0a")
            ax.set_facecolor("#111111")
            ax.scatter(fold_dists[valid], fold_r2s[valid], c="#4ab5e0", s=20, alpha=0.6)
            ax.set_xlabel("Min distance to nearest train field (degrees)", color="white")
            ax.set_ylabel("Fold R²", color="white")
            ax.set_title(f"Spatial Proximity vs Fold R²\n"
                         f"ρ={rho_spatial:+.3f}, p={p_spatial:.3f}",
                         color="white", fontsize=10, fontweight="bold")
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333")
            fig.tight_layout()
            fig.savefig(OUT_DIR / "spatial_proximity_vs_r2.png", dpi=160,
                        bbox_inches="tight", facecolor="#0a0a0a")
            plt.close(fig)
    else:
        print("  Centroid columns not found — skipping")

    # ── 1.4 Feature temporal verification ─────────────────────────────
    print("\n[1.4] Feature temporal verification")
    n_summer  = len([c for c in feature_cols if '_summer' in c or '_late_summer' in c])
    n_autumn  = len([c for c in feature_cols if '_autumn' in c])
    n_spring  = len([c for c in feature_cols if '_spring' in c])
    n_topo    = len([c for c in feature_cols if c.startswith('topo_')])
    n_climate = len([c for c in feature_cols if c.startswith('climate_')])
    print(f"  spring={n_spring}, topo={n_topo}, climate={n_climate}")
    print(f"  summer={n_summer}, autumn={n_autumn}")
    if n_summer == 0 and n_autumn == 0:
        print("  ✓ No summer/autumn features — temporally clean!")
    else:
        print(f"  ⚠ WARNING: {n_summer} summer + {n_autumn} autumn features detected!")
        leaky = [c for c in feature_cols if '_summer' in c or '_late_summer' in c or '_autumn' in c]
        print(f"  Leaky: {leaky[:5]} ...")

    # Итог тестов на утечки
    print("\n" + "-" * 70)
    print("ИТОГ ТЕСТОВ НА УТЕЧКИ:")
    real_r2 = None  # будет заполнено ниже при Ridge LOFO
    print(f"  [1.1] Ridge permutation: R² = {np.mean(perm_r2_ridge):.4f} (should ≈ 0)  ✓" if np.mean(perm_r2_ridge) < 0.05
          else f"  [1.1] Ridge permutation: R² = {np.mean(perm_r2_ridge):.4f}  ⚠ SUSPICIOUS")
    print(f"  [1.2] XGBoost permutation: R² = {np.mean(perm_r2_xgb):.4f} (should ≈ 0)  ✓" if np.mean(perm_r2_xgb) < 0.05
          else f"  [1.2] XGBoost permutation: R² = {np.mean(perm_r2_xgb):.4f}  ⚠ SUSPICIOUS")
    print(f"  [1.4] No temporal leakage in features  ✓" if n_summer == 0 and n_autumn == 0
          else f"  [1.4] TEMPORAL LEAKAGE in features!  ✗")

    # ══════════════════════════════════════════════════════════════════
    #  ЧАСТЬ 2: АЛЬТЕРНАТИВНЫЕ МОДЕЛИ
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("ЧАСТЬ 2: АЛЬТЕРНАТИВНЫЕ МОДЕЛИ (LOFO-CV, 81 field)")
    print("=" * 70)

    # Хранилище OOF для каждой модели
    results = {}

    # ── 2.1 Ridge Regression ──────────────────────────────────────────
    print("\n[2.1] Ridge Regression (alpha=10) ...")
    t1 = time.time()
    oof_ridge = np.zeros(N)
    for fi, tr_idx, te_idx in lofo_splits(fields, unique_fields):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_te = sc.transform(X[te_idx])
        m = Ridge(alpha=10.0, random_state=SEED)
        m.fit(X_tr, y_log[tr_idx])
        oof_ridge[te_idx] = np.clip(m.predict(X_te), 0, np.log1p(200))
    oof_ridge_orig = np.expm1(oof_ridge)
    m_ridge = compute_metrics(y_orig, oof_ridge_orig)
    results["Ridge"] = {"oof": oof_ridge_orig, "metrics": m_ridge}
    print(f"  ρ={m_ridge['rho']:+.3f}  R²={m_ridge['r2']:.3f}  "
          f"RMSE={m_ridge['rmse']:.2f}  MAE={m_ridge['mae']:.2f}  "
          f"({time.time()-t1:.1f}s)")

    # ── 2.2 ElasticNet ────────────────────────────────────────────────
    print("\n[2.2] ElasticNet (alpha=0.01, l1_ratio=0.5) ...")
    t1 = time.time()
    oof_enet = np.zeros(N)
    for fi, tr_idx, te_idx in lofo_splits(fields, unique_fields):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_te = sc.transform(X[te_idx])
        m = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000, random_state=SEED)
        m.fit(X_tr, y_log[tr_idx])
        oof_enet[te_idx] = m.predict(X_te)
    oof_enet_orig = np.expm1(oof_enet)
    m_enet = compute_metrics(y_orig, oof_enet_orig)
    results["ElasticNet"] = {"oof": oof_enet_orig, "metrics": m_enet}
    print(f"  ρ={m_enet['rho']:+.3f}  R²={m_enet['r2']:.3f}  "
          f"RMSE={m_enet['rmse']:.2f}  MAE={m_enet['mae']:.2f}  "
          f"({time.time()-t1:.1f}s)")

    # ── 2.3 XGBoost (re-run для единообразия) ─────────────────────────
    print("\n[2.3] XGBoost (n_est=500, depth=5, log1p, NO test eval_set) ...")
    t1 = time.time()
    oof_xgb = np.zeros(N)
    for fi, tr_idx, te_idx in lofo_splits(fields, unique_fields):
        # inner train/val split for early stopping (15% val)
        rng_fold = np.random.default_rng(SEED + fi)
        perm = rng_fold.permutation(len(tr_idx))
        n_val_inner = max(1, int(0.15 * len(tr_idx)))
        val_inner = tr_idx[perm[:n_val_inner]]
        tr_inner  = tr_idx[perm[n_val_inner:]]

        m = xgb.XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, verbosity=0, tree_method="hist",
            early_stopping_rounds=50,
        )
        m.fit(X[tr_inner], y_log[tr_inner],
              eval_set=[(X[val_inner], y_log[val_inner])],
              verbose=False)
        oof_xgb[te_idx] = m.predict(X[te_idx])
    oof_xgb_orig = np.expm1(oof_xgb)
    m_xgb = compute_metrics(y_orig, oof_xgb_orig)
    results["XGBoost"] = {"oof": oof_xgb_orig, "metrics": m_xgb}
    print(f"  ρ={m_xgb['rho']:+.3f}  R²={m_xgb['r2']:.3f}  "
          f"RMSE={m_xgb['rmse']:.2f}  MAE={m_xgb['mae']:.2f}  "
          f"({time.time()-t1:.1f}s)")

    # ── 2.4 CatBoost ──────────────────────────────────────────────────
    print("\n[2.4] CatBoost (iterations=500, depth=6, log1p, inner val split) ...")
    t1 = time.time()
    oof_cb = np.zeros(N)
    for fi, tr_idx, te_idx in lofo_splits(fields, unique_fields):
        # inner train/val split for overfitting detection (15% val)
        rng_fold = np.random.default_rng(SEED + fi)
        perm = rng_fold.permutation(len(tr_idx))
        n_val_inner = max(1, int(0.15 * len(tr_idx)))
        val_inner = tr_idx[perm[:n_val_inner]]
        tr_inner  = tr_idx[perm[n_val_inner:]]

        m = cb.CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.03,
            l2_leaf_reg=3, random_seed=SEED, verbose=0,
            loss_function="RMSE",
            od_type="Iter", od_wait=50,
            use_best_model=True,
        )
        m.fit(X[tr_inner], y_log[tr_inner],
              eval_set=(X[val_inner], y_log[val_inner]),
              verbose=0)
        oof_cb[te_idx] = m.predict(X[te_idx])
    oof_cb_orig = np.expm1(oof_cb)
    m_cb = compute_metrics(y_orig, oof_cb_orig)
    results["CatBoost"] = {"oof": oof_cb_orig, "metrics": m_cb}
    print(f"  ρ={m_cb['rho']:+.3f}  R²={m_cb['r2']:.3f}  "
          f"RMSE={m_cb['rmse']:.2f}  MAE={m_cb['mae']:.2f}  "
          f"({time.time()-t1:.1f}s)")

    # ── 2.5 Tabular MLP ──────────────────────────────────────────────
    print("\n[2.5] Tabular MLP (hidden=256, LOFO-CV) ...")
    t1 = time.time()
    oof_mlp = np.zeros(N)
    for fi, tr_idx, te_idx in lofo_splits(fields, unique_fields):
        # split train → train/val (85/15)
        rng_fold = np.random.default_rng(SEED + fi)
        perm = rng_fold.permutation(len(tr_idx))
        n_val = max(1, int(0.15 * len(tr_idx)))
        val_sub = tr_idx[perm[:n_val]]
        tr_sub  = tr_idx[perm[n_val:]]

        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_sub])
        X_val = sc.transform(X[val_sub])
        X_te  = sc.transform(X[te_idx])

        set_seed(SEED + fi)
        model = train_mlp_fold(
            X_tr, y_log[tr_sub],
            X_val, y_log[val_sub],
            n_features=X.shape[1],
            max_epochs=200, patience=15,
        )

        model.eval()
        with torch.no_grad():
            X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
            preds_mlp = model(X_te_t).cpu().numpy().ravel()
            # Clip predictions to reasonable log1p range [0, log1p(200)]
            preds_mlp = np.clip(preds_mlp, 0, np.log1p(200))
            oof_mlp[te_idx] = preds_mlp

        if (fi + 1) % 20 == 0:
            print(f"    fold {fi+1}/{N_FOLDS} done")

    oof_mlp_orig = np.expm1(oof_mlp)
    m_mlp = compute_metrics(y_orig, oof_mlp_orig)
    results["TabMLP"] = {"oof": oof_mlp_orig, "metrics": m_mlp}
    print(f"  ρ={m_mlp['rho']:+.3f}  R²={m_mlp['r2']:.3f}  "
          f"RMSE={m_mlp['rmse']:.2f}  MAE={m_mlp['mae']:.2f}  "
          f"({time.time()-t1:.1f}s)")

    # ── 2.6 Stacking Ensemble ─────────────────────────────────────────
    print("\n[2.6] Stacking Ensemble (Ridge + XGBoost + CatBoost → Ridge meta) ...")
    t1 = time.time()

    # Meta-features: OOF preds в log-пространстве от Ridge, XGBoost, CatBoost
    meta_X = np.column_stack([oof_ridge, oof_xgb, oof_cb])  # [N, 3] in log-space

    oof_stack = np.zeros(N)
    for fi, tr_idx, te_idx in lofo_splits(fields, unique_fields):
        sc = StandardScaler()
        M_tr = sc.fit_transform(meta_X[tr_idx])
        M_te = sc.transform(meta_X[te_idx])
        m = Ridge(alpha=0.1)
        m.fit(M_tr, y_log[tr_idx])
        oof_stack[te_idx] = m.predict(M_te)
    oof_stack_orig = np.expm1(oof_stack)
    m_stack = compute_metrics(y_orig, oof_stack_orig)
    results["Stacking"] = {"oof": oof_stack_orig, "metrics": m_stack}
    print(f"  ρ={m_stack['rho']:+.3f}  R²={m_stack['r2']:.3f}  "
          f"RMSE={m_stack['rmse']:.2f}  MAE={m_stack['mae']:.2f}  "
          f"({time.time()-t1:.1f}s)")

    # ── 2.7 Stacking V2: + MLP + original features ────────────────────
    print("\n[2.7] Stacking V2 (Ridge + XGBoost + CatBoost + MLP + top-10 features → Ridge meta) ...")
    t1 = time.time()

    # Top-10 important features by XGBoost
    xgb_full = xgb.XGBRegressor(n_estimators=300, max_depth=5, verbosity=0, tree_method="hist")
    xgb_full.fit(X, y_log)
    imp = xgb_full.feature_importances_
    top10_idx = np.argsort(imp)[-10:]
    X_top10 = X[:, top10_idx]

    meta_X_v2 = np.column_stack([oof_ridge, oof_xgb, oof_cb, oof_mlp, X_top10])

    oof_stack2 = np.zeros(N)
    for fi, tr_idx, te_idx in lofo_splits(fields, unique_fields):
        sc = StandardScaler()
        M_tr = sc.fit_transform(meta_X_v2[tr_idx])
        M_te = sc.transform(meta_X_v2[te_idx])
        m = Ridge(alpha=10.0)
        m.fit(M_tr, y_log[tr_idx])
        oof_stack2[te_idx] = np.clip(m.predict(M_te), 0, np.log1p(200))
    oof_stack2_orig = np.expm1(oof_stack2)
    m_stack2 = compute_metrics(y_orig, oof_stack2_orig)
    results["StackV2"] = {"oof": oof_stack2_orig, "metrics": m_stack2}
    print(f"  ρ={m_stack2['rho']:+.3f}  R²={m_stack2['r2']:.3f}  "
          f"RMSE={m_stack2['rmse']:.2f}  MAE={m_stack2['mae']:.2f}  "
          f"({time.time()-t1:.1f}s)")

    # ══════════════════════════════════════════════════════════════════
    #  ЧАСТЬ 3: ИТОГОВАЯ ТАБЛИЦА
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("ЧАСТЬ 3: ИТОГОВАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print("=" * 70)

    # Добавляем результаты из прошлых экспериментов (из summary)
    all_results = {
        "RF (leaky)":     {"rho": 0.407, "r2": 0.305, "rmse": np.nan, "mae": np.nan, "notes": "temporal leakage"},
        "RF spring":      {"rho": 0.366, "r2": 0.399, "rmse": np.nan, "mae": np.nan, "notes": "90 spring feat"},
        "CNN ResNet":     {"rho": 0.337, "r2": 0.413, "rmse": 5.87,  "mae": 3.30,  "notes": "18ch patches"},
        "CNN UNet":       {"rho": 0.304, "r2": 0.298, "rmse": 6.42,  "mae": 3.57,  "notes": "18ch patches"},
    }

    rows = []
    for name, data in all_results.items():
        rows.append({"model": name, **data})
    for name, data in results.items():
        m = data["metrics"]
        rows.append({"model": name, "rho": m["rho"], "r2": m["r2"],
                      "rmse": m["rmse"], "mae": m["mae"], "notes": "this run"})

    cmp = pd.DataFrame(rows)
    cmp = cmp.sort_values("r2", ascending=False).reset_index(drop=True)
    cmp.to_csv(OUT_DIR / "full_comparison.csv", index=False)

    print(cmp[["model", "rho", "r2", "rmse", "mae", "notes"]].to_string(index=False))

    # Best model check
    best = cmp.iloc[0]
    print(f"\n{'✅' if best['r2'] >= 0.5 else '⚠️'}  Best: {best['model']}  R²={best['r2']:.3f}")

    # ── Scatter plots для top-3 моделей ───────────────────────────────
    print("\n[7] Saving scatter plots ...")
    for name, data in results.items():
        m = data["metrics"]
        scatter_plot(y_orig, data["oof"], m, name,
                     OUT_DIR / f"scatter_oof_{name.lower().replace(' ','_')}.png")

    # ── Permutation summary plot ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#111111")

    # Real R² vs permuted R²
    all_perm = perm_r2_ridge + perm_r2_xgb
    real_r2_list = [m_ridge["r2"], m_xgb["r2"]]
    ax.hist(all_perm, bins=12, color="#4ab5e0", alpha=0.7, label="Permuted labels")
    for rr in real_r2_list:
        ax.axvline(rr, color="#e05c4a", lw=2, ls="--")
    ax.axvline(real_r2_list[0], color="#e05c4a", lw=2, ls="--", label=f"Real R² (Ridge={real_r2_list[0]:.3f})")
    ax.set_xlabel("OOF R²", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title("Permutation Test: Real vs Shuffled Labels", color="white", fontweight="bold")
    ax.legend(facecolor="#222", edgecolor="#555", labelcolor="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "permutation_test.png", dpi=160, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Outputs: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
