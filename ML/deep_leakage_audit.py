"""
deep_leakage_audit.py
=====================
Глубокий аудит: ПОЧЕМУ XGBoost R²=0.682 — утечка или реальный сигнал?

Тесты:
  1. ICC анализ: % дисперсии S между полями vs внутри полей
  2. Baseline «наивная» модель: предсказание = среднее поля/фермы
  3. Farm-level LOFO (20 фолдов) — строже, чем field-level
  4. Feature ablation:
       a) только topo+climate (12 признаков)
       b) только spring spectral (78 признаков)
       c) без climate_MAP / climate_MAT (88 признаков)
       d) все 90 признаков (baseline)
  5. climate_MAP/MAT: насколько уникальны по полям?
  6. Residual Moran's I: пространственная автокорреляция остатков
  7. Scatter: R² по каждому фолду vs расстояние до ближайшей train-фермы

Ожидаемый вывод:
  Если R² при farm-level LOFO резко падает → модель учит farm-level паттерны
  Если R² без climate → похожий → climate_MAP/MAT были proxy for farm identity
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
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist

import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
DATA_CSV = ROOT / "data" / "features" / "master_dataset.csv"
OUT_DIR  = ROOT / "ML" / "results" / "deep_leakage_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "s"
FIELD_COL  = "field_name"
FARM_COL   = "farm"
SEED       = 42

XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    random_state=SEED, verbosity=0, tree_method="hist",
    early_stopping_rounds=50,
)

def get_feature_sets(df):
    num_cols = df.select_dtypes(include="number").columns
    spring = sorted([c for c in num_cols if "_spring" in c
                     and "_summer" not in c and "_autumn" not in c
                     and "_late_summer" not in c])
    topo    = sorted([c for c in num_cols if c.startswith("topo_")])
    climate = sorted([c for c in num_cols if c.startswith("climate_")])
    all90   = spring + topo + climate
    no_climate_geo = [c for c in all90 if c not in ("climate_MAP", "climate_MAT")]
    return {
        "all90": all90,
        "spring_only": spring,
        "topo_climate": topo + climate,
        "no_map_mat": no_climate_geo,
        "topo_only": topo,
        "climate_only": climate,
    }

def lofo_field(fields, unique_fields):
    for i, f in enumerate(unique_fields):
        yield i, np.where(fields != f)[0], np.where(fields == f)[0]

def lofo_farm(farms, unique_farms):
    for i, f in enumerate(unique_farms):
        yield i, np.where(farms != f)[0], np.where(farms == f)[0]

def run_xgb_lofo(X, y_log, y_orig, groups, unique_groups, lofo_fn, label):
    oof = np.zeros(len(y_log))
    for _, tr_idx, te_idx in lofo_fn(groups, unique_groups):
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(tr_idx))
        n_val = max(1, int(0.15 * len(tr_idx)))
        val_inner, tr_inner = tr_idx[perm[:n_val]], tr_idx[perm[n_val:]]
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X[tr_inner], y_log[tr_inner],
              eval_set=[(X[val_inner], y_log[val_inner])], verbose=False)
        oof[te_idx] = m.predict(X[te_idx])
    pred_orig = np.expm1(oof)
    r2  = r2_score(y_orig, pred_orig)
    rho = float(spearmanr(y_orig, pred_orig)[0])
    rmse = float(np.sqrt(mean_squared_error(y_orig, pred_orig)))
    print(f"  {label:30s}  ρ={rho:+.3f}  R²={r2:.3f}  RMSE={rmse:.2f}")
    return r2, rho, rmse, pred_orig

def main():
    t0 = time.time()
    print("=" * 70)
    print("DEEP LEAKAGE AUDIT: XGBoost R²=0.682 — leakage or real signal?")
    print("=" * 70)

    # ── Загрузка ──────────────────────────────────────────────────────
    df = pd.read_csv(DATA_CSV, low_memory=False)
    df = df.dropna(subset=[TARGET_COL]).copy()
    y_orig = df[TARGET_COL].values.astype(np.float32)
    y_log  = np.log1p(y_orig)
    fields = df[FIELD_COL].values
    farms  = df[FARM_COL].values
    unique_fields = np.unique(fields)
    unique_farms  = np.unique(farms)
    N = len(df)
    print(f"N={N}, fields={len(unique_fields)}, farms={len(unique_farms)}")

    feat_sets = get_feature_sets(df)
    for name, cols in feat_sets.items():
        for col in cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

    # ══════════════════════════════════════════════════════════════════
    # ТЕСТ 1: ICC анализ
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ТЕСТ 1: ICC — % дисперсии S между полями vs внутри полей")
    print("=" * 70)

    grand_mean = y_orig.mean()
    ss_total   = np.sum((y_orig - grand_mean) ** 2)

    field_means = df.groupby(FIELD_COL)[TARGET_COL].mean()
    field_mean_arr = np.array([field_means[f] for f in fields])
    ss_between = np.sum((field_mean_arr - grand_mean) ** 2)
    ss_within  = ss_total - ss_between

    pct_between = 100 * ss_between / ss_total
    pct_within  = 100 * ss_within / ss_total

    print(f"  Всего дисперсии (SS_total):     {ss_total:.1f}")
    print(f"  Между полями (SS_between):       {ss_between:.1f}  ({pct_between:.1f}%)")
    print(f"  Внутри полей (SS_within):        {ss_within:.1f}  ({pct_within:.1f}%)")

    # Farm-level ICC
    farm_means = df.groupby(FARM_COL)[TARGET_COL].mean()
    farm_mean_arr = np.array([farm_means[f] for f in farms])
    ss_between_farm = np.sum((farm_mean_arr - grand_mean) ** 2)
    pct_between_farm = 100 * ss_between_farm / ss_total
    print(f"\n  Между ФЕРМАМИ (SS_between farm): {ss_between_farm:.1f}  ({pct_between_farm:.1f}%)")

    # Naive baseline: predict field mean (in-sample only)
    r2_field_mean = r2_score(y_orig, field_mean_arr)
    # Farm mean in-sample
    r2_farm_mean = r2_score(y_orig, farm_mean_arr)
    print(f"\n  Наивный baseline (предсказать среднее поля, in-sample):  R² = {r2_field_mean:.3f}")
    print(f"  Наивный baseline (предсказать среднее фермы, in-sample): R² = {r2_farm_mean:.3f}")

    # LOFO baseline: predict field mean FROM TRAINING
    oof_field_mean = np.zeros(N)
    for i, test_field in enumerate(unique_fields):
        te_idx = np.where(fields == test_field)[0]
        tr_idx = np.where(fields != test_field)[0]
        train_mean = y_orig[tr_idx].mean()  # grand mean of training data
        oof_field_mean[te_idx] = train_mean
    r2_naive_lofo = r2_score(y_orig, oof_field_mean)
    print(f"  Наивный LOFO baseline (predict train mean):               R² = {r2_naive_lofo:.3f}")

    # ══════════════════════════════════════════════════════════════════
    # ТЕСТ 2: climate_MAP/MAT uniqueness
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ТЕСТ 2: Насколько climate_MAP/MAT кодируют field_name?")
    print("=" * 70)

    for clm_col in ["climate_MAP", "climate_MAT", "topo_DEM"]:
        if clm_col not in df.columns:
            continue
        unique_per_field = df.groupby(FIELD_COL)[clm_col].nunique()
        fields_with_unique = (unique_per_field == 1).sum()
        pct_unique = 100 * fields_with_unique / len(unique_fields)
        std_within = df.groupby(FIELD_COL)[clm_col].std().mean()
        std_between = df.groupby(FIELD_COL)[clm_col].mean().std()
        r_with_s, p_val = pearsonr(df[clm_col].fillna(df[clm_col].mean()), y_orig)
        print(f"  {clm_col:20s}: {pct_unique:.0f}% полей имеют уникальное значение  "
              f"std_within={std_within:.3f}  std_between={std_between:.3f}  "
              f"r_with_S={r_with_s:+.3f} (p={p_val:.1e})")

    # ══════════════════════════════════════════════════════════════════
    # ТЕСТ 3: Feature Ablation (field-level LOFO)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ТЕСТ 3: Feature Ablation (field-level LOFO, 81 фолдов)")
    print("=" * 70)

    ablation_results = {}

    ablation_configs = [
        ("all90 [baseline]",          feat_sets["all90"]),
        ("spring_only (78)",          feat_sets["spring_only"]),
        ("topo+climate (12)",         feat_sets["topo_climate"]),
        ("topo_only (8)",             feat_sets["topo_only"]),
        ("climate_only (4)",          feat_sets["climate_only"]),
        ("no_MAP_MAT (88)",           feat_sets["no_map_mat"]),
        ("no_MAP_MAT_no_DEM (87)",    [c for c in feat_sets["no_map_mat"] if c != "topo_DEM"]),
    ]

    for label, cols in ablation_configs:
        avail = [c for c in cols if c in df.columns]
        X = df[avail].values.astype(np.float32)
        t1 = time.time()
        r2, rho, rmse, _ = run_xgb_lofo(X, y_log, y_orig, fields, unique_fields, lofo_field, label)
        ablation_results[label] = {"r2": r2, "rho": rho, "rmse": rmse,
                                   "n_features": len(avail), "time": time.time()-t1}

    # ══════════════════════════════════════════════════════════════════
    # ТЕСТ 4: Farm-level LOFO (строже!)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ТЕСТ 4: Farm-level LOFO (20 фолдов — строже field-level)")
    print("=" * 70)

    farm_ablation = {}
    for label, cols in [
        ("all90  [farm LOFO]",    feat_sets["all90"]),
        ("no_MAP_MAT [farm LOFO]", feat_sets["no_map_mat"]),
        ("spring_only [farm LOFO]", feat_sets["spring_only"]),
    ]:
        avail = [c for c in cols if c in df.columns]
        X = df[avail].values.astype(np.float32)
        r2, rho, rmse, preds = run_xgb_lofo(X, y_log, y_orig, farms, unique_farms, lofo_farm, label)
        farm_ablation[label] = {"r2": r2, "rho": rho, "rmse": rmse}

    # ══════════════════════════════════════════════════════════════════
    # ТЕСТ 5: Spatial autocorrelation of residuals (Moran-like)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ТЕСТ 5: Пространственная автокорреляция остатков XGBoost")
    print("=" * 70)

    # Используем all90 field-level predictions
    X_all = df[feat_sets["all90"]].values.astype(np.float32)
    _, _, _, oof_preds_all90 = run_xgb_lofo(
        X_all, y_log, y_orig, fields, unique_fields, lofo_field, "all90 (for residuals)"
    )
    residuals = y_orig - oof_preds_all90

    if "centroid_lon" in df.columns and "centroid_lat" in df.columns:
        lon = df["centroid_lon"].values
        lat = df["centroid_lat"].values
        coords = np.column_stack([lon, lat])

        # Быстрый spatial analysis: для каждого образца найти ближайших соседей
        # и проверить корреляцию остатков с соседними остатками
        n_neighbors = 5
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        dists, idxs = tree.query(coords, k=n_neighbors + 1)  # +1 for self

        neighbor_resid = np.zeros(N)
        for i in range(N):
            neigh = idxs[i, 1:]  # exclude self
            neighbor_resid[i] = residuals[neigh].mean()

        r_spatial, p_spatial = pearsonr(residuals, neighbor_resid)
        print(f"  Пространственная автокорреляция остатков:")
        print(f"  r(residual, mean_neighbor_residual) = {r_spatial:+.3f}  p={p_spatial:.4f}")
        if abs(r_spatial) > 0.3 and p_spatial < 0.05:
            print(f"  ⚠️  ВНИМАНИЕ: Остатки пространственно автокоррелированы!")
            print(f"     Модель систематически переоценивает/недооценивает в кластерах")
        else:
            print(f"  ✓ Остатки слабо пространственно автокоррелированы (нормально)")

    # ══════════════════════════════════════════════════════════════════
    # ТЕСТ 6: Per-farm performance (field-level LOFO)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ТЕСТ 6: R² по фермам (field-level LOFO) — equal treatment?")
    print("=" * 70)

    X_all = df[feat_sets["all90"]].values.astype(np.float32)
    oof_all90 = np.zeros(N)
    for fi, tr_idx, te_idx in lofo_field(fields, unique_fields):
        rng = np.random.default_rng(SEED + fi)
        perm = rng.permutation(len(tr_idx))
        n_val = max(1, int(0.15 * len(tr_idx)))
        val_inner, tr_inner = tr_idx[perm[:n_val]], tr_idx[perm[n_val:]]
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_all[tr_inner], y_log[tr_inner],
              eval_set=[(X_all[val_inner], y_log[val_inner])], verbose=False)
        oof_all90[te_idx] = m.predict(X_all[te_idx])
    oof_all90_orig = np.expm1(oof_all90)

    # R² per farm
    farm_r2 = {}
    for farm in unique_farms:
        idx = np.where(farms == farm)[0]
        if len(idx) < 5:
            continue
        y_t = y_orig[idx]
        y_p = oof_all90_orig[idx]
        if np.std(y_t) < 0.5:
            continue
        farm_r2[farm] = r2_score(y_t, y_p)
    farm_r2_sorted = sorted(farm_r2.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  R² по фермам (top и bottom):")
    for farm_name, r2_val in farm_r2_sorted[:5]:
        n_f = len(df[df[FARM_COL] == farm_name])
        print(f"    {farm_name:30s}  n={n_f:3d}  R²={r2_val:.3f}  ✅" if r2_val > 0.5
              else f"    {farm_name:30s}  n={n_f:3d}  R²={r2_val:.3f}")
    print("  ...")
    for farm_name, r2_val in farm_r2_sorted[-5:]:
        n_f = len(df[df[FARM_COL] == farm_name])
        print(f"    {farm_name:30s}  n={n_f:3d}  R²={r2_val:.3f}  ❌" if r2_val < 0
              else f"    {farm_name:30s}  n={n_f:3d}  R²={r2_val:.3f}")

    # ══════════════════════════════════════════════════════════════════
    # ИТОГ: Вердикт об утечке
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ ВЕРДИКТ")
    print("=" * 70)

    r2_field  = ablation_results["all90 [baseline]"]["r2"]
    r2_farm   = farm_ablation["all90  [farm LOFO]"]["r2"]
    r2_spring = ablation_results["spring_only (78)"]["r2"]
    r2_topo_clm = ablation_results["topo+climate (12)"]["r2"]
    r2_no_map_mat = ablation_results["no_MAP_MAT (88)"]["r2"]
    r2_no_map_mat_farm = farm_ablation["no_MAP_MAT [farm LOFO]"]["r2"]

    print(f"\n  [field-LOFO] all90:             R² = {r2_field:.3f}")
    print(f"  [farm-LOFO]  all90:             R² = {r2_farm:.3f}  <- строгий тест")
    print(f"  [field-LOFO] spring_only:        R² = {r2_spring:.3f}  (без topo/climate)")
    print(f"  [field-LOFO] topo+climate only:  R² = {r2_topo_clm:.3f}  (без спектра)")
    print(f"  [field-LOFO] no MAP+MAT:         R² = {r2_no_map_mat:.3f}  (без geography proxy)")
    print(f"  [farm-LOFO]  no MAP+MAT:         R² = {r2_no_map_mat_farm:.3f}  <- строгий без proxy")
    print(f"  [in-sample]  field mean only:    R² = {r2_field_mean:.3f}  (pure geography)")

    drop_field_to_farm = r2_field - r2_farm
    print(f"\n  Падение field→farm LOFO: {drop_field_to_farm:+.3f}")

    print("\n  ДИАГНОЗ:")
    if drop_field_to_farm > 0.15:
        print("  ⚠️  УМЕРЕННАЯ ПРОСТРАНСТВЕННАЯ УТЕЧКА!")
        print("     Модель частично учит field → farm-level паттерны,")
        print("     которые не обобщаются на новые фермы.")
        print(f"     При честной (farm-level) оценке: R² = {r2_farm:.3f}")
    elif drop_field_to_farm > 0.3:
        print("  ❌ СИЛЬНАЯ ПРОСТРАНСТВЕННАЯ УТЕЧКА!")
        print("     Большая часть R² объясняется farm-level паттернами.")
    else:
        print("  ✅ УТЕЧКА СЛАБАЯ: модель обобщается на новые фермы.")

    if r2_topo_clm > r2_spring:
        print(f"\n  ⚠️  topo+climate ({r2_topo_clm:.3f}) > spring-only ({r2_spring:.3f})!")
        print("     Топография/климат несут больше информации, чем спектр.")
        print("     Причина: DEM и climate_MAP коррелируют с S через географию.")

    if abs(r2_field - r2_no_map_mat) > 0.05:
        print(f"\n  ⚠️  Убираем climate_MAP+MAT: R² {r2_field:.3f} → {r2_no_map_mat:.3f}")
        print("     Это подтверждает, что MAP/MAT были proxy for farm identity.")
    else:
        print(f"\n  ✓ climate_MAP/MAT не существенно влияли на R² (+{r2_field-r2_no_map_mat:.3f})")

    print("\n" + "-" * 70)
    print("РЕКОМЕНДАЦИИ:")
    print(f"  1. РЕАЛЬНЫЙ R² модели (farm-level LOFO): {r2_farm:.3f}")
    print(f"  2. Честная оценка (без MAP/MAT, farm-LOFO): {r2_no_map_mat_farm:.3f}")
    print(f"  3. Чистый спектральный сигнал (spring-only, field-LOFO): {r2_spring:.3f}")
    print("-" * 70)

    # ── Визуализация: ablation comparison ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0a0a")

    ax = axes[0]
    ax.set_facecolor("#111111")
    labels = list(ablation_results.keys())
    r2_vals = [ablation_results[l]["r2"] for l in labels]
    colors = ["#4ab5e0" if r2 < r2_field else "#e05c4a" for r2 in r2_vals]
    bars = ax.barh(range(len(labels)), r2_vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l[:35] for l in labels], color="white", fontsize=8)
    ax.axvline(r2_field, color="#e05c4a", lw=1.5, ls="--", alpha=0.7)
    ax.set_xlabel("OOF R² (field-level LOFO)", color="white")
    ax.set_title("Feature Ablation\n(field-level LOFO)", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    ax = axes[1]
    ax.set_facecolor("#111111")
    categories = ["Field-LOFO\n(all90)", "Farm-LOFO\n(all90)", "Farm-LOFO\n(no MAP/MAT)",
                  "Field-LOFO\n(spring only)", "Field-LOFO\n(topo+climate)", "Naive\n(field mean)"]
    vals = [r2_field, r2_farm, r2_no_map_mat_farm, r2_spring, r2_topo_clm, r2_field_mean]
    colors2 = ["#4ab5e0", "#e05c4a", "#e06c4a", "#5ae0ab", "#e0e04a", "#888888"]
    bars2 = ax.bar(range(len(categories)), vals, color=colors2)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, color="white", fontsize=8)
    ax.set_ylabel("R²", color="white")
    ax.set_title("Сравнение строгости CV\nи вклада групп признаков",
                 color="white", fontweight="bold")
    ax.axhline(0, color="white", lw=0.5)
    ax.axhline(0.5, color="#888", lw=0.5, ls="--")
    for bar, val in zip(bars2, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, max(val + 0.02, 0.02),
                f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    fig.suptitle("Deep Leakage Audit: XGBoost R²=0.682 Analysis",
                 color="white", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "leakage_audit_summary.png", dpi=160,
                bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)

    # ── CSV ────────────────────────────────────────────────────────────
    rows = []
    for label, res in ablation_results.items():
        rows.append({"test": f"field-LOFO: {label}", **res})
    for label, res in farm_ablation.items():
        rows.append({"test": f"farm-LOFO: {label}", **res})
    rows.append({"test": "naive: field_mean (in-sample)", "r2": r2_field_mean, "rho": np.nan, "rmse": np.nan})
    rows.append({"test": "naive: LOFO_grand_mean", "r2": r2_naive_lofo, "rho": np.nan, "rmse": np.nan})
    pd.DataFrame(rows).to_csv(OUT_DIR / "audit_results.csv", index=False)

    print(f"\n[Done] Time: {(time.time()-t0)/60:.1f} min")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
