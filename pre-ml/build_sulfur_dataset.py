"""
pre-ml/build_sulfur_dataset.py
==============================
Формирует датасет для предсказания серы (S) с обогащёнными признаками:

Целевая переменная:
  - s (подвижная сера, мг/кг) — все 1085 образцов

Признаки (Features):
  1. OOF-предсказания pH, SOC, NO3 от лучших моделей (RF/GBDT)
       → кросс-валидированные псевдо-метки, не содержат утечки
  2. Временны́е ряды спутниковой динамики (100 колонок)
       → ts_l8/s2/spectral × 25 индексов × {mean, std, slope, cv}
  3. Многосезонные спектральные признаки (154 колонки: spring/summer/autumn/late_summer)
  4. Топографические признаки (8 колонок: DEM, slope, aspect, curvature, ...)
  5. Климатические признаки (4 колонки: MAP, MAT, seasonality)

Выход:
  data/preml/sulfur_enriched_dataset.csv
  data/preml/feature_groups.json  — словарь групп признаков для интерпретации
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Пути ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_CSV = ROOT / "data" / "features" / "master_dataset.csv"
OOF_DIR  = ROOT / "ML" / "results"
OUT_DIR  = ROOT / "data" / "preml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Загрузка мастер-датасета ───────────────────────────────────────────────────
print("Loading master dataset ...")
df = pd.read_csv(DATA_CSV, low_memory=False)
print(f"  {len(df)} rows × {df.shape[1]} columns")
print(f"  S valid: {df['s'].notna().sum()} rows")

# ── Загрузка OOF-предсказаний (pH, SOC, NO3) ──────────────────────────────────
# OOF предсказания — это out-of-fold предсказания, полученные при field-LOFO CV.
# Каждый образец предсказывался моделью, обученной БЕЗ его участия → нет утечки.
oof_preds = {}
oof_sources = {
    "ph":  ("baselines/GBDT", "ph"),   # GBDT — лучший для pH
    "soc": ("baselines/ET",   "soc"),  # ET   — лучший для SOC
    "no3": ("rf",             "no3"),  # RF   — лучший для NO3
}

for tgt, (folder, tgt_key) in oof_sources.items():
    path = OOF_DIR / folder / f"{tgt_key}_oof_predictions.csv"
    if path.exists():
        tmp = pd.read_csv(path, usecols=["oof_pred"], low_memory=False)
        # Align by index (same order as master dataset)
        if len(tmp) == len(df):
            oof_preds[f"oof_{tgt}"] = tmp["oof_pred"].values
            print(f"  OOF {tgt}: {path.name} loaded ({len(tmp)} rows)")
        else:
            print(f"  WARNING: OOF {tgt} length mismatch ({len(tmp)} vs {len(df)}), skipping")
    else:
        print(f"  WARNING: OOF file not found: {path}")

# ── Определение групп признаков ────────────────────────────────────────────────
all_cols = df.columns.tolist()

ts_cols      = sorted([c for c in all_cols if "ts_" in c.lower()])
spectral_cols = sorted([c for c in all_cols
                        if "spectral_" in c.lower()
                        and c not in ["spectral_B2_spring"]])  # не утечка для S
topo_cols    = sorted([c for c in all_cols if "topo_" in c.lower()])
climate_cols = sorted([c for c in all_cols if "climate_" in c.lower()])

# Включаем только весенние спектральные признаки (нет temporal leakage)
spring_spectral = sorted([c for c in spectral_cols if "_spring" in c])

print(f"\nFeature groups:")
print(f"  TS dynamics:       {len(ts_cols)}")
print(f"  Spectral spring:   {len(spring_spectral)}")
print(f"  Spectral all:      {len(spectral_cols)}")
print(f"  Topography:        {len(topo_cols)}")
print(f"  Climate:           {len(climate_cols)}")
print(f"  OOF predictions:   {len(oof_preds)}")

# ── Сборка финального датасета ─────────────────────────────────────────────────
meta_cols = ["id", "year", "farm", "field_name", "centroid_lon", "centroid_lat"]
meta_cols = [c for c in meta_cols if c in all_cols]

# Версия 1: без temporal leakage (только spring + ts + topo + climate + OOF)
safe_features = spring_spectral + ts_cols + topo_cols + climate_cols

# Версия 2: все признаки (для baseline сравнения, содержит temporal leakage для S)
all_features = spectral_cols + ts_cols + topo_cols + climate_cols

print(f"\nSafe features (no temporal leakage): {len(safe_features)}")
print(f"All features (baseline, with leakage): {len(all_features)}")

# Строим итоговый DataFrame
result = df[meta_cols + ["s"]].copy()
result = result.rename(columns={"s": "target_S"})

# Добавляем OOF-признаки
for col_name, values in oof_preds.items():
    result[col_name] = values

# Добавляем safe-признаки (основной набор)
for c in safe_features:
    if c in df.columns:
        result[c] = df[c].values

result = result.dropna(subset=["target_S"])
print(f"\nFinal dataset: {result.shape} | S valid: {result['target_S'].notna().sum()}")

# ── Сохранение ─────────────────────────────────────────────────────────────────
out_path = OUT_DIR / "sulfur_enriched_dataset.csv"
result.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")

# Сохраняем словарь групп
feature_groups = {
    "meta":             meta_cols,
    "target":           ["target_S"],
    "oof_predictions":  list(oof_preds.keys()),
    "ts_dynamics":      ts_cols,
    "spectral_spring":  spring_spectral,
    "topography":       topo_cols,
    "climate":          climate_cols,
    "safe_features":    safe_features,   # без temporal leakage
    "all_features":     all_features,    # включает все сезоны
}
groups_path = OUT_DIR / "feature_groups.json"
with open(groups_path, "w") as f:
    json.dump(feature_groups, f, indent=2)
print(f"Saved → {groups_path}")

# ── Краткая статистика по целевой переменной ───────────────────────────────────
s = result["target_S"]
print(f"\nS statistics:")
print(f"  n={len(s)}  mean={s.mean():.2f}  median={s.median():.2f}")
print(f"  std={s.std():.2f}  min={s.min():.2f}  max={s.max():.2f}")
print(f"  CV={s.std()/s.mean()*100:.1f}%  skewness={s.skew():.2f}")
print(f"  Farms: {result['farm'].nunique()}")
