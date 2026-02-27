"""
Delta-feature engineering: сезонные разности спектральных индексов.

Гипотеза:
  Накопление питательных элементов (P, K, S, N) в почве проявляется не в
  абсолютном значении индекса (NDVI, NDRE, ...) в конкретный момент, а в
  ДИНАМИКЕ вегетации — насколько быстро нарастает биомасса весной,
  каков пик, насколько рано наступает сенесценция.

  Дельты кодируют эту динамику напрямую:
    Δ_spring→summer     = INDEX_summer - INDEX_spring      (скорость нарастания)
    Δ_summer→late_summer = INDEX_late_summer - INDEX_summer (пик или начало спада)
    Δ_late_summer→autumn = INDEX_autumn - INDEX_late_summer (скорость сенесценции)
    Δ_spring→autumn     = INDEX_autumn - INDEX_spring      (суммарный сезонный сдвиг)
    RANGE               = max(4 сезона) - min(4 сезона)   (амплитуда за сезон)

  Для почв с дефицитом P/K/S кривая NDVI:
    - Нарастает медленнее (Δ_spring→summer ↓)
    - Достигает меньшего пика
    - Быстрее угасает осенью (Δ_late_summer→autumn <0 сильнее)

Что делает скрипт:
  1. Загружает full_dataset.csv
  2. Вычисляет Δ для всех 4 пар сезонов × все индексы (S2 + L8 + spectral)
  3. Вычисляет RANGE (амплитуду) для каждого индекса
  4. Считает корреляцию Спирмена Δ-признаков с ph, k, p, hu, s, no3
  5. Сохраняет:
     - delta_dataset.csv    — исходные строки + дельта-признаки (~120 новых колонок)
     - delta_correlations.csv — таблица ρ (Spearman) Δ-признак × хим. элемент
     - delta_heatmap.png    — heatmap корреляций: Δ-признаки × элементы
     - delta_top_scatter.png — scatter для топ-10 коррелирующих пар

Запуск: python approximated/build_delta_features.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─── Параметры ───────────────────────────────────────────────────
BASE      = Path(__file__).parent
DATA_PATH = BASE.parent / "data" / "features" / "full_dataset.csv"
OUT_DIR   = BASE.parent / "math_statistics" / "output" / "plots"
OUT_CSV   = BASE.parent / "data" / "features" / "delta_dataset.csv"
OUT_CORR  = OUT_DIR / "delta_correlations.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHEM_COLS = ["ph", "k", "p", "hu", "s", "no3"]
CHEM_LABELS = {
    "ph":  "pH",
    "k":   "K, mg/kg",
    "p":   "P, mg/kg",
    "hu":  "Humus, %",
    "s":   "S, mg/kg",
    "no3": "NO3, mg/kg",
}

# Сезоны в хронологическом порядке
SEASONS = ["spring", "summer", "late_summer", "autumn"]
SEASON_LABELS = {
    "spring":      "Spring (Apr–May)",
    "summer":      "Summer (Jun–Jul)",
    "late_summer": "Late Summer (Aug–Sep)",
    "autumn":      "Autumn (Oct)",
}

# Пары сезонов для дельт
DELTA_PAIRS = [
    ("spring",      "summer",      "Δ spr→sum"),
    ("summer",      "late_summer", "Δ sum→lsm"),
    ("late_summer", "autumn",      "Δ lsm→aut"),
    ("spring",      "autumn",      "Δ spr→aut"),   # суммарный сдвиг
]

# Источники и индексы для дельт
SOURCES_INDICES = [
    # (prefix_in_csv, index_name, display_name)
    ("s2",       "NDVI",        "S2·NDVI"),
    ("s2",       "NDRE",        "S2·NDRE"),
    ("s2",       "GNDVI",       "S2·GNDVI"),
    ("s2",       "EVI",         "S2·EVI"),
    ("s2",       "BSI",         "S2·BSI"),
    ("s2",       "SAVI",        "S2·SAVI"),
    ("s2",       "Cl_Red_Edge", "S2·CRE"),
    ("l8",       "NDVI",        "L8·NDVI"),
    ("l8",       "GNDVI",       "L8·GNDVI"),
    ("l8",       "SAVI",        "L8·SAVI"),
    ("spectral", "NDVI",        "Sp·NDVI"),
    ("spectral", "NDRE",        "Sp·NDRE"),
    ("spectral", "GNDVI",       "Sp·GNDVI"),
    ("spectral", "EVI",         "Sp·EVI"),
    ("spectral", "BSI",         "Sp·BSI"),
    ("spectral", "SAVI",        "Sp·SAVI"),
]

# ─── 1. Загрузка данных ───────────────────────────────────────────
print("Loading full_dataset.csv ...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Farms: {df['farm'].nunique()}  Fields: {df['field_name'].nunique()}  Years: {sorted(df['year'].unique())}")

# ─── 2. Вычисление дельт и RANGE ─────────────────────────────────
print("\nComputing delta features ...")

delta_cols_meta = []  # (col_name, display_name, source, index, pair_label)

for prefix, idx, disp in SOURCES_INDICES:
    # Колонки по сезонам
    season_cols = {}
    for s in SEASONS:
        cname = f"{prefix}_{idx}_{s}"
        if cname in df.columns:
            season_cols[s] = cname

    if len(season_cols) < 2:
        print(f"  Skip {prefix}_{idx}: only {len(season_cols)} seasons found")
        continue

    # Дельты
    for s1, s2, plabel in DELTA_PAIRS:
        if s1 in season_cols and s2 in season_cols:
            new_col = f"delta_{prefix}_{idx}_{s1}_to_{s2}"
            df[new_col] = df[season_cols[s2]] - df[season_cols[s1]]
            delta_cols_meta.append((new_col, f"{disp} {plabel}", prefix, idx, plabel))

    # RANGE = max - min по доступным сезонам
    avail = list(season_cols.values())
    range_col = f"range_{prefix}_{idx}"
    df[range_col] = df[avail].max(axis=1) - df[avail].min(axis=1)
    delta_cols_meta.append((range_col, f"{disp} RANGE", prefix, idx, "RANGE"))

delta_feature_names = [m[0] for m in delta_cols_meta]
print(f"  Generated {len(delta_feature_names)} delta/range features")

# ─── 3. Корреляции Спирмена: Δ × химия ───────────────────────────
print("\nComputing Spearman correlations (delta features × chemistry) ...")

corr_rows = []
for feat_col, feat_disp, src, idx, plabel in delta_cols_meta:
    row = {
        "feature":     feat_col,
        "display":     feat_disp,
        "source":      src,
        "index":       idx,
        "pair":        plabel,
    }
    for chem in CHEM_COLS:
        sub = df[[feat_col, chem]].dropna()
        if len(sub) < 20:
            row[f"rho_{chem}"]  = np.nan
            row[f"pval_{chem}"] = np.nan
            row[f"n_{chem}"]    = len(sub)
        else:
            r, p = spearmanr(sub[feat_col].values, sub[chem].values)
            row[f"rho_{chem}"]  = round(r, 4)
            row[f"pval_{chem}"] = round(p, 5)
            row[f"n_{chem}"]    = len(sub)
    corr_rows.append(row)

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(OUT_CORR, index=False)
print(f"  Saved correlations: {OUT_CORR.name}")

# ─── 4. Сохранение delta_dataset.csv ─────────────────────────────
print(f"\nSaving delta_dataset.csv ...")
df.to_csv(OUT_CSV, index=False, float_format="%.6f")
print(f"  Shape: {df.shape}  →  {OUT_CSV.name}")

# ─── 5. Топ-корреляции по каждому элементу ───────────────────────
print("\nTop-5 delta features per chemistry element:")
rho_matrix = corr_df.set_index("display")[[f"rho_{c}" for c in CHEM_COLS]]
rho_matrix.columns = [CHEM_LABELS[c] for c in CHEM_COLS]

for chem in CHEM_COLS:
    col = f"rho_{chem}"
    top = corr_df.nlargest(3, col, keep="all")[["display", col, f"n_{chem}"]]
    bot = corr_df.nsmallest(3, col, keep="all")[["display", col, f"n_{chem}"]]
    best = pd.concat([top, bot]).drop_duplicates()
    best = best.reindex(best[col].abs().sort_values(ascending=False).index).head(5)
    print(f"  {CHEM_LABELS[chem]:12s}:")
    for _, r in best.iterrows():
        print(f"    ρ={r[col]:+.3f}  {r['display']}  (n={int(r[f'n_{chem}'])})")

# ─── 6. Heatmap: Δ-признаки × химия ─────────────────────────────
print("\nRendering correlation heatmap ...")

# Выбираем топ-40 по |max ρ| по всем элементам
rho_cols = [f"rho_{c}" for c in CHEM_COLS]
corr_df["max_abs_rho"] = corr_df[rho_cols].abs().max(axis=1)
top_feats = corr_df.nlargest(40, "max_abs_rho")

heat = top_feats.set_index("display")[rho_cols].copy()
heat.columns = [CHEM_LABELS[c] for c in CHEM_COLS]

fig, ax = plt.subplots(figsize=(10, 14))
fig.patch.set_facecolor("#0a0a0a")
ax.set_facecolor("#111111")

vmax = 0.7
im = ax.imshow(heat.values, aspect="auto", cmap="RdBu_r",
               vmin=-vmax, vmax=vmax, interpolation="nearest")

# Аннотации
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        val = heat.values[i, j]
        if np.isnan(val):
            continue
        color = "black" if abs(val) > 0.35 else "white"
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                fontsize=7, color=color)

ax.set_xticks(range(len(heat.columns)))
ax.set_xticklabels(heat.columns, color="white", fontsize=9)
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels(heat.index, color="white", fontsize=7.5)
ax.tick_params(colors="white")

cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.6, aspect=30)
cb.set_label("Spearman ρ", color="white", fontsize=9)
cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)
cb.outline.set_edgecolor("#333")

ax.set_title(
    "Spearman ρ: seasonal delta features × soil chemistry\n"
    "(top-40 by max |ρ| across all elements)",
    color="white", fontsize=11, fontweight="bold", pad=10,
)
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")

fig.text(0.5, 0.005,
         f"n=1215 grid points · S2+L8+spectral indices · Δ=season2−season1 · RANGE=max−min",
         ha="center", color="#666", fontsize=7.5)
fig.tight_layout(rect=[0, 0.015, 1, 1])
out_hm = OUT_DIR / "delta_heatmap.png"
fig.savefig(out_hm, dpi=180, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig)
print(f"  Saved: {out_hm.name}")

# ─── 7. Top scatter plots ─────────────────────────────────────────
print("Rendering top scatter plots ...")

# Топ-2 delta-признака на каждый элемент (по |ρ|)
top_pairs = []
for chem in CHEM_COLS:
    col = f"rho_{chem}"
    best = corr_df.reindex(
        corr_df[col].abs().sort_values(ascending=False).index
    ).head(2)
    for _, row in best.iterrows():
        top_pairs.append((row["feature"], row["display"], chem, row[col]))

# Дедупликация
seen = set()
top_pairs_uniq = []
for feat, disp, chem, rho in top_pairs:
    key = (feat, chem)
    if key not in seen:
        seen.add(key)
        top_pairs_uniq.append((feat, disp, chem, rho))

n_panels = min(len(top_pairs_uniq), 12)
ncols = 4
nrows = (n_panels + ncols - 1) // ncols

fig2, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.0))
fig2.patch.set_facecolor("#0a0a0a")
axes_flat = axes.ravel() if nrows > 1 else [axes] if ncols == 1 else axes.ravel()

for i, (feat, disp, chem, rho) in enumerate(top_pairs_uniq[:n_panels]):
    ax = axes_flat[i]
    ax.set_facecolor("#111111")

    sub = df[[feat, chem]].dropna()
    x_vals = sub[feat].values
    y_vals = sub[chem].values

    # Color by chemistry value
    sc = ax.scatter(x_vals, y_vals, c=y_vals,
                    cmap="plasma", s=20, alpha=0.65, linewidths=0, zorder=3)

    # Trend line
    z = np.polyfit(x_vals, y_vals, 1)
    xr = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(xr, np.polyval(z, xr), color="#e05c4a", linewidth=1.5, zorder=4)

    ax.set_xlabel(disp, color="white", fontsize=8)
    ax.set_ylabel(CHEM_LABELS[chem], color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.text(0.05, 0.95, f"ρ = {rho:+.3f}\nn = {len(sub)}",
            transform=ax.transAxes, ha="left", va="top", color="white",
            fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#1a1a1a",
                      edgecolor="#555", alpha=0.9))

    cb2 = fig2.colorbar(sc, ax=ax, pad=0.02, shrink=0.80)
    cb2.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
    cb2.outline.set_edgecolor("#333")
    cb2.set_label(CHEM_LABELS[chem], color="white", fontsize=6)

# Скрываем лишние панели
for j in range(n_panels, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig2.suptitle(
    "Top delta-feature × soil chemistry scatter plots\n"
    "(Spearman ρ, n=1215 grid points, S2+L8+spectral seasonal deltas)",
    color="white", fontsize=11, fontweight="bold", y=1.01,
)
fig2.tight_layout(pad=0.5)
out_sc = OUT_DIR / "delta_top_scatter.png"
fig2.savefig(out_sc, dpi=180, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig2)
print(f"  Saved: {out_sc.name}")

# ─── 8. Сравнение: лучший абсолютный признак vs лучший delta ─────
print("\nRendering absolute vs delta comparison ...")

# Для каждого элемента: топ-1 абсолютный признак vs топ-1 дельта
abs_index_cols = [c for c in df.columns
                  if any(c.startswith(f"{p}_") for p in ["s2", "l8", "spectral"])
                  and any(c.endswith(f"_{s}") for s in SEASONS)
                  and c not in delta_feature_names]

abs_corr_rows = []
for feat_col in abs_index_cols:
    row = {"feature": feat_col}
    for chem in CHEM_COLS:
        sub = df[[feat_col, chem]].dropna()
        if len(sub) < 20:
            row[f"rho_{chem}"] = np.nan
        else:
            r, _ = spearmanr(sub[feat_col].values, sub[chem].values)
            row[f"rho_{chem}"] = round(r, 4)
    abs_corr_rows.append(row)

abs_corr_df = pd.DataFrame(abs_corr_rows)

print("\n  Comparison: best absolute feature vs best delta feature per element:")
print(f"  {'Element':<14} {'Best absolute':>30}  ρ_abs   {'Best delta':>40}  ρ_delta  Gain")
print("  " + "-" * 110)
for chem in CHEM_COLS:
    col = f"rho_{chem}"
    # Best absolute
    best_abs_idx = abs_corr_df[col].abs().idxmax()
    if pd.isna(best_abs_idx):
        continue
    best_abs_feat = abs_corr_df.loc[best_abs_idx, "feature"]
    best_abs_rho  = abs_corr_df.loc[best_abs_idx, col]
    # Best delta
    best_del_idx = corr_df[col].abs().idxmax()
    best_del_feat = corr_df.loc[best_del_idx, "display"]
    best_del_rho  = corr_df.loc[best_del_idx, col]

    gain = abs(best_del_rho) - abs(best_abs_rho)
    marker = "↑" if gain > 0.02 else ("↓" if gain < -0.02 else "~")
    print(f"  {CHEM_LABELS[chem]:<14} {best_abs_feat:>30}  {best_abs_rho:>+6.3f}  "
          f"{best_del_feat:>40}  {best_del_rho:>+6.3f}  {gain:>+5.3f} {marker}")

# ─── 9. Финальный summary: сводная таблица ρ ─────────────────────
print("\n" + "=" * 72)
print("SUMMARY: Top-3 delta features per element (by |ρ|)")
print("=" * 72)
for chem in CHEM_COLS:
    col = f"rho_{chem}"
    pval_col = f"pval_{chem}"
    top3 = corr_df.reindex(
        corr_df[col].abs().sort_values(ascending=False).index
    ).head(3)[["display", col, pval_col, f"n_{chem}"]]
    print(f"\n  {CHEM_LABELS[chem]} ({chem}):")
    for _, r in top3.iterrows():
        stars = "***" if r[pval_col] < 0.001 else ("**" if r[pval_col] < 0.01 else "*")
        print(f"    ρ={r[col]:+.3f} {stars}  {r['display']}  "
              f"(p={r[pval_col]:.4f}, n={int(r[f'n_{chem}'])})")

print(f"\n{'=' * 72}")
print(f"Output files:")
print(f"  {OUT_CSV.name}          — delta_dataset (original + {len(delta_feature_names)} new features)")
print(f"  {OUT_CORR.name}   — Spearman ρ table")
print(f"  delta_heatmap.png       — correlation heatmap (top-40 Δ features)")
print(f"  delta_top_scatter.png   — scatter plots top Δ features × chemistry")
print(f"\nNext step: use delta_dataset.csv as input for ML model in pixel_geo_approx.py")
print(f"  → Replace BEST_PREDICTOR with delta-features for potentially higher ρ_cv")
