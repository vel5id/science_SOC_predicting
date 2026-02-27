"""
Extra Feature Engineering — post-processing, no GEE required.
=============================================================
Вычисляет из уже имеющихся данных (full_dataset.csv):

A. Новые спектральные индексы S2 (из существующих полос s2_B*):
   - NDWI   = (B3 - B8)  / (B3 + B8)       вода / влага листьев
   - NDMI   = (B8 - B11) / (B8 + B11)       влажность/moisture stress
   - MSI    = B11 / B8                       moisture stress index
   - NBR    = (B8 - B12) / (B8 + B12)       burn ratio (чувствителен к органике)
   - PSRI   = (B4 - B2)  / B5               пигментный индекс (каротиноиды)
   - CIre   = (B7 / B5) - 1                 хлорофилл red-edge (от B5 и B7)
   - RENDVI = (B8A - B5) / (B8A + B5)       red-edge NDVI (более точный)
   - IRECI  = (B7 - B4)  / (B5 / B6)        inverted red-edge chlorophyll
   - S2REP  = 700 + 35*((B4+B7)/2 - B5) / (B6 - B5)  red-edge position
     (приближение через имеющиеся полосы; точный S2REP требует непрерывного спектра)

B. Временны́е статистики по 4 сезонам (для каждого индекса):
   - {source}_{index}_mean   = среднее по 4 сезонам
   - {source}_{index}_std    = стандартное отклонение
   - {source}_{index}_cv     = коэффициент вариации (std/|mean|)
   - {source}_{index}_range  = max - min  (уже есть в build_delta_features, добавим сюда тоже)
   - {source}_{index}_slope  = линейный тренд spr→sum→lsm→aut (наклон регрессии)

C. Cross-sensor ратио (S2 / L8 для одноимённых индексов):
   - cs_{index}_spring, ..._summer, ..._late_summer, ..._autumn
   - Общие индексы: NDVI, GNDVI, SAVI
   - Физический смысл: разница 10m vs 30m пространственного разрешения,
     атмосферная коррекция, субпиксельная неоднородность

D. GLCM cross-channel ратио (nir/red для каждой метрики):
   - glcm_ratio_{metric}_{season} = glcm_nir_{metric} / glcm_red_{metric}
   - Физически: отношение текстуры NIR к RED — чувствительно к структуре почвы
   - Также: delta GLCM (glcm_nir_contrast_summer - glcm_nir_contrast_spring)

Выходные файлы:
   - enriched_dataset.csv       — full_dataset + все новые признаки
   - extra_correlations.csv     — Spearman ρ только новых признаков × химия
   - extra_heatmap.png          — heatmap топ-50 новых признаков
   - extra_top_scatter.png      — scatter топ-2 на элемент

Запуск: python approximated/build_extra_features.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─── Параметры ───────────────────────────────────────────────────
BASE      = Path(__file__).parent
DATA_PATH = BASE.parent / "data" / "features" / "full_dataset.csv"
OUT_DIR   = BASE.parent / "math_statistics" / "output" / "plots"
OUT_CSV   = BASE.parent / "data" / "features" / "enriched_dataset.csv"
OUT_CORR  = OUT_DIR / "extra_correlations.csv"
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

SEASONS      = ["spring", "summer", "late_summer", "autumn"]
SEASON_IDX   = {s: i for i, s in enumerate(SEASONS)}   # для линейного тренда

EPS = 1e-6  # защита от деления на 0

# ─── 1. Загрузка данных ───────────────────────────────────────────
print("Loading full_dataset.csv ...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}  |  farms: {df['farm'].nunique()}  fields: {df['field_name'].nunique()}")
print(f"  Years: {sorted(df['year'].unique())}")

new_cols_meta = []  # (col_name, display_name, group)


# ─── Helpers ─────────────────────────────────────────────────────
def safe_div(a, b, eps=EPS):
    """Safe division with NaN where |b| < eps."""
    b_arr = np.where(np.abs(b) < eps, np.nan, b)
    return np.where(np.isnan(b_arr), np.nan, a / b_arr)


def get_band(season: str, band: str) -> pd.Series:
    """Return s2_{band}_{season} as Series (raw DN, divide by 10000 for reflectance)."""
    col = f"s2_{band}_{season}"
    if col in df.columns:
        return df[col] / 10000.0   # DN → reflectance [0,1]
    return pd.Series(np.nan, index=df.index)


def add_col(name: str, values, display: str, group: str):
    df[name] = values
    new_cols_meta.append((name, display, group))


# ─────────────────────────────────────────────────────────────────
# A. Новые спектральные индексы S2
# ─────────────────────────────────────────────────────────────────
print("\n[A] Computing new S2 spectral indices ...")

for season in SEASONS:
    B2  = get_band(season, "B2")   # Blue
    B3  = get_band(season, "B3")   # Green
    B4  = get_band(season, "B4")   # Red
    B5  = get_band(season, "B5")   # Red-edge 1 (705 nm)
    B6  = get_band(season, "B6")   # Red-edge 2 (740 nm)
    B7  = get_band(season, "B7")   # Red-edge 3 (783 nm)
    B8  = get_band(season, "B8")   # NIR (842 nm)
    B8A = get_band(season, "B8A")  # Narrow NIR (865 nm)
    B11 = get_band(season, "B11")  # SWIR1 (1610 nm)
    B12 = get_band(season, "B12")  # SWIR2 (2190 nm)

    # NDWI — вода / влага листьев (McFeeters 1996)
    #   > 0 → открытая вода; -0.1…0 → влажная почва
    ndwi = safe_div(B3 - B8, B3 + B8)
    add_col(f"s2_NDWI_{season}", ndwi, f"S2·NDWI {season}", "S2_extra_index")

    # NDMI — Moisture Index (B8-B11)/(B8+B11) (Gao 1996)
    #   Vegetation water content; -1 → dry, +1 → very wet
    ndmi = safe_div(B8 - B11, B8 + B11)
    add_col(f"s2_NDMI_{season}", ndmi, f"S2·NDMI {season}", "S2_extra_index")

    # MSI — Moisture Stress Index = B11/B8
    #   High MSI → drought stress → less water in leaves
    msi = safe_div(B11, B8)
    add_col(f"s2_MSI_{season}", msi, f"S2·MSI {season}", "S2_extra_index")

    # NBR — Normalized Burn Ratio (B8-B12)/(B8+B12)
    #   Sensitive to soil organic matter; negative in burned/bare areas
    nbr = safe_div(B8 - B12, B8 + B12)
    add_col(f"s2_NBR_{season}", nbr, f"S2·NBR {season}", "S2_extra_index")

    # PSRI — Plant Senescence Reflectance Index (Merzlyak et al. 1999)
    #   PSRI = (B4 - B2) / B5  → carotenoid/chlorophyll ratio
    #   Increases during senescence (autumn) → reflects organic matter cycling
    psri = safe_div(B4 - B2, B5)
    add_col(f"s2_PSRI_{season}", psri, f"S2·PSRI {season}", "S2_extra_index")

    # CIre — Chlorophyll Index Red-Edge (Gitelson et al. 2003)
    #   CIre = B7/B5 - 1
    #   Strong proxy for canopy chlorophyll content
    cire = safe_div(B7, B5) - 1.0
    add_col(f"s2_CIre_{season}", cire, f"S2·CIre {season}", "S2_extra_index")

    # RENDVI — Red-Edge NDVI using B8A and B5 (Frampton et al. 2013)
    #   More sensitive to chlorophyll than classic NDVI
    rendvi = safe_div(B8A - B5, B8A + B5)
    add_col(f"s2_RENDVI_{season}", rendvi, f"S2·RENDVI {season}", "S2_extra_index")

    # IRECI — Inverted Red-Edge Chlorophyll Index (Frampton et al. 2013)
    #   IRECI = (B7 - B4) / (B5 / B6)
    ireci = (B7 - B4) * safe_div(B6, B5)
    add_col(f"s2_IRECI_{season}", ireci, f"S2·IRECI {season}", "S2_extra_index")

    # S2REP — Sentinel-2 Red-Edge Position (Frampton et al. 2013)
    #   Approximation: S2REP ≈ 700 + 35 * ((B4+B7)/2 - B5) / (B6 - B5)
    #   Reports the inflection wavelength of the red-edge (sensitive to chlorophyll)
    numer = (B4 + B7) / 2.0 - B5
    denom = B6 - B5
    s2rep = 700.0 + 35.0 * safe_div(numer, denom)
    add_col(f"s2_S2REP_{season}", s2rep, f"S2·S2REP {season}", "S2_extra_index")

n_s2_extra = sum(1 for _, _, g in new_cols_meta if g == "S2_extra_index")
print(f"  Added {n_s2_extra} new S2 index columns ({n_s2_extra // len(SEASONS)} indices × {len(SEASONS)} seasons)")


# ─────────────────────────────────────────────────────────────────
# B. Временны́е статистики по 4 сезонам
# ─────────────────────────────────────────────────────────────────
print("\n[B] Computing temporal statistics (mean / std / CV / slope) ...")

# Список пар (csv_prefix, index_name) для которых вычисляем статистики
TEMPORAL_SOURCES = [
    # (csv_prefix, index) — должны давать {csv_prefix}_{index}_{season}
    ("s2",       "NDVI"),
    ("s2",       "NDRE"),
    ("s2",       "GNDVI"),
    ("s2",       "EVI"),
    ("s2",       "BSI"),
    ("s2",       "SAVI"),
    ("s2",       "Cl_Red_Edge"),
    # Новые индексы (добавлены выше)
    ("s2",       "NDWI"),
    ("s2",       "NDMI"),
    ("s2",       "MSI"),
    ("s2",       "NBR"),
    ("s2",       "PSRI"),
    ("s2",       "CIre"),
    ("s2",       "RENDVI"),
    ("s2",       "IRECI"),
    ("s2",       "S2REP"),
    # Landsat
    ("l8",       "NDVI"),
    ("l8",       "GNDVI"),
    ("l8",       "SAVI"),
    # Spectral (post-processed)
    ("spectral", "NDVI"),
    ("spectral", "NDRE"),
    ("spectral", "GNDVI"),
    ("spectral", "EVI"),
    ("spectral", "BSI"),
    ("spectral", "SAVI"),
]

# Числа для линейного тренда (spr=0, sum=1, lsm=2, aut=3)
x_trend = np.array([0, 1, 2, 3], dtype=float)

for prefix, idx in TEMPORAL_SOURCES:
    # Собираем колонки 4 сезонов
    season_cols = [f"{prefix}_{idx}_{s}" for s in SEASONS]
    available   = [c for c in season_cols if c in df.columns]
    if len(available) < 2:
        continue

    mat = df[available].values.astype(float)   # shape (n_rows, n_seasons)
    n_av = len(available)

    # Индексы доступных сезонов для тренда
    x_av = np.array([SEASON_IDX[c.split("_")[-1]] for c in available], dtype=float)

    # mean
    ts_mean = np.nanmean(mat, axis=1)
    cname = f"ts_{prefix}_{idx}_mean"
    add_col(cname, ts_mean, f"{prefix.upper()}·{idx} mean", "temporal_stat")

    # std (ddof=1: sample std for small seasonal samples n=4)
    ts_std = np.nanstd(mat, axis=1, ddof=1)
    cname = f"ts_{prefix}_{idx}_std"
    add_col(cname, ts_std, f"{prefix.upper()}·{idx} std", "temporal_stat")

    # CV = std / |mean|
    ts_cv = safe_div(ts_std, np.abs(ts_mean))
    cname = f"ts_{prefix}_{idx}_cv"
    add_col(cname, ts_cv, f"{prefix.upper()}·{idx} CV", "temporal_stat")

    # slope — линейный тренд через 4 сезона (наклон регрессии)
    # для каждой строки: slope = cov(x, y) / var(x)
    # вручную, чтобы корректно обрабатывать NaN в отдельных строках
    slopes = np.full(len(df), np.nan)
    for i in range(len(df)):
        y_i = mat[i]
        mask = ~np.isnan(y_i)
        if mask.sum() >= 3:
            x_i = x_av[mask]
            y_i2 = y_i[mask]
            x_m = x_i.mean()
            y_m = y_i2.mean()
            cov = ((x_i - x_m) * (y_i2 - y_m)).sum()
            var = ((x_i - x_m) ** 2).sum()
            slopes[i] = cov / var if var > 1e-12 else np.nan
    cname = f"ts_{prefix}_{idx}_slope"
    add_col(cname, slopes, f"{prefix.upper()}·{idx} slope", "temporal_stat")

n_ts = sum(1 for _, _, g in new_cols_meta if g == "temporal_stat")
print(f"  Added {n_ts} temporal stat columns ({n_ts // 4} features × 4 stats)")


# ─────────────────────────────────────────────────────────────────
# C. Cross-sensor ратио S2/L8
# ─────────────────────────────────────────────────────────────────
print("\n[C] Computing cross-sensor ratios (S2 / L8) ...")

COMMON_INDICES = ["NDVI", "GNDVI", "SAVI"]   # индексы есть в обоих сенсорах

for idx in COMMON_INDICES:
    for season in SEASONS:
        s2_col = f"s2_{idx}_{season}"
        l8_col = f"l8_{idx}_{season}"
        if s2_col not in df.columns or l8_col not in df.columns:
            continue
        s2_vals = df[s2_col].values.astype(float)
        l8_vals = df[l8_col].values.astype(float)
        # Ratio: S2/L8 (>1 → S2 sees more vegetation at 10m vs 30m)
        ratio = safe_div(s2_vals, l8_vals)  # safe_div already handles zero denominator
        cname = f"cs_{idx}_ratio_{season}"
        add_col(cname, ratio, f"CS·{idx} S2/L8 {season}", "cross_sensor")
        # Difference: S2 - L8 (scale-consistent units)
        diff = s2_vals - l8_vals
        cname = f"cs_{idx}_diff_{season}"
        add_col(cname, diff, f"CS·{idx} S2-L8 {season}", "cross_sensor")

n_cs = sum(1 for _, _, g in new_cols_meta if g == "cross_sensor")
print(f"  Added {n_cs} cross-sensor columns")


# ─────────────────────────────────────────────────────────────────
# D. GLCM дополнительные признаки
# ─────────────────────────────────────────────────────────────────
print("\n[D] Computing GLCM cross-channel ratios and temporal deltas ...")

GLCM_METRICS  = ["asm", "contrast", "ent", "idm"]
GLCM_SEASONS  = SEASONS
# Двойной префикс из-за naming: glcm_glcm_nir_asm_spring
GLCM_PREFIX   = "glcm_glcm"

for metric in GLCM_METRICS:
    for season in GLCM_SEASONS:
        nir_col = f"{GLCM_PREFIX}_nir_{metric}_{season}"
        red_col = f"{GLCM_PREFIX}_red_{metric}_{season}"
        if nir_col not in df.columns or red_col not in df.columns:
            continue
        nir_v = df[nir_col].values.astype(float)
        red_v = df[red_col].values.astype(float)

        # NIR/Red ратио — чувствителен к NIR vs Red текстурному контрасту
        ratio = safe_div(nir_v, red_v)  # safe_div already handles zero denominator
        cname = f"glcm_ratio_{metric}_{season}"
        add_col(cname, ratio, f"GLCM NIR/Red·{metric} {season}", "glcm_extra")

    # Temporal delta для каждой метрики: summer - spring (пик вегетации)
    nir_spr = f"{GLCM_PREFIX}_nir_{metric}_spring"
    nir_sum = f"{GLCM_PREFIX}_nir_{metric}_summer"
    if nir_spr in df.columns and nir_sum in df.columns:
        delta = df[nir_sum].values - df[nir_spr].values
        cname = f"glcm_delta_nir_{metric}_spr2sum"
        add_col(cname, delta, f"GLCM Δ NIR·{metric} spr→sum", "glcm_extra")

    red_spr = f"{GLCM_PREFIX}_red_{metric}_spring"
    red_aut = f"{GLCM_PREFIX}_red_{metric}_autumn"
    if red_spr in df.columns and red_aut in df.columns:
        delta = df[red_aut].values - df[red_spr].values
        cname = f"glcm_delta_red_{metric}_spr2aut"
        add_col(cname, delta, f"GLCM Δ Red·{metric} spr→aut", "glcm_extra")

n_glcm_extra = sum(1 for _, _, g in new_cols_meta if g == "glcm_extra")
print(f"  Added {n_glcm_extra} GLCM extra columns")


# ─── Summary ─────────────────────────────────────────────────────
n_new = len(new_cols_meta)
print(f"\n{'='*60}")
print(f"Total new features added: {n_new}")
print(f"  [A] S2 extra indices:     {sum(1 for _,_,g in new_cols_meta if g=='S2_extra_index')}")
print(f"  [B] Temporal stats:       {sum(1 for _,_,g in new_cols_meta if g=='temporal_stat')}")
print(f"  [C] Cross-sensor ratios:  {sum(1 for _,_,g in new_cols_meta if g=='cross_sensor')}")
print(f"  [D] GLCM extras:          {sum(1 for _,_,g in new_cols_meta if g=='glcm_extra')}")
print(f"  Original columns:         {len(df.columns) - n_new}")
print(f"  Enriched total:           {len(df.columns)}")


# ─── 2. Корреляции Спирмена: новые признаки × химия ─────────────
print("\nComputing Spearman correlations (new features × chemistry) ...")

new_col_names = [m[0] for m in new_cols_meta]
corr_rows = []
for feat_col, feat_disp, group in new_cols_meta:
    row = {"feature": feat_col, "display": feat_disp, "group": group}
    for chem in CHEM_COLS:
        sub = df[[feat_col, chem]].dropna()
        if len(sub) < 20:
            row[f"rho_{chem}"]  = np.nan
            row[f"pval_{chem}"] = np.nan
            row[f"n_{chem}"]    = len(sub)
        else:
            r, p = spearmanr(sub[feat_col].values, sub[chem].values)
            row[f"rho_{chem}"]  = round(r, 4)
            row[f"pval_{chem}"] = round(p, 6)
            row[f"n_{chem}"]    = len(sub)
    corr_rows.append(row)

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(OUT_CORR, index=False)
print(f"  Saved: {OUT_CORR.name}")


# ─── 3. Top correlations per element ─────────────────────────────
rho_cols = [f"rho_{c}" for c in CHEM_COLS]
corr_df["max_abs_rho"] = corr_df[rho_cols].abs().max(axis=1)

print("\nTop-5 NEW features per chemistry element:")
for chem in CHEM_COLS:
    col = f"rho_{chem}"
    top = corr_df.reindex(corr_df[col].abs().sort_values(ascending=False).index).head(5)
    print(f"  {CHEM_LABELS[chem]:12s}:")
    for _, r in top.iterrows():
        pv = r.get(f"pval_{chem}", np.nan)
        stars = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else ""))
        print(f"    ρ={r[col]:+.4f} {stars:3s}  [{r['group']:18s}]  {r['display']}")


# ─── 4. Сохранение enriched_dataset.csv ─────────────────────────
print(f"\nSaving enriched_dataset.csv ...")
df.to_csv(OUT_CSV, index=False, float_format="%.6f")
print(f"  Shape: {df.shape}  →  {OUT_CSV.name}")


# ─── 5. Heatmap: новые признаки × химия ─────────────────────────
print("\nRendering correlation heatmap ...")

top_n = 50
top_feats = corr_df.nlargest(top_n, "max_abs_rho")

heat = top_feats.set_index("display")[rho_cols].copy()
heat.columns = [CHEM_LABELS[c] for c in CHEM_COLS]

fig, ax = plt.subplots(figsize=(10, 16))
fig.patch.set_facecolor("#0a0a0a")
ax.set_facecolor("#111111")

vmax = 0.7
im = ax.imshow(heat.values, aspect="auto", cmap="RdBu_r",
               vmin=-vmax, vmax=vmax, interpolation="nearest")

for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        val = heat.values[i, j]
        if np.isnan(val):
            continue
        color = "black" if abs(val) > 0.35 else "white"
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                fontsize=6.5, color=color)

ax.set_xticks(range(len(heat.columns)))
ax.set_xticklabels(heat.columns, color="white", fontsize=9)
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels(heat.index, color="white", fontsize=7)
ax.tick_params(colors="white")

cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.5, aspect=30)
cb.set_label("Spearman ρ", color="white", fontsize=9)
cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)
cb.outline.set_edgecolor("#333")

ax.set_title(
    "Spearman ρ: EXTRA features × soil chemistry\n"
    f"(top-{top_n} by max |ρ|, Groups: S2_extra / temporal / cross-sensor / GLCM-extra)",
    color="white", fontsize=10, fontweight="bold", pad=10,
)
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")

fig.text(0.5, 0.005,
         "n=1215 grid points · pure post-processing from full_dataset.csv bands",
         ha="center", color="#666", fontsize=7.5)
fig.tight_layout(rect=[0, 0.015, 1, 1])
out_hm = OUT_DIR / "extra_heatmap.png"
fig.savefig(out_hm, dpi=180, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig)
print(f"  Saved: {out_hm.name}")


# ─── 6. Top scatter plots ─────────────────────────────────────────
print("Rendering top scatter plots ...")

top_pairs = []
for chem in CHEM_COLS:
    col = f"rho_{chem}"
    best = corr_df.reindex(corr_df[col].abs().sort_values(ascending=False).index).head(2)
    for _, row in best.iterrows():
        top_pairs.append((row["feature"], row["display"], chem, row[col]))

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
axes_flat = axes.ravel() if nrows > 1 else axes if ncols == 1 else axes.ravel()

for i, (feat, disp, chem, rho) in enumerate(top_pairs_uniq[:n_panels]):
    ax = axes_flat[i]
    ax.set_facecolor("#111111")

    sub = df[[feat, chem]].dropna()
    x_vals = sub[feat].values
    y_vals = sub[chem].values

    sc = ax.scatter(x_vals, y_vals, c=y_vals,
                    cmap="plasma", s=18, alpha=0.65, linewidths=0, zorder=3)

    z = np.polyfit(x_vals, y_vals, 1)
    xr = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(xr, np.polyval(z, xr), color="#4a9eda", linewidth=1.5, zorder=4)

    ax.set_xlabel(disp, color="white", fontsize=7.5)
    ax.set_ylabel(CHEM_LABELS[chem], color="white", fontsize=7.5)
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

for j in range(n_panels, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig2.suptitle(
    "Top EXTRA feature × soil chemistry scatter plots\n"
    "(Spearman ρ, n=1215 grid points, new indices + temporal + cross-sensor + GLCM-extra)",
    color="white", fontsize=10, fontweight="bold", y=1.01,
)
fig2.tight_layout(pad=0.5)
out_sc = OUT_DIR / "extra_top_scatter.png"
fig2.savefig(out_sc, dpi=180, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig2)
print(f"  Saved: {out_sc.name}")


# ─── 7. Summary ──────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY: Top-3 EXTRA features per element (by |ρ|)")
print("=" * 72)
for chem in CHEM_COLS:
    col = f"rho_{chem}"
    top3 = corr_df.reindex(
        corr_df[col].abs().sort_values(ascending=False).index
    ).head(3)[["display", "group", col, f"pval_{chem}", f"n_{chem}"]]
    print(f"\n  {CHEM_LABELS[chem]} ({chem}):")
    for _, r in top3.iterrows():
        pv = r[f"pval_{chem}"]
        stars = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else " "))
        print(f"    ρ={r[col]:+.4f} {stars}  [{r['group']}]  {r['display']}")

print(f"\n{'='*72}")
print("Output files:")
print(f"  {OUT_CSV.name:<30} — full_dataset + {n_new} new features")
print(f"  {OUT_CORR.name:<30} — Spearman ρ table (new features only)")
print(f"  extra_heatmap.png              — correlation heatmap (top-{top_n})")
print(f"  extra_top_scatter.png          — scatter plots top new features")
print()
print("Next steps:")
print("  1. python approximated/build_delta_features.py  (seasonal deltas)")
print("  2. Review extra_correlations.csv → update BEST_PREDICTOR in pixel_geo_approx.py")
print("  3. Build RF model using enriched_dataset.csv + delta_dataset.csv")
