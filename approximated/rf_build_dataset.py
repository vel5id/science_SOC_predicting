"""
rf_build_dataset.py
===================
Stage 1 of the RF pipeline: build a clean ML-ready dataset.

Steps:
  1. Load enriched_dataset.csv (1215 x 456)
  2. Merge delta columns from delta_dataset.csv (+80 delta/range cols)
  3. Drop metadata + secondary chemistry (not remote-sensing features)
  4. Per-target sigma filtering (±3σ per target, stored as separate masks)
  5. Log1p transform for right-skewed targets: P, S, NO3
  6. Save rf_dataset.csv and a text report

Output:
  data/features/rf_dataset.csv
  math_statistics/output/rf/rf_dataset_report.txt

Run: python approximated/rf_build_dataset.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis

BASE       = Path(__file__).parent.parent
ENRICH_CSV = BASE / "data" / "features" / "enriched_dataset.csv"
DELTA_CSV  = BASE / "data" / "features" / "delta_dataset.csv"
OUT_CSV    = BASE / "data" / "features" / "rf_dataset.csv"
OUT_DIR    = BASE / "math_statistics" / "output" / "rf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_TXT = OUT_DIR / "rf_dataset_report.txt"

TARGETS = ["ph", "k", "p", "hu", "s", "no3"]

# Secondary chemistry — real lab measurements but NOT remote-sensing features
# Including them as predictors would cause data leakage (they're co-outputs of same lab analysis)
SECONDARY_CHEM = {"cu", "mo", "fe", "zn", "mg", "mn", "soc", "b", "ca", "na"}

# Metadata — identifiers, coordinates, strings
META_COLS = {
    "id", "year", "farm", "field_name", "grid_id",
    "centroid_lon", "centroid_lat", "geometry_wkt",
    "protocol_number", "analysis_date", "sampling_date",
    "sample_id",
}

# Targets that get log1p transform (right-skewed: skew > 1.5)
# pH, K, HU — moderate/negative skew → no transform
# P (skew=2.58), S (skew=3.46), NO3 (skew=2.31) → log1p
LOG_TARGETS = {"p", "s", "no3"}

# Sigma threshold for outlier filtering
SIGMA_THRESHOLD = 3.0

# ─── Which targets get sigma filtering ───────────────────────────
# pH: skew=-0.73, 0 outliers at 3σ  → no filtering needed
# K:  skew=-0.49, 4 outliers (0.4%) → filter
# P:  skew=2.58,  23 outliers (2.1%)→ filter + log
# HU: skew=-0.27, 12 outliers (1.1%)→ filter
# S:  skew=3.46,  39 outliers (3.5%)→ filter + log (CRITICAL)
# NO3:skew=2.31,  22 outliers (2.0%)→ filter + log
SIGMA_FILTER_TARGETS = {"k", "p", "hu", "s", "no3"}  # ph excluded


print("=" * 65)
print("rf_build_dataset.py  —  RF Dataset Builder")
print("=" * 65)

# ─── 1. Load enriched dataset ────────────────────────────────────
print("\nLoading enriched_dataset.csv ...")
df = pd.read_csv(ENRICH_CSV)
print(f"  Shape: {df.shape}")

# ─── 2. Merge delta columns ──────────────────────────────────────
print("Merging delta_dataset.csv ...")
df_dlt = pd.read_csv(DELTA_CSV)
delta_cols = [c for c in df_dlt.columns
              if (c.startswith("delta_") or c.startswith("range_"))
              and c not in df.columns]
for c in delta_cols:
    df[c] = df_dlt[c]
print(f"  Added {len(delta_cols)} delta/range columns")
print(f"  Combined shape: {df.shape}")

# ─── 3. Identify column categories ───────────────────────────────
all_num = set(df.select_dtypes(include=[np.number]).columns)
meta_present     = META_COLS & set(df.columns)
secondary_present= SECONDARY_CHEM & set(df.columns)
target_present   = set(TARGETS) & set(df.columns)

# Feature columns = numeric - meta - secondary chem - targets
feat_cols = sorted(
    all_num - META_COLS - SECONDARY_CHEM - set(TARGETS)
)
print(f"\nColumn breakdown:")
print(f"  Meta cols retained for join: {len(meta_present)}")
print(f"  Secondary chemistry (excluded from features): {sorted(secondary_present)}")
print(f"  Targets: {sorted(target_present)}")
print(f"  Feature columns: {len(feat_cols)}")

# ─── 4. Target distribution analysis ─────────────────────────────
print("\nTarget distribution analysis:")
print(f"  {'Target':<6}  {'n':>5}  {'NaN%':>6}  {'mean':>8}  {'std':>7}  "
      f"{'skew':>6}  {'kurt':>6}  {'p1':>6}  {'p99':>6}")
print("  " + "-" * 75)

target_stats = {}
for t in TARGETS:
    v = df[t].dropna()
    nan_pct = 100 * df[t].isna().mean()
    sk  = float(skew(v))
    ku  = float(kurtosis(v))
    target_stats[t] = {
        "n": len(v), "nan_pct": nan_pct,
        "mean": v.mean(), "std": v.std(),
        "skew": sk, "kurt": ku,
        "p01": v.quantile(0.01), "p99": v.quantile(0.99),
        "min": v.min(), "max": v.max(),
    }
    log_flag = "log1p" if t in LOG_TARGETS else "none"
    sig_flag = "3σ" if t in SIGMA_FILTER_TARGETS else "—"
    print(f"  {t:<6}  {len(v):>5}  {nan_pct:>5.1f}%  {v.mean():>8.3f}  "
          f"{v.std():>7.3f}  {sk:>6.2f}  {ku:>6.2f}  "
          f"{v.quantile(0.01):>6.2f}  {v.quantile(0.99):>6.2f}"
          f"  [{sig_flag}, {log_flag}]")

# ─── 5. Per-target sigma filtering ───────────────────────────────
print("\nApplying per-target sigma filtering (±3σ) ...")

# We store a boolean mask per target in the dataset:
# mask_{t} = True → row is VALID for training target t
# This allows each target to have its own clean subset.
sigma_masks = {}

for t in TARGETS:
    v = df[t]
    mean = v.mean()
    std  = v.std()
    lo   = mean - SIGMA_THRESHOLD * std
    hi   = mean + SIGMA_THRESHOLD * std

    if t in SIGMA_FILTER_TARGETS:
        # True = not an outlier AND not NaN
        mask = v.notna() & (v >= lo) & (v <= hi)
        n_removed = v.notna().sum() - mask.sum()
        print(f"  {t}: [{lo:.3f}, {hi:.3f}]  removed {n_removed} outliers  "
              f"kept {mask.sum()} rows")
    else:
        # pH: only remove NaN (no sigma filter)
        mask = v.notna()
        print(f"  {t}: no sigma filter  kept {mask.sum()} rows")

    sigma_masks[t] = mask
    df[f"mask_{t}"] = mask.astype(int)  # 1 = valid row for this target

# ─── 6. Log1p transform ──────────────────────────────────────────
print("\nApplying log1p transform to right-skewed targets ...")
for t in LOG_TARGETS:
    # Store log-transformed values; original stays in t column
    log_col = f"log_{t}"
    valid_mask = df[f"mask_{t}"].astype(bool)
    # Guard: log1p(x) is undefined for x < -1; assert all valid values >= 0
    neg_vals = df.loc[valid_mask & (df[t] < 0), t]
    if len(neg_vals) > 0:
        raise ValueError(
            f"log1p applied to {t}: {len(neg_vals)} negative values found: "
            f"min={neg_vals.min():.4f}. Fix upstream sigma filter or data."
        )
    df[log_col] = np.where(valid_mask, np.log1p(df[t]), np.nan)
    v_log = df.loc[valid_mask, t]
    v_log_t = np.log1p(v_log)
    print(f"  {t}: original skew={skew(v_log.dropna()):.2f}  "
          f"log1p skew={skew(v_log_t.dropna()):.2f}  "
          f"(range: [{v_log.min():.2f}, {v_log.max():.2f}] → "
          f"[{v_log_t.min():.2f}, {v_log_t.max():.2f}])")

# ─── 7. Feature quality check ────────────────────────────────────
print(f"\nFeature quality check ({len(feat_cols)} features) ...")
n_const = 0
n_highnaN = 0
for c in feat_cols:
    if df[c].std() < 1e-6:
        n_const += 1
    if df[c].isna().mean() > 0.40:
        n_highnaN += 1
print(f"  Constant features (std < 1e-6): {n_const}")
print(f"  High-NaN features (>40% NaN):   {n_highnaN}")
print(f"  (These will be removed in rf_train_cv.py stage-1 filter)")

# ─── 8. Build output dataset ─────────────────────────────────────
# Keep: meta (for LOFO join) + all features + targets + masks + log targets
keep_meta = ["year", "farm", "field_name", "centroid_lon", "centroid_lat"]
keep_meta = [c for c in keep_meta if c in df.columns]

log_cols  = [f"log_{t}" for t in LOG_TARGETS]
mask_cols = [f"mask_{t}" for t in TARGETS]

out_cols  = keep_meta + feat_cols + TARGETS + log_cols + mask_cols
out_cols  = [c for c in out_cols if c in df.columns]

df_out = df[out_cols].copy()
print(f"\nOutput dataset shape: {df_out.shape}")
print(f"  Meta cols: {len(keep_meta)}")
print(f"  Feature cols: {len(feat_cols)}")
print(f"  Target cols: {len(TARGETS)}")
print(f"  Log target cols: {len(log_cols)}")
print(f"  Mask cols: {len(mask_cols)}")

# ─── 9. Save ─────────────────────────────────────────────────────
df_out.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
print(f"  Final shape: {df_out.shape}")

# ─── 10. Save report ─────────────────────────────────────────────
lines = []
lines.append("RF DATASET REPORT")
lines.append("=" * 65)
lines.append(f"Source: enriched_dataset.csv + delta_dataset.csv")
lines.append(f"Output: {OUT_CSV.name}")
lines.append(f"Shape: {df_out.shape[0]} rows x {df_out.shape[1]} columns")
lines.append(f"  Feature columns: {len(feat_cols)}")
lines.append(f"  Meta columns: {len(keep_meta)}")
lines.append(f"  Delta columns added: {len(delta_cols)}")
lines.append("")
lines.append("TARGET STATISTICS (before sigma filtering):")
lines.append(f"  {'Target':<6}  {'n':>5}  {'NaN%':>6}  {'mean':>9}  "
             f"{'std':>8}  {'skew':>6}  {'filter':>8}  {'transform'}")
lines.append("  " + "-" * 75)
for t in TARGETS:
    s = target_stats[t]
    sig_flag = "±3σ" if t in SIGMA_FILTER_TARGETS else "none"
    log_flag = "log1p" if t in LOG_TARGETS else "none"
    lines.append(f"  {t:<6}  {s['n']:>5}  {s['nan_pct']:>5.1f}%  "
                 f"{s['mean']:>9.3f}  {s['std']:>8.3f}  {s['skew']:>6.2f}  "
                 f"{sig_flag:>8}  {log_flag}")
lines.append("")
lines.append("SIGMA FILTER RESULTS (rows kept after ±3σ per target):")
for t in TARGETS:
    n_kept = int(df[f"mask_{t}"].sum())
    n_total = df[t].notna().sum()
    n_removed = n_total - n_kept
    lines.append(f"  {t:<6}: {n_kept}/{n_total} kept  ({n_removed} removed)")
lines.append("")
lines.append("FEATURE COLUMN GROUPS:")
groups = {
    "s2_raw":     [c for c in feat_cols if c.startswith("s2_B") and "_" in c and not c.startswith("s2_B1")],
    "s2_idx":     [c for c in feat_cols if c.startswith("s2_") and not c.startswith("s2_B")],
    "l8":         [c for c in feat_cols if c.startswith("l8_")],
    "spectral":   [c for c in feat_cols if c.startswith("spectral_")],
    "glcm":       [c for c in feat_cols if c.startswith("glcm_")],
    "topo":       [c for c in feat_cols if c.startswith("topo_")],
    "climate":    [c for c in feat_cols if c.startswith("climate_")],
    "temporal":   [c for c in feat_cols if c.startswith("ts_")],
    "cross_sensor":[c for c in feat_cols if c.startswith("cs_")],
    "delta":      [c for c in feat_cols if c.startswith("delta_") or c.startswith("range_")],
}
for gname, gcols in groups.items():
    lines.append(f"  {gname:<15}: {len(gcols):>4} features")
lines.append("")
lines.append("LOG TARGETS (log1p applied before RF training):")
for t in LOG_TARGETS:
    lines.append(f"  {t}: stored as log_{t} column")
lines.append("")
lines.append("NOTES:")
lines.append("  - mask_{t} = 1 means row is VALID for training target t")
lines.append("  - Per-target filtering: each element trained on its own clean subset")
lines.append("  - Secondary chemistry (cu,mo,fe,zn,mg,mn,soc) excluded (data leakage)")
lines.append("  - Metadata retained for LOFO-CV field-level splitting")

report_text = "\n".join(lines)
REPORT_TXT.write_text(report_text, encoding="utf-8")
print(f"Saved report: {REPORT_TXT.name}")

print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Input:  enriched ({df.shape[0]} rows) + {len(delta_cols)} delta cols")
print(f"  Output: {df_out.shape[0]} rows x {df_out.shape[1]} cols")
print(f"  Valid rows per target after ±3σ filter:")
for t in TARGETS:
    n = int(df[f"mask_{t}"].sum())
    log_note = " (log1p)" if t in LOG_TARGETS else ""
    print(f"    {t:<6}: {n}{log_note}")
print(f"\nNext step: python approximated/rf_train_cv.py")
