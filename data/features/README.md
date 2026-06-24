# `data/features/` — dataset files

This directory holds the tabular feature datasets for the modelling pipeline. The heavy
data is **not** stored in git; it is archived on Zenodo
([10.5281/zenodo.19496443](https://doi.org/10.5281/zenodo.19496443)) and unpacked here.
Only the small feature-selection config (`selected/`) and this README are tracked in git.

## TL;DR — which file to use

| Task | File |
|------|------|
| **Reproduce the paper** (ML models, all validation strategies) | **`master_dataset_old.csv`** — the canonical 1085 × 530 build |
| Anything else | see the table below |

`ML/data_loader.py` reads `data/features/master_dataset.csv`. To reproduce the published
metrics, `master_dataset.csv` **must be the canonical build** (1085 samples × 530 features,
81 fields, 20 farms, 2022–2023). The loader warns if the loaded file is not that build.

> **Version note.** In the current archive the file literally named `master_dataset.csv`
> is a *later experimental build* (2060 rows / 100 fields / 29 farms, includes 2021, a
> reduced 142-column feature set) and does **not** reproduce the paper. Until the Zenodo
> archive is refreshed, use `master_dataset_old.csv` as `master_dataset.csv` (copy/rename
> it, or point `DATA_PATH` to it).

## All files

| File | rows | cols | fields | farms | years | What it is | Used by paper? |
|------|-----:|-----:|-------:|------:|-------|------------|:--------------:|
| **`master_dataset_old.csv`** | 1085 | **530** | 81 | 20 | 2022–23 | **Canonical paper build.** One row per soil sample; full engineered feature set. Matches abstract "530 features", Table 1 (n=1085), §2.2 (174+911). | ✅ **yes** |
| `master_dataset.csv` | 2060 | 142 | 100 | 29 | 2021–23 | Later experimental build: extra farms/fields/year, reduced single-season feature set (`s2_B02_*` naming). | ❌ no |
| `master_dataset_leaky_backup.csv` | 1085 | 536 | 81 | 20 | 2022–23 | `master_dataset_old.csv` **plus 6 lab micronutrient columns** (`mg, fe, mn, zn, cu, mo`). Kept for reference only — those columns are leaky (see below). | ❌ no |
| `full_dataset.csv` | 1085 | 272 | 81 | 20 | 2022–23 | Earlier / reduced feature-engineering build. | auxiliary |
| `enriched_dataset.csv` | 1085 | 456 | 81 | 20 | 2022–23 | Extra engineered features for the `approximated/` (pixel-level RF) experiments. | auxiliary |
| `delta_dataset.csv` | 1085 | 352 | 81 | 20 | 2022–23 | Seasonal change / delta features for `approximated/`. | auxiliary |
| `field_static_features.csv` | 103 | 15 | 81 | 20 | 2022–23 | Field-level (not sample-level) static aggregates. | auxiliary |
| `mpc_checkpoint_*.csv` (×30) | — | — | — | — | — | Incremental checkpoints from the `representativeness/` sampling sub-study. | auxiliary |
| `selected/` | — | — | — | — | — | Per-target top-15 MDI feature lists (`*_best_features.txt` + `best_features.json`) used by the models. **Tracked in git.** | ✅ **yes** |
| `selected_old/` | — | — | — | — | — | Earlier feature-selection snapshot (same content now mirrored into `selected/`). | reference |

## How `master_dataset_old.csv` and `master_dataset_leaky_backup.csv` differ

They are **identical in all 529 shared columns and all 1085 rows**. The *only* difference:
`master_dataset_leaky_backup.csv` adds **6 lab-measured soil micronutrients** —
`mg, fe, mn, zn, cu, mo` (Mg, Fe, Mn, Zn, Cu, Mo). `master_dataset_old.csv` is the same
table with those 6 columns removed.

Why those columns are **leaky** (and excluded from the canonical build):

1. **Operational / target leakage.** They are wet-chemistry lab measurements taken on the
   same samples, at the same time, as the six prediction targets. The goal of the study is
   to predict agrochemistry *from satellite data* (to avoid lab analysis); at deployment
   these lab values would not exist. Using them as predictors defeats the purpose.
2. **Co-determination with the targets.** They are strong in-sample proxies for several
   targets (e.g. |Spearman| Mg↔S = 0.84, Mn↔NO₃ = 0.75, Cu↔P₂O₅ = 0.72), so a tree model
   can read the answer off a correlated lab measurement (high in-bag MDI importance).

Two caveats worth knowing (both verified by recompute):
- They are measured on only **148 / 1085 samples (~14 %)** — sparse, mostly imputed.
- Under spatial Farm-LOFO they do **not** improve (in fact slightly lower) Spearman ρ — the
  within-farm correlation does not transfer out-of-farm. A textbook reminder that in-bag MDI
  importance overstates a feature's real out-of-sample value, and that spatial CV exposes it.

The paper's feature lists (`selected/`) do not use these micronutrients, so the published
results are unaffected by this leakage.

## Feature families (canonical build)

`s2_*` Sentinel-2 seasonal bands/indices · `l8_*` Landsat-8 · `spectral_*` band ratios & PCA ·
`ts_*` cross-season time-series statistics · `delta_*` / `cs_*` / `range_*` cross-season
change features · `glcm_*` texture · `topo_*` SRTM terrain · `climate_*` ERA5-Land.
Targets: `ph, soc` (`hu` = humus), `no3, p, k, s`. Metadata: `id, year, farm, field_name,
grid_id, centroid_lon/lat, geometry_wkt, sampling_date`.

Reproduced RF Farm-LOFO Spearman ρ on the canonical build (`master_dataset_old.csv` +
`selected/`): pH 0.750 · SOC 0.529 · P₂O₅ 0.490 · K₂O 0.448 · NO₃ 0.232 · S 0.240.
