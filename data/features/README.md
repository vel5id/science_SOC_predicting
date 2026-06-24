# `data/features/` — dataset files

This directory holds the tabular feature datasets for the modelling pipeline. The heavy
data is **not** stored in git; it is archived on Zenodo
([10.5281/zenodo.19496443](https://doi.org/10.5281/zenodo.19496443)) and unpacked here.
Only the small feature-selection config (`selected/`) and this README are tracked in git.

## TL;DR — which file to use

| Task | File |
|------|------|
| **Reproduce the paper** (ML models, all validation strategies) | **`master_dataset.csv`** — the canonical 1085 × 530 build |
| Anything else | see the table below |

`ML/data_loader.py` reads `data/features/master_dataset.csv`. In this archive that file
**is** the canonical build (1085 samples × 530 features, 81 fields, 20 farms, 2022–2023),
so the modelling pipeline runs out of the box. The loader warns if a non-canonical file is
substituted.

> **Version note.** An earlier archive snapshot shipped a *later experimental build*
> (2060 rows / 100 fields / 29 farms, includes 2021, a reduced 142-column feature set)
> under the name `master_dataset.csv`; it does **not** reproduce the paper. If your copy of
> `master_dataset.csv` has those dimensions, re-download this corrected archive.

## All files

| File | rows | cols | fields | farms | years | What it is | Used by paper? |
|------|-----:|-----:|-------:|------:|-------|------------|:--------------:|
| **`master_dataset.csv`** | 1085 | **530** | 81 | 20 | 2022–23 | **Canonical paper build** (what the pipeline loads). One row per soil sample; full engineered feature set. Matches abstract "530 features", Table 1 (n=1085), §2.2 (174+911). | ✅ **yes** |
| `master_dataset_old.csv` | 1085 | 530 | 81 | 20 | 2022–23 | Provenance copy of the canonical build (identical to `master_dataset.csv`). | reference |
| `master_dataset_leaky_backup.csv` | 1085 | 536 | 81 | 20 | 2022–23 | `master_dataset_old.csv` **plus 6 lab micronutrient columns** (`mg, fe, mn, zn, cu, mo`). Kept for reference only — those columns are leaky (see below). | ❌ no |
| `full_dataset.csv` | 1085 | 272 | 81 | 20 | 2022–23 | Earlier / reduced feature-engineering build. | auxiliary |
| `enriched_dataset.csv` | 1085 | 456 | 81 | 20 | 2022–23 | Extra engineered features for the `approximated/` (pixel-level RF) experiments. | auxiliary |
| `delta_dataset.csv` | 1085 | 352 | 81 | 20 | 2022–23 | Seasonal change / delta features for `approximated/`. | auxiliary |
| `field_static_features.csv` | 103 | 15 | 81 | 20 | 2022–23 | Field-level (not sample-level) static aggregates. | auxiliary |
| `mpc_checkpoint_*.csv` (×30) | — | — | — | — | — | Incremental checkpoints from the `representativeness/` sampling sub-study. | auxiliary |
| `selected/` | — | — | — | — | — | Per-target top-15 MDI feature lists (`*_best_features.txt` + `best_features.json`) used by the models. **Tracked in git.** | ✅ **yes** |
| `selected_old/` | — | — | — | — | — | Earlier feature-selection snapshot (same content now mirrored into `selected/`). | reference |

## Known data-quality note — duplicate field "19-20"

A post-publication audit found that field **`19-20`** (farm *Агро Парасат*, sampled
**2023-04-25**) was entered a **second time** under the name **`19-20 (1)`**. Its 14 grid
points appear **twice** in the dataset — identical coordinates, geometry and identical six
lab values (pH, SOC, NO₃, P₂O₅, K₂O, S), differing only in the `id` column (the duplicate
copies are `id` **379–392**).

Therefore the raw build holds **1085 records / 81 field names**, but the number of **unique
samples is 1071** and the number of **unique fields is 80**.

- Because the duplicate pair shares coordinates, under both **Field-LOFO** and **Farm-LOFO**
  the two copies always fall in the **same** CV fold — they do **not** leak across
  train/test. The only effect is a ~1.3 % inflation of `n` (those 14 samples are counted
  twice in pooled metrics), which is negligible for the Spearman-ρ rankings the paper reports.
- **The published paper (Agriculture 2026, 16, 1239) reports n = 1085 / 81 fields**, and the
  canonical `master_dataset.csv` (1085 × 530) is kept **unchanged** so the published metrics
  remain reproducible.

A de-duplicated build (**1071 × 530, 80 fields**) can be produced on demand, without touching
the canonical file:

```bash
python scripts/deduplicate_dataset.py \
    --input  data/features/master_dataset.csv \
    --output data/features/master_dataset_dedup.csv
```

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

Reproduced RF Farm-LOFO Spearman ρ on the canonical build (`master_dataset.csv` +
`selected/`): pH 0.750 · SOC 0.529 · P₂O₅ 0.490 · K₂O 0.448 · NO₃ 0.232 · S 0.240.
