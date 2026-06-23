# |ρ|max Discrepancy Report — Article 1 (published modeling) vs Article 2 (draft correlation)

**Date:** 2026-06-23
**Author:** automated forensic audit (recomputed from raw data; no values fabricated)
**Scope:** Explain why the maximum Spearman |ρ| between soil properties and remote-sensing
features disagrees between the two manuscripts, and specify the fix required before the
draft correlation paper is submitted.

---

## TL;DR (top-line conclusion) — **MIXED verdict (form c)**

Both papers screen the **same feature file** (`data/features/master_dataset_old.csv`:
530 columns = 11 metavariables + 7 soil targets + **512 base RS features**; +110 analysis-time
composites = **622 screened**). The "530 vs 512 vs 622" is **not** a contradiction — it is three
counts of one dataset (verified: the screening pool computes to exactly **512**).

The difference in |ρ|max is **entirely a feature-pool / leakage-filtering difference**, not a
different correlation definition, n, or dataset regeneration:

- **The draft (Article 2) reports the *raw* full-pool maxima** — these reproduce **exactly** from
  `master_dataset_old.csv` with plain abs-max Spearman.
- **The published paper (Article 1) reports *leakage-controlled / target-specific* maxima** — lower,
  because it excluded cross-season (summer/late-summer/autumn) features, GLCM texture, and (for the
  nutrients NO₃/S) canopy spectral proxies, plus a spatial-permutation check.

Per property:

| Property | Verdict | Action |
|---|---|---|
| pH, K₂O | Match, spring/non-leaky | No change |
| **S = 0.383** | **ARTIFACT (temporal leakage)** | **Revert to 0.280** |
| **P₂O₅ = 0.525** | **ARTIFACT (autumn texture, leakage-suspect)** | **Revert headline to 0.476** (show 0.525 only as labeled raw max) |
| **NO₃ = 0.431** | **DEFENSIBLE** (spring SAVI, *not* temporally leaky) | Keep, but add reconciliation + biomass-proxy caveat |
| **SOC = 0.368** | **DEFENSIBLE** (slope, static, non-leaky) | Keep (≈ published 0.350); one-line note |

The single most important finding: **the draft's S=0.383 winner (`SAVI_L8, late-summer`) is the exact
temporal-leakage pattern Article 1 deliberately removed**, and the draft's own Discussion already
declares S "essentially unpredictable" (ICC 0.17). The S headline is therefore both an artifact and
internally self-contradictory.

---

## 1. Inventory (what was used)

**Screening code (identical logic for both papers):**
[`math_statistics/correlation_analysis.py`](math_statistics/correlation_analysis.py) →
`compute_all_spearman()` + `top_correlations()`.
Method: `scipy.stats.spearmanr` per (target × feature), **pairwise NaN deletion** (min n=10),
ranked by `abs(rho)`, **Benjamini–Hochberg FDR** per target.
Feature pool = all numeric columns except `{id, year, farm, field_name, grid_id, centroid_lon,
centroid_lat, geometry_wkt, protocol_number, analysis_date, sampling_date, hu}` and the 6 targets.

**Feature matrices present** (`data/features/`):

| File | rows | cols | screening pool | notes |
|---|---|---|---|---|
| `master_dataset_old.csv` | 1085 | **530** | **512** | **the screened file for BOTH papers**; full SAVI(47)+GLCM(56) banks |
| `master_dataset_leaky_backup.csv` | 1085 | 536 | 518 | near-identical superset of `_old` |
| `full_dataset.csv` | 1085 | 272 | 254 | reduced set **+ 6 leaked soil micronutrients** (`cu,fe,mg,mn,mo,zn`, n=148) — *not* a paper's pool; this is what `math_statistics/config.py` currently points to |
| `master_dataset.csv` (current) | **2060** | 142 | 124 | leakage-pruned **modeling** set; different sample construction; reproduces neither paper (pH→0.536) |

**Manuscripts** (named files `agriculture-16-01239.pdf` / `Yaskak_article_2…docx` are **not in the
repo**; their content is present as):
- Published modeling = [`articles/article2_prediction/main.docx`](articles/article2_prediction/main.docx)
  and the MDPI under-review set in `for_review_answer_and_change/`.
- Draft correlation = [`articles/article1_correlations/article1_correlations.docx`](articles/article1_correlations/article1_correlations.docx) (Jun 19 2026).

> Note: the repo's internal numbering (`article1_correlations`, `article2_prediction`) is the
> **reverse** of the task's numbering. This report tracks papers by content
> (**published-modeling** vs **draft-correlation**), not by number.

---

## 2. Reproduction — matching anchors and disputed values

**Minimal reproducible command** (root venv; data is private/local):

```bash
wsl -d ubuntu-new -- bash -lc 'cd ~/projects/fertilizers_remote_sensing_sci && .venv/bin/python - <<PY
import pandas as pd
from scipy import stats
SOIL=["ph","soc","no3","p","k","s"]
EXCL={"id","year","farm","field_name","grid_id","centroid_lon","centroid_lat",
      "geometry_wkt","protocol_number","analysis_date","sampling_date","hu"}
df=pd.read_csv("data/features/master_dataset_old.csv",low_memory=False)
F=[c for c in df.columns if c not in EXCL and c not in SOIL and df[c].dtype in ("float64","int64")]
for t in SOIL:
    best=(0,None,0)
    for c in F:
        m=df[[t,c]].notna().all(axis=1); n=int(m.sum())
        if n<10: continue
        r,_=stats.spearmanr(df.loc[m,t],df.loc[m,c])
        if pd.notna(r) and abs(r)>best[0]: best=(abs(r),c,n)
    print(t, round(best[0],3), best[1], best[2])
PY'
```

**Result (raw full-pool max on the 512-feature pool) = the DRAFT values, exactly:**

| Property | Reproduced |ρ|max | Winning feature | effective n |
|---|---|---|---|---|
| pH | 0.670 | `l8_GNDVI_spring` | 1085 |
| K₂O | 0.478 | `s2_BSI_spring` | 1085 |
| P₂O₅ | **0.525** | `glcm_glcm_red_ent_autumn` | **1003** |
| NO₃ | **0.431** | `s2_SAVI_spring` | 1085 |
| S | 0.418 / **0.383** | `ts_l8_NDVI_mean` / `l8_SAVI_late_summer` | 1085 |
| SOC | **0.368** | `topo_slope` | 1085 |

(The draft reports S = **0.383** = best *single-season index* `l8_SAVI_late_summer`; the absolute
pool max is 0.418 via a time-series composite.)

**n actually used = 1085 base samples, pairwise-complete per feature** → effective n varies
**1003–1085** (autumn GLCM features miss ~82 samples → n=1003). It is **not** 1071, 1051, or 14/20/34-reduced.

`full_dataset.csv` and `master_dataset_leaky_backup.csv` give absurd maxima (SOC 0.65, NO₃ 0.75,
S 0.85) because of the **6 soil micronutrient columns** (`mn,cu,mg,…`, n=148) leaking target
chemistry into the pool — **neither paper's numbers**; do not screen on those files.

---

## 3. Per-property verdict table

Leakage rule applied (Article 1's own standard): the sample is **~75 % spring-collected**, so any
winner from **summer / late-summer / autumn** imagery is **temporal-leakage-suspect**.

| Property | Published |ρ| | Draft |ρ| | Reproduced (raw pool) | Draft winning feature | Season | Root cause | Verdict | Required fix |
|---|---|---|---|---|---|---|---|---|---|
| **pH** | 0.670 | 0.670 | 0.670 ✓ | GNDVI_L8 | spring | — | **MATCH** | none |
| **K₂O** | 0.478 | 0.478 | 0.478 ✓ | BSI_S2 | spring | — | **MATCH** | none |
| **P₂O₅** | 0.476 | 0.525 | 0.525 ✓ | GLCM ENT_Red | **autumn** | **H2** (cross-season texture) + H1 (texture bank) | **ARTIFACT** | revert headline → **0.476** (`climate_GS_temp`, n=1085) |
| **SOC** | 0.350 | 0.368 | 0.368 ✓ | Slope | static | H1/H3 (pool/curation; 0.018 gap) | **DEFENSIBLE** | keep 0.368 (≈0.350); footnote |
| **NO₃** | 0.290 | 0.431 | 0.431 ✓ | SAVI_S2 | **spring** | H1 (published excluded canopy spectral proxies for nutrients) | **DEFENSIBLE** (not temporally leaky) | keep 0.431 + reconciliation sentence + biomass-proxy caveat |
| **S** | 0.280 | 0.383 | 0.383 ✓ | SAVI_L8 | **late-summer** | **H2** (temporal leakage — exactly what Art. 1 removed) | **ARTIFACT** | revert → **0.280**; update all 5 locations |

### Why P₂O₅ = 0.476 (not 0.525)
Ranked P₂O₅ features: positions 1–8 are all **GLCM autumn/summer texture** (0.525→0.477, several at
n=1003); position 9 = `climate_GS_temp` **0.476** (n=1085). The published paper drops texture +
cross-season → lands on `climate_GS_temp` = **0.476**. Texture entropy of an autumn red band has no
plausible direct link to soil P₂O₅ and correlates with farm identity (spatial leakage at farm scale).

### Why S = 0.280 (not 0.383)
Ranked S features are dominated by `ts_*` time-series means and **late-summer/summer** vegetation
indices (0.42→0.36). The draft picks `l8_SAVI_late_summer` = 0.383. This is the canonical temporal
leak (late-summer canopy vs spring soil sampling). The draft's **own Discussion** states S is
"по существу непредсказуема … сульфаты не имеют диагностических полос поглощения 400–2500 nm"
(ICC 0.17) — so 0.383 contradicts the paper's own conclusion.

### Why NO₃ = 0.431 is *defensible* but still contradicts 0.290
`s2_SAVI_spring` is a **spring** feature, temporally aligned with the spring-dominant sample — **not**
temporal leakage. It is a genuine, reproducible pool maximum. The published 0.290 is lower because the
modeling paper treated canopy vegetation indices as **biomass/management proxies** for the nutrient
targets and excluded them (a *single global* veg-index exclusion is impossible — it would also drop
pH to 0.659 — so the published filter is **target-specific** + spatial-permutation-corrected; this
recipe could be partially but not exactly reproduced from the available code). Interpretively, spring
canopy vigor reflects fertilization history rather than direct soil-nitrate sensing, so 0.431 must
**not** be over-interpreted as soil-predictive.

### Why SOC 0.368 ≈ 0.350
Both are the **slope/topography** signal (`topo_slope`, static, non-leaky); 0.018 apart, within
feature-curation noise. Not an artifact.

---

## 4. Hypotheses — outcome

- **H1 (expanded pool):** Confirmed for **NO₃** and **SOC** (and partly P₂O₅) — the draft's winners
  exist and are non-leaky; the published paper used a narrower/target-specific candidate set.
- **H2 (temporal-leakage contamination):** Confirmed for **S** (late-summer SAVI) and **P₂O₅**
  (autumn GLCM). These are inflated artifacts; Article 1 is correct.
- **H3 (different definition/preprocessing):** **Rejected** — identical Spearman + pairwise-NaN logic;
  pH/K₂O reproduce to 3 decimals on the same file; n=1085 (pairwise 1003–1085).
- **H4 (different dataset version):** **Rejected as the cause** — both papers screen
  `master_dataset_old.csv` (530/512/622). The other files (`full_dataset`, current `master_dataset`
  2060-row) reproduce neither and were not the screening basis.

---

## 5. Feature-pool count reconciliation (530 vs 512 vs 622)

All three describe **one** file, `master_dataset_old.csv`, and are mutually consistent:

- **530** = total columns = **11 metavariables + 7 soil targets + 512 features** (verbatim in the draft:
  *"канонический набор данных из 530 столбцов … 11 метапеременных, 7 почвенных целей и 512 признаков"*).
- **512** = numeric RS feature pool actually screened (reproduced exactly: `npool=512`).
- **622** = 512 base + **110 analysis-time composites** (`composite_features.py`) = the screened space.

The published modeling paper says "530 covariates"; the draft says "512 base + 110 = 622". **Same data.**
Recommendation: both papers should state the **512 base / 622 screened** breakdown to avoid a reviewer
reading "530" as a third number.

---

## 6. Required edits to the draft (propagate to ALL locations)

The draft prints the disputed hierarchy in **six** places — all must stay consistent:
abstract «Основные результаты», results «Иерархия предсказуемости по ρ среди всех признаков»,
RQ1, **Table 8** (the |ρ|max hierarchy = task's "Table 7"), **Table 13** ("Single |ρ|" column),
and the Conclusion.

**Mandatory (artifacts):**
- **S: 0.383 → 0.280**, winner `SAVI_L8, late-summer` → leakage-controlled best (spring/bare-soil).
- **P₂O₅: 0.525 → 0.476** headline, winner `GLCM ENT_Red autumn` → `climate_GS_temp`. The 0.525 raw
  texture max may be retained **only** in a clearly labeled "raw full-pool maximum (uncontrolled for
  temporal leakage)" column with a caveat.

**Keep but reconcile (defensible):**
- **NO₃ 0.431** and **SOC 0.368** — add the sentence below.

> Note: keeping NO₃=0.431 changes the hierarchy **order** relative to the published paper
> (published: SOC > NO₃; draft: NO₃ > SOC). If exact cross-paper agreement is preferred over
> preserving the draft's larger screening pool, the alternative is to revert all four to the
> published leakage-controlled values **0.476 / 0.350 / 0.290 / 0.280** and present 0.525/0.368/0.431/0.383
> as a separate "raw full-pool" supplementary column.

### Exact reconciliation sentence to add (draft Methods / start of §3 results)

> «Приведённые значения |ρ| — максимумы по полному пулу из 622 признаков **без** фильтрации
> темпоральной утечки. В предиктивной работе (Часть 2) для свойств, где ведущий признак относится к
> летнему/позднелетнему/осеннему окну (P₂O₅ — осенняя GLCM-текстура; S — позднелетний SAVI), при
> 75 %-весенней выборке применяется контроль темпоральной утечки, что снижает |ρ| до 0.476 (P₂O₅) и
> 0.280 (S); для NO₃ канопийные спектральные индексы трактуются как прокси биомассы/агроменеджмента и
> исключаются, что снижает |ρ| с 0.431 до 0.290. Настоящая работа сообщает скрининговые максимумы
> полного пула, отмечая эти случаи как требующие осторожной интерпретации.»

English equivalent:

> "The reported |ρ| are maxima over the full 622-feature pool **without** temporal-leakage filtering.
> For properties whose leading feature falls outside the spring sampling window (P₂O₅ — autumn GLCM
> texture; S — late-summer SAVI), the companion modeling study applies temporal-leakage control,
> reducing |ρ| to 0.476 (P₂O₅) and 0.280 (S); for NO₃, canopy spectral indices are treated as
> biomass/management proxies and excluded, reducing |ρ| from 0.431 to 0.290. This screening paper
> reports full-pool maxima and flags these cases as requiring cautious interpretation."

---

## 7. Grant-number finding (reporting only — do **not** edit the published PDF)

- The repo contains **only `BR24992839`** — it appears in the published English manuscript funding line
  (`for_review_answer_and_change/merged_fulltext_final.txt` and `merged_fulltext.txt`):
  *"…funded by the grant … of the Republic of Kazakhstan **BR24992839** …"*.
- **`BR24992785` is absent from the entire repo** (no `.py/.tex/.md/.txt/.docx` match).
- I **cannot verify from the repo alone** which number is correct (no grant-award document is present).
  **Flag for the author:** confirm against the official award. If the true grant is `BR24992785`, the
  published PDF's `BR24992839` is a typo to correct in the draft (Article 2) and any erratum —
  **but do not alter the published Article 1 PDF**.

---

## 8. Integrity statement

All |ρ| values in §2–§5 were **recomputed** from `data/features/master_dataset_old.csv` with the
repo's own screening logic; none are fabricated. Where a published value (NO₃ 0.290, S 0.280) could
**not** be exactly reproduced from available code, this is stated explicitly as a partially-reproduced,
target-specific leakage-aware screen rather than guessed. The grant number is reported as observed,
not adjudicated.
