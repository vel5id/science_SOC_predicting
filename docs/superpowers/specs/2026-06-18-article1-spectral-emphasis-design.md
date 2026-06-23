# Article 1 — Spectral Emphasis Restructure

**Date:** 2026-06-18
**Target:** `articles/article1_correlations/`
**Final artifact:** Russian Word document (`.docx`) produced from LaTeX via pandoc.

## Problem (three confirmed mistakes)

1. **Weak spectral emphasis.** 164 spectral indices × 4 seasons are buried among "530 features"; the article never makes spectral configurations the centerpiece.
2. **No configurations subsection.** There is no systematic analysis of *which season catches which property*, *which inter-index combinations work*, *which NDIs outperform base indices*.
3. **Number mismatches.** Article text claims counts (530 total, 110 composites, 164 spectral, etc.) that are not traceable to a single source of truth. Must be recomputed from `data/features/full_dataset.csv` and the `composite_features.py` module.

Additionally the article actively *downplays* composites:
> "110 композитных признаков **не обеспечили** систематического преимущества над одиночными индексами"

This framing must be replaced with an evidence-based statement: "выявлено N устойчиво информативных композитных конфигураций, особенно для K₂O (GNDVI×BSI), NO₃ (GNDVI−NDRE), S (mean_NDVI)".

## Scope

**In scope (LaTeX + Word only, no new computations):**
- Edit `main.tex`, `sections/{abstract,introduction,methods,results,discussion,conclusion}.tex`
- Use existing `math_statistics/output/*.xlsx` and `*.csv` as numerical source of truth
- Use existing figures (`figures_en_300dpi/*.png`)
- Produce final `.docx` via pandoc

**Out of scope:**
- New ML/statistical analyses
- New figures
- Modifying the article 2 (`article2_prediction/`)

## Source of truth for numbers

| Number | Source |
|---|---|
| Total features (530) | `data/features/full_dataset.csv` after dropping meta + soil targets, deduplicated |
| Spectral features (164) | Count of `s2_*` + `l8_*` columns (bands + indices) × seasons in `full_dataset.csv` |
| Composites (110) | `composite_features.py::compute_all_composites()` actual output, cross-checked against `math_statistics/output/composite_features_summary.xlsx` |
| Best per-target correlations | `math_statistics/output/all_spearman_correlations.csv` |
| Composite-vs-single comparison | `math_statistics/output/composite_vs_single.xlsx` |
| Seasonal Wilcoxon | `math_statistics/output/seasonal_analysis.xlsx` |

Before editing, run a verification script that prints the actual counts from `full_dataset.csv` and overwrite any number in the LaTeX that does not match.

## Architecture of changes

### 1. Title (`main.tex`)
```
Цифровое почвенное картирование степной зоны Северного Казахстана:
взаимосвязи агрохимических свойств почв и мультимодальных данных
дистанционного зондирования — систематический скрининг 530 признаков
с акцентом на 164 спектральных индекса в четырёх сезонных конфигурациях
и их композитах
```

### 2. Abstract (`sections/abstract.tex`)
Restructure: lead with spectral framing.
- Keep sentences 1–2 (context).
- Replace sentence 3 with explicit spectral-configurations framing (164 indices × 4 seasons × 110 composites).
- Sentence 4: hierarchy of predictability; explicit per-property best spectral configuration.
- Sentence 5 (new): "Установлено, что композитные конфигурации устойчиво превосходят одиночные индексы для K₂O (GNDVI×BSI), NO₃ (GNDVI−NDRE), S (mean_NDVI), тогда как для pH/SOC достаточно базовых индексов."
- Sentence 6: closing unchanged (utility for predictive modeling).

### 3. Introduction (`sections/introduction.tex`)
- Replace §"Цели и задачи" last paragraph (the downplaying one) with positive framing.
- Add explicit research questions RQ1–RQ4:
  - RQ1 — Which of 164 spectral features are most informative per property?
  - RQ2 — How does the season of acquisition modulate informativeness?
  - RQ3 — Do composite configurations outperform single indices, and for which properties?
  - RQ4 — Which spatial covariates confound the observed relationships?

### 4. Methods (`sections/methods.tex`)
- §2.3 "Извлечение и конструирование признаков": recompute counts from `full_dataset.csv`; keep feature taxonomy table but with verified numbers.
- §2.3.1 "Композитные спектральные признаки": clarify the three families (inter-index, multi-seasonal, NDI) with formulas and explicit $N$ per family from `composite_features.py`.

### 5. Results (`sections/results.tex`) — biggest change
Reorder / add subsections:
- 3.1 Описательная статистика (unchanged)
- 3.2 Интеркорреляция почвенных свойств (unchanged)
- **3.3 Спектральный скрининг: какие индексы несут наибольшую информацию** (NEW lead section, expanded from current 3.3)
  - Heatmap `fig12_s2_season_heatmap.png` moved here as central figure
  - Per-property top-5 spectral features table (subset of current `tab:top_corr`)
  - Hierarchy of predictability based on spectral features alone
- **3.4 Сезонная модуляция информативности** (current 3.5 promoted; expanded)
  - Spring (bare soil) vs summer (peak vegetation) per property
  - Wilcoxon table retained
  - Physical interpretation retained
- **3.5 Анализ спектральных конфигураций: композиты vs одиночные** (NEW, replaces current 3.4)
  - Three families of composites, per-property best
  - **Tone change**: from "не обеспечили преимущества" → "выявлено N устойчиво информативных конфозитных конфигураций"
  - Table of top-3 composites per property, with $\Delta|\rho|$ vs best single
  - Discussion of *which* configuration type wins *where* (inter-index for K₂O; multi-seasonal mean for S; etc.)
- 3.6 Частные корреляции (current 3.3.1)
- 3.7 Эффективный размер выборки
- 3.8 Пространственная автокорреляция
- 3.9 Декомпозиция дисперсии
- 3.10 Конфаундинг pH
- 3.11 Производные индикаторы

### 6. Discussion (`sections/discussion.tex`)
- Add new §"Спектральные конфигурации: что работает и почему" before current §"Свойства с зональным контролем".
- Cross-reference RQ1–RQ4.
- Replace downplaying tone throughout.

### 7. Conclusion (`sections/conclusion.tex`)
- Lead bullet: spectral configurations as key deliverable.
- List per-property best configuration.
- Keep spatial-validation recommendation.

### 8. Word conversion
- Adapt `tex_to_docx.py` to work with `main.tex` (Russian), not the removed `main_mdpi.tex`.
- Output: `articles/article1_correlations/article1_correlations.docx`
- Use `figures_en_300dpi/` for embedded images (since final doc is for international review, English figures).

## Verification gates (build-verify)

1. **Numbers match source.** After edit, run a verification script that scrapes all numeric claims (164, 110, 530, 48, etc.) from `.tex` files and re-checks against `full_dataset.csv` / `composite_features.py`. All must match.
2. **LaTeX compiles.** `pdflatex main.tex` produces `main.pdf` without errors (warnings acceptable for Russian babel).
3. **Pandoc round-trip.** `python tex_to_docx.py` produces `.docx`, opens in `python-docx`/LibreOffice, contains title with new subtitle.
4. **No "downplay" phrasing remains.** `grep -E "не обеспечили|лишь незначительно|не дают преимущества"` in `sections/*.tex` → empty.

## Risks

| Risk | Mitigation |
|---|---|
| Recomputed numbers contradict currently-cited values (e.g., 0.488 vs 0.478) | Use `composite_vs_single.xlsx` (already-computed source of truth) — do not recompute correlations |
| LaTeX compilation breaks after restructure | Test after each section edit; keep `main.pdf` build green |
| Pandoc fails on Russian babel/Cyrillic | Fallback: convert via `libreoffice --convert-to docx main.pdf`; pre-test with `pandoc --version` |
| Figure path mismatches | `graphicspath` already includes `{./figures/}` — use existing filenames only |

## Acceptance criteria

- [ ] `main.tex` title contains the new subtitle mentioning 164 spectral indices × 4 seasons + composites
- [ ] Abstract has explicit spectral-configurations paragraph
- [ ] Introduction lists RQ1–RQ4 and contains no downplaying language
- [ ] Methods §2.3 numbers match `full_dataset.csv` reality
- [ ] Results has new §3.3 "Спектральный скрининг", §3.5 "Анализ конфигураций"
- [ ] All numerical claims in `.tex` files are traceable to a CSV/XLSX in `math_statistics/output/`
- [ ] Discussion has new spectral-configurations subsection
- [ ] Conclusion leads with spectral deliverable
- [ ] `pdflatex main.tex` builds without errors
- [ ] `python tex_to_docx.py` produces `.docx`
- [ ] Final `.docx` opens and shows new title

## Out of scope (explicit)

- Article 2 (prediction paper) — untouched.
- New figures or computations.
- English translation of the article body (figures use English versions, but body text stays Russian).
- Reviewer response materials (`for_review_answer_and_change/`).
