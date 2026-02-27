# RF Training Plan — Digital Soil Mapping (DSM)
# Random Forest для предсказания pH, K, P, Humus, S, NO3

## Цель

Обучить Random Forest модели для каждого из 6 химических показателей почвы.
Входные данные: объединённый датасет (full_dataset + extra + delta features).
Оценка: пространственно честная кросс-валидация (Spatial LOFO-CV).
Выход: карты предсказания + рейтинг важности признаков для статьи.

---

## 0. Предварительные условия

### 0.1 Датасеты (уже созданы)
| Файл | Строк | Колонок | Описание |
|------|-------|---------|---------|
| `data/features/full_dataset.csv` | 1121 | 272 | Оригинальный (S2, L8, Spectral, GLCM, Topo, Climate) |
| `data/features/enriched_dataset.csv` | 1121 | 456 | +184 extra (NDWI/NDMI/MSI/…, temporal stats, CS ratios, GLCM extras) |
| `data/features/delta_dataset.csv` | 1121 | 352 | +80 delta (Δ seasonal, RANGE) |

### 0.2 Слияние в единый датасет для RF
Скрипт слияния: `approximated/build_rf_dataset.py` (будет создан)
- Загружает `enriched_dataset.csv` (456 cols)
- Добавляет delta-колонки из `delta_dataset.csv` (80 новых cols)
- Итого: **~536 feature columns** + мета + таргеты
- Сохраняет: `data/features/rf_dataset.csv`

---

## 1. Определение признакового пространства

### 1.1 Группы признаков (feature groups для анализа важности)

| Group ID | Описание | Кол-во признаков |
|----------|----------|----------------|
| `s2_raw` | S2 bands (B2–B12) × 4 seasons | 40 |
| `s2_idx` | S2 indices (NDVI, NDRE, GNDVI, EVI, BSI, SAVI, CRE) × 4 | 28 |
| `s2_extra` | S2 extra (NDWI, NDMI, MSI, NBR, PSRI, CIre, RENDVI, IRECI, S2REP) × 4 | 36 |
| `l8_raw` | L8 bands × 4 seasons | 24 |
| `l8_idx` | L8 indices (NDVI, GNDVI, SAVI) × 4 | 12 |
| `spectral` | Spectral Eng (bands, PCA5, ratios) × 4 | 100 |
| `glcm` | GLCM textures (asm, contrast, ent, idm) × 2ch × 4 | 32 |
| `glcm_extra` | GLCM cross-channel ratio + temporal delta | 24 |
| `topo` | DEM, slope, aspect, TWI, TPI, curvatures | 8 |
| `climate` | MAT, MAP, GS-temp, GS-precip | 4 |
| `temporal` | ts_*_mean/std/cv/slope | 100 |
| `delta` | Δ seasonal + RANGE | 80 |
| `cross_sensor` | CS S2/L8 ratio + diff | 24 |
| **TOTAL** | | **~512** |

### 1.2 Таргеты
- `ph`, `k`, `p`, `hu`, `s`, `no3`
- Обучается **отдельная модель** на каждый таргет (multi-output не используем — разные оптимальные признаки)

### 1.3 Исключаемые колонки (не признаки)
```python
EXCLUDE_COLS = [
    # Мета
    "id", "year", "farm", "field_name", "grid_id",
    "centroid_lon", "centroid_lat", "geometry_wkt",
    "protocol_number", "analysis_date", "sampling_date",
    # Таргеты (все 12 — исключаем все при обучении на один)
    "ph", "k", "p", "hu", "s", "no3",
    "zn", "mo", "fe", "mg", "mn", "cu", "soc",
]
```

---

## 2. Feature Selection Strategy (3-stage pipeline)

### Stage 1: Pre-filter by variance (быстрый отсев)
- Удалить признаки с `std < 1e-5` (константы, заполненные NaN → 0)
- Удалить признаки с `NaN% > 30%` (мало данных для надёжной оценки)
- Ожидаемый результат: ~500 → ~450 признаков

### Stage 2: Correlation-based deduplication (VIF / collinearity control)
**Проблема:** многие признаки сильно коррелируют (NDVI ≈ GNDVI ≈ SAVI в одном сезоне, r>0.95).
Оставлять все — не ошибка для RF (он устойчив к мультиколлинеарности), но:
- Размывает важность признаков (importance сплитуется между коррелятами)
- Замедляет обучение
- Затрудняет интерпретацию для статьи

**Подход:** Иерархическая кластеризация по Spearman ρ:
```
1. Compute Spearman correlation matrix между всеми признаками
2. Иерархическая кластеризация (complete linkage, threshold = 0.1 = ρ > 0.9)
3. Из каждого кластера берём один представитель:
   - Признак с max |ρ| с таргетом (target-aware selection)
   - Если ничья → берём с наименьшим % NaN
4. Результат: ~80–120 "canonical" признаков (оценка)
```
- Реализация: `scipy.cluster.hierarchy` + `pandas.DataFrame.corr(method='spearman')`
- Делается **отдельно для каждого таргета** (canonical set зависит от таргета)

### Stage 3: RF-based importance ranking (финальный отбор)
После Stage 2 (~80–120 признаков) обучаем RF на всех данных:
- `RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=42)`
- Получаем `feature_importances_` (mean decrease impurity, MDI)
- **Дополнительно:** permutation importance на OOF (более надёжная оценка)
- Отбираем топ-20 признаков по `(MDI + permutation_importance) / 2`
- Эти топ-20 — **финальный признаковый набор для итоговой модели**

---

## 3. Модель RF — архитектура и гиперпараметры

### 3.1 Базовый вариант
```python
from sklearn.ensemble import RandomForestRegressor

RF_PARAMS = {
    "n_estimators":   500,      # достаточно для стабилизации OOB error
    "max_features":   "sqrt",   # стандарт для регрессии (√n_features)
    "min_samples_leaf": 3,      # защита от переобучения на малой выборке
    "max_depth":      None,     # деревья растут до чистоты → regularize через min_samples_leaf
    "n_jobs":         -1,       # параллельно на всех ядрах
    "random_state":   42,
    "oob_score":      True,     # Out-of-bag оценка (бесплатная валидация)
}
```

### 3.2 Гиперпараметрический поиск (опционально, после базового запуска)
Если базовый ρ_cv < 0.5 → тюнинг через `sklearn.model_selection.RandomizedSearchCV`:
```python
PARAM_DIST = {
    "n_estimators":     [300, 500, 800],
    "max_features":     ["sqrt", 0.33, 0.5],
    "min_samples_leaf": [1, 3, 5, 10],
    "max_depth":        [None, 15, 20, 30],
}
# CV: используем тот же LOFO (не random split!)
# n_iter=30, scoring='r2', refit=True
```

---

## 4. Кросс-валидация (Spatial LOFO-CV)

### Стратегия: Leave-One-Field-Out
```
Для каждого из 92 уникальных полей (field_name):
  1. test  = все точки с farm_df["field_name"] == test_field
  2. train = все остальные точки (независимо от фермы/года)
  3. Обучаем RF на train → предсказываем test → сохраняем OOF предсказания
4. Метрики на OOF: ρ_cv (Spearman), RMSE_cv, MAE_cv, R²_cv
5. Bootstrap CI (n=500, seed=42): 95% CI для ρ_cv
```

**Почему LOFO, а не random K-fold:**
- Пространственная автокорреляция почвы: точки одного поля дают оптимистичные оценки при случайном разбиении
- LOFO гарантирует независимость test от train по пространству

### Оценка пространственного лага (Moran's I)
- Перед CV: проверяем Moran's I для таргетов → подтверждаем spatial leakage риск
- После CV: сравниваем ρ_train vs ρ_cv → "optimism" = переобучение

---

## 5. Feature Importance — для публикации

### 5.1 MDI (Mean Decrease Impurity)
- `model.feature_importances_` → нормировано, сумма = 1
- Быстро, но смещено в сторону высококардинальных признаков

### 5.2 Permutation Importance (SHAP-style, но быстрее)
- Перемешиваем один признак → измеряем падение ρ_cv
- Надёжнее MDI, честнее

### 5.3 SHAP values (опционально, для детального анализа)
- `shap.TreeExplainer(model)` → бесплатно для RF деревьев
- SHAP summary plot → видно знак (+ или −) и распределение влияния
- **Ключевое для статьи:** SHAP выявляет нелинейность и межпризнаковые взаимодействия

### 5.4 Визуализация для статьи
1. **Grouped bar chart:** суммарная важность по группам (s2_idx > temporal > glcm > …)
2. **Top-20 horizontal bar chart:** лучшие индивидуальные признаки с CI
3. **SHAP beeswarm plot:** для топ-10 признаков самого интерпретируемого таргета (pH)
4. **Comparison table:** ρ_train vs ρ_cv + CI по всем таргетам

---

## 6. Выходные файлы (план)

```
math_statistics/output/rf/
├── rf_report.txt                    # Текстовый отчёт: метрики LOFO-CV по всем таргетам
├── rf_oof_predictions.csv           # OOF предсказания: element, y_true, y_pred, field_name
├── rf_feature_importance.csv        # MDI + permutation importance для топ-признаков
├── rf_cv_summary.png                # Сводный барчарт: ρ_train vs ρ_cv + CI
├── rf_scatter_{element}.png         # OOF scatter plot для каждого таргета
├── rf_importance_{element}.png      # Feature importance plot для каждого таргета
├── rf_shap_{element}.png            # SHAP beeswarm (если shap установлен)
└── rf_feature_selection_log.csv     # Лог отбора признаков по этапам
```

---

## 7. Научная интерпретация (для Methods секции статьи)

### 7.1 Что писать в Methods
```
We employed a Random Forest regression (Breiman, 2001) as the primary ML estimator
for Digital Soil Mapping. For each soil property (pH, K, P, Humus, S, NO₃),
a separate RF model was trained on field-level spectral and derived covariates.

Feature engineering encompassed: (i) 7 Sentinel-2 spectral indices across 4 phenological
seasons (spring, summer, late summer, autumn); (ii) 9 additional S2 spectral indices
(NDWI, NDMI, MSI, NBR, PSRI, CIre, RENDVI, IRECI, S2REP); (iii) temporal statistics
(seasonal mean, std, CV, linear slope) for all spectral indices; (iv) cross-sensor ratios
between S2 (10m) and Landsat-8 (30m); (v) GLCM texture features (contrast, entropy, IDM,
ASM) with cross-channel ratios; and (vi) seasonal delta features encoding phenological
dynamics. The complete feature space comprised ~536 candidate features.

Feature selection employed a 3-stage pipeline: (1) variance filtering (std < 1e-5,
NaN > 30%); (2) hierarchical correlation clustering (Spearman ρ > 0.90 threshold,
target-aware representative selection); (3) RF-based permutation importance ranking,
retaining the top-20 features per target element.

Model performance was evaluated via Spatial Leave-One-Field-Out Cross-Validation (LOFO-CV),
wherein each of the 92 study fields was sequentially excluded from training. This ensures
spatial independence of test sets and prevents spatial leakage inflating performance metrics.
Bootstrap confidence intervals (n=500, 95% CI) for the Spearman correlation coefficient
were computed by resampling out-of-fold predictions.
```

### 7.2 Ожидаемые ρ_cv диапазоны (ориентировочно)
| Таргет | Лучший однопредикторный ρ | Ожидаемый RF ρ_cv | Уверенность |
|--------|--------------------------|-------------------|-------------|
| pH | −0.732 (l8_GNDVI_spring) | 0.70–0.80 | высокая |
| K | −0.478 (s2_BSI_spring) | 0.45–0.60 | средняя |
| P | +0.370 (spectral_PCA5) | 0.35–0.50 | средняя |
| Humus | −0.362 (spectral_PCA5) | 0.35–0.50 | средняя |
| S | +0.383 (l8_SAVI_lsm) | 0.40–0.55 | средняя |
| NO3 | −0.419 (l8_SAVI_spring) | 0.40–0.55 | средняя |

---

## 8. Скрипты к созданию (порядок)

> RF писать ещё не нужно. Ниже — план для следующей сессии.

| Порядок | Файл | Что делает |
|---------|------|-----------|
| 1 | `approximated/build_rf_dataset.py` | Сливает enriched + delta → rf_dataset.csv (536 cols) |
| 2 | `approximated/rf_feature_selection.py` | Stage 1–3 отбора признаков; выдаёт selected_features_{element}.json |
| 3 | `approximated/rf_train_cv.py` | Обучение RF + LOFO-CV + bootstrap CI; выдаёт rf_oof_predictions.csv |
| 4 | `approximated/rf_importance.py` | MDI + permutation + SHAP plots |
| 5 | `approximated/rf_maps.py` | Pixel-level карты предсказания (интеграция в pixel_geo_approx.py) |

---

## 9. Зависимости (pip install)

```bash
pip install scikit-learn scipy numpy pandas matplotlib
pip install shap          # для SHAP values (опционально)
pip install lightgbm      # альтернатива RF (быстрее, часто точнее)
```

**Проверить наличие:**
```bash
python -c "import sklearn; print(sklearn.__version__)"
python -c "import shap; print(shap.__version__)"
```

---

## 10. Критические решения (нужно подтвердить перед написанием кода)

| Решение | Вариант A (рекомендуемый) | Вариант B |
|---------|--------------------------|-----------|
| Единый датасет | enriched + delta (536 features) | только full_dataset (272) |
| Feature selection | 3-stage (variance→corr→RF importance) | manual top-20 по ρ |
| CV стратегия | LOFO по field_name | GroupKFold(n_groups=10) |
| Нормализация | Не нужна для RF | StandardScaler для сравнения с Ridge |
| Обработка NaN | SimpleImputer(strategy='median') | удалить строки с NaN в любом признаке |
| Таргеты | 6 отдельных моделей | Multi-output RF |
| Baseline | Ridge (уже есть) | LinearRegression |
| SHAP | Да (для топ-таргета pH) | Нет (только MDI) |
