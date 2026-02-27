# Digital Soil Mapping — Northern Kazakhstan

Predicting six agrochemical soil properties (pH, SOC, NO₃, P₂O₅, K₂O, S) from multimodal satellite imagery using machine learning and deep learning.

> **Paper**: *Digital Soil Mapping of Steppe Zone in Northern Kazakhstan: Predicting Agrochemical Soil Properties Using Multimodal Satellite Data and ML/DL Methods* (in preparation)

---

## Overview

This repository contains the full data processing pipeline and modelling code for a digital soil mapping (DSM) study covering 20 farms (81 fields, 1 085 soil samples) in northern Kazakhstan (2022–2023).

**Key results:**
- 11 ML models + 3 DL architectures (ResNet-18, ConvNeXt+SE) compared under 3 spatial validation strategies
- Ensemble models (RF, GBDT, CatBoost) consistently outperform CNNs on tabular features
- pH is the most predictable property: ρ = 0.750 (RF, Farm-LOFO), RPD = 1.62
- Strict Farm-LOFO validation reveals up to 70% metric inflation for NO₃ compared to Field-LOFO

## Project Structure

```
science-article/
│
├── scripts/                    # GEE feature extraction pipeline (s01–s12)
│   ├── s01_temperature.py      #   ERA5-Land climate windows
│   ├── s02_sentinel2.py        #   Sentinel-2 spectral features
│   ├── s03_landsat8.py         #   Landsat-8 spectral features
│   ├── s04_sentinel1.py        #   Sentinel-1 SAR backscatter
│   ├── s05_topography.py       #   SRTM DEM derivatives
│   ├── s06_soil_maps.py        #   SoilGrids v2.0 (excluded from modelling)
│   ├── s08_merge_features.py   #   Merge all features → master dataset
│   ├── s09_climate.py          #   MAT, MAP, GS_temp, GS_precip
│   ├── s11_spectral_eng.py     #   Band ratios, PCA, EVI
│   └── s12_glcm.py             #   GLCM texture features
│
├── src/                        # Shared utilities
│   ├── db_utils.py             #   Database I/O
│   └── file_utils.py           #   File helpers
│
├── math_statistics/            # Statistical analysis (Article 1)
│   ├── run_all.py              #   Orchestrator
│   ├── descriptive_stats.py    #   Descriptive statistics, Shapiro-Wilk, Kruskal-Wallis
│   ├── correlation_analysis.py #   Spearman correlations, BH correction
│   ├── variance_decomposition.py # ICC, between-farm vs within-farm
│   └── ...
│
├── ML/                         # Predictive modelling (Article 2)
│   ├── data_loader.py          #   Load master_dataset.csv
│   ├── train_unified_ml.py     #   11 ML models, Field-LOFO & Farm-LOFO
│   ├── train_rf_gridsearch_lofo.py  # RF with per-fold MDI + GridSearchCV
│   ├── farm_lofo_all_models.py #   Farm-LOFO for all 11 models
│   ├── train_cnn.py            #   ResNet-18 (scratch & ImageNet TL)
│   ├── train_multiseason_convnext.py  # ConvNeXt + SE, 54-channel
│   ├── rf_vs_cnn_spatial_split.py     # Fair RF vs CNN comparison
│   ├── soilgrids_baseline.py   #   SoilGrids v2.0 baseline
│   ├── friedman_test.py        #   Friedman + Nemenyi post-hoc
│   ├── deep_leakage_audit.py   #   Temporal leakage audit (S)
│   └── ...
│
├── articles/                   # LaTeX source
│   ├── article1_correlations/  #   Statistical analysis paper
│   └── article2_prediction/    #   Prediction paper (main)
│       ├── main.tex
│       └── sections/
│
├── generate_prediction_maps.py # pH/NO₃ spatial prediction maps
├── generate_four_panel_map.py  # 4-panel comparison figure
├── generate_figures.py         # Result figures
├── build_soil_db.py            # Build SQLite from shapefiles
└── pyproject.toml              # Dependencies
```

## Data

| Source | Resolution | Features |
|--------|-----------|----------|
| Sentinel-2 | 10–20 m | Spectral bands, NDVI, EVI, GNDVI, BSI, etc. |
| Landsat-8 | 30 m | Spectral bands, indices |
| Sentinel-1 | 10 m | VV/VH backscatter (SAR) |
| SRTM DEM | 30 m | Elevation, slope, aspect, TWI, curvature |
| ERA5-Land | 0.1° | MAT, MAP, growing season temperature & precipitation |
| SoilGrids v2.0 | 250 m | Extracted but **excluded** from modelling (leakage risk) |

**Soil samples**: 1 085 samples from 20 farms, 6 properties (pH, SOC, NO₃, P₂O₅, K₂O, S). Lab analysis by GOST standards. Soil data is not included (commercially sensitive).

## Validation Strategies

| Strategy | Folds | Strictness | Purpose |
|----------|-------|-----------|---------|
| **Field-LOFO** | 81 | Low | Within-farm prediction |
| **Spatial split** | 65/6/10 farms | Medium | Geographic extrapolation |
| **Farm-LOFO** | 20 | High | True out-of-farm generalization |

## Quickstart

```bash
# 1. Clone
git clone https://github.com/vel5id/science_SOC_predicting.git
cd science_SOC_predicting

# 2. Create environment (requires Python ≥ 3.12)
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. ML environment (separate venv with PyTorch)
cd ML
python -m venv .venv
source .venv/bin/activate
pip install scikit-learn xgboost catboost scipy pandas numpy matplotlib tqdm

# 4. Run ML pipeline (requires data/features/master_dataset.csv)
python train_unified_ml.py
```

> **Note**: The feature extraction pipeline (scripts/s01–s12) requires Google Earth Engine authentication and the soil samples database, which is not included in this repository.

## Key Models

| Model | Type | Best for |
|-------|------|----------|
| RF (Random Forest) | Ensemble | pH, NO₃, K₂O |
| CatBoost | Gradient boosting | SOC |
| XGBoost | Gradient boosting | S |
| SVR | Kernel | P₂O₅ |
| ResNet-18 | CNN (18-ch patches) | Ablation studies |
| ConvNeXt + SE | CNN (54-ch multi-season) | NO₃ (+36%) |

## Citation

```
TBD — paper in preparation
```

## License

This project is provided for academic and research purposes. See the repository for details.
