# Code Future Roadmap

Development plans and future directions for the Digital Soil Mapping project.

---

## 1. Prediction Maps Integration (M5)

### Goal
Generate and integrate pixel-level soil property prediction maps into Article 2.

### Current State
- **Scripts**: `approximated/` contains the full pipeline for pixel-level maps:
  - Ridge (geo-spectral): `pixel_geo_approx.py`
  - RF (40+ features): `rf_pixel_maps.py`, `rf_pixel_geo_maps.py`
  - Ensemble (Stack + Kriging): `rf_ensemble_maps.py`
- **Farm**: "Agro Parasat" (UTM 41N, 2023)
- **Pixel data**: CSV with ~32K inside-pixels + TIFF `s2_2023-06-05_B4B8B3B5B11.tif`
- **Trained RF**: `math_statistics/output/rf/rf_models/rf_{element}.pkl`

### Steps
1. **Generate pixel data**: `python approximated/pixel_ndvi_real.py` -> `pixels_Agro_Parasat_2023_real.csv`
2. **Generate RF maps**: `python approximated/rf_pixel_geo_maps.py` -> per-element PNG maps
3. **Select elements for demo**: pH (best, rho=0.750), P2O5 (moderate, rho=0.571), NO3 (weak, rho=0.232)
4. **Add figure to article**: 3x1 subplot with RGB background + colorbar + field boundaries
5. **Update article text**: remove limitation #9, add Results subsection, update Discussion

### Requirements
- Use 15 MDI-selected features (from `data/features/selected/`) for consistency with main experiments
- Retrain RF models on 15 MDI features before map generation if needed

---

## 2. Random Forest Training Plan

### Goal
Train Random Forest models for each of the 6 soil properties (pH, K, P, Humus, S, NO3).

### Datasets
| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `data/features/full_dataset.csv` | 1121 | 272 | Original (S2, L8, Spectral, GLCM, Topo, Climate) |
| `data/features/enriched_dataset.csv` | 1121 | 456 | +184 extras (NDWI/NDMI/MSI/..., temporal stats, CS ratios) |
| `data/features/delta_dataset.csv` | 1121 | 352 | +80 delta (seasonal delta, RANGE) |

### Feature Groups (~512 total)
| Group | Features |
|-------|----------|
| S2 raw bands | 40 |
| S2 spectral indices | 28 |
| S2 extra indices | 36 |
| L8 raw bands | 24 |
| L8 indices | 12 |
| Spectral Engineering | 100 |
| GLCM textures | 32 |
| GLCM extras | 24 |
| Topography | 8 |
| Climate | 4 |
| Temporal statistics | 100 |
| Seasonal deltas | 80 |
| Cross-sensor ratios | 24 |

### Feature Selection Pipeline (3-stage)
1. **Variance filter**: Remove features with std < 1e-5 or NaN > 30%
2. **Correlation clustering**: Hierarchical clustering (Spearman rho > 0.90), target-aware representative selection
3. **RF importance ranking**: MDI + permutation importance, retain top-20 per target

### Model Architecture
```python
RF_PARAMS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 3,
    "max_depth": None,
    "n_jobs": -1,
    "random_state": 42,
    "oob_score": True,
}
```

### Validation
- Spatial Leave-One-Field-Out Cross-Validation (LOFO-CV)
- Bootstrap 95% CI (n=500, seed=42) for Spearman rho
- Compare rho_train vs rho_cv for optimism estimation

### Expected Performance
| Target | Best single-predictor rho | Expected RF rho_cv |
|--------|---------------------------|---------------------|
| pH | -0.732 | 0.70-0.80 |
| K | -0.478 | 0.45-0.60 |
| P | +0.370 | 0.35-0.50 |
| Humus | -0.362 | 0.35-0.50 |
| S | +0.383 | 0.40-0.55 |
| NO3 | -0.419 | 0.40-0.55 |

### Scripts to Create
| Order | File | Purpose |
|-------|------|---------|
| 1 | `approximated/build_rf_dataset.py` | Merge enriched + delta -> rf_dataset.csv |
| 2 | `approximated/rf_feature_selection.py` | 3-stage feature selection |
| 3 | `approximated/rf_train_cv.py` | RF training + LOFO-CV + bootstrap CI |
| 4 | `approximated/rf_importance.py` | MDI + permutation + SHAP plots |
| 5 | `approximated/rf_maps.py` | Pixel-level prediction maps |
