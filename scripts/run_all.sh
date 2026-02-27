#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# Run all extraction steps sequentially
echo "========================================"
echo "Satellite Data Extraction Pipeline"
echo "========================================"
echo
echo "This will run all extraction steps:"
echo "1. Setup (install dependencies)"
echo "2. ERA5 Temperature"
echo "3. Sentinel-2"
echo "4. Landsat 8"
echo "5. Sentinel-1"
echo "6. Topography (DEM, slope, aspect, TWI, curvature, TPI)"
echo "7. Soil Maps"
echo "8. Hyperspectral Investigation"
echo "9. Climate Covariates (MAT, MAP, GS)"
echo "10. Spectral Engineering (EVI, Band Ratios, PCA)"
echo "11. GLCM Textures"
echo "12. Semivariogram Analysis"
echo "13. Feature Merging"
echo
echo "Press any key to start, or Ctrl+C to cancel..."
# Step 0: Setup
echo
echo "========================================"
echo "Step 0: Installing dependencies..."
echo "========================================"
bash scripts/00_setup.sh
# Step 1: Temperature
echo
echo "========================================"
echo "Step 1: ERA5 Temperature Extraction"
echo "========================================"
uv run python -m src.s01_temperature
if [ $? -ne 0 ]; then
echo "ERROR in temperature extraction!"
    exit 1
fi
# Step 2: Sentinel-2
echo
echo "========================================"
echo "Step 2: Sentinel-2 Feature Extraction"
echo "========================================"
uv run python -m src.s02_sentinel2
if [ $? -ne 0 ]; then
echo "ERROR in Sentinel-2 extraction!"
    exit 1
fi
# Step 3: Landsat 8
echo
echo "========================================"
echo "Step 3: Landsat 8 Feature Extraction"
echo "========================================"
uv run python -m src.s03_landsat8
if [ $? -ne 0 ]; then
echo "ERROR in Landsat 8 extraction!"
    exit 1
fi
# Step 4: Sentinel-1
echo
echo "========================================"
echo "Step 4: Sentinel-1 SAR Feature Extraction"
echo "========================================"
uv run python -m src.s04_sentinel1
if [ $? -ne 0 ]; then
echo "ERROR in Sentinel-1 extraction!"
    exit 1
fi
# Step 5: Topography
echo
echo "========================================"
echo "Step 5: Topographic Feature Extraction"
echo "========================================"
uv run python -m src.s05_topography
if [ $? -ne 0 ]; then
echo "ERROR in topography extraction!"
    exit 1
fi
# Step 6: Soil Maps
echo
echo "========================================"
echo "Step 6: Soil Map Feature Extraction"
echo "========================================"
uv run python -m src.s06_soil_maps
if [ $? -ne 0 ]; then
echo "ERROR in soil map extraction!"
    exit 1
fi
# Step 7: Hyperspectral
echo
echo "========================================"
echo "Step 7: Hyperspectral Data Investigation"
echo "========================================"
uv run python -m src.s07_hyperspectral
if [ $? -ne 0 ]; then
echo "ERROR in hyperspectral investigation!"
    exit 1
fi
# Step 9: Climate
echo
echo "========================================"
echo "Step 9: Climate Covariates Extraction"
echo "========================================"
uv run python -m src.s09_climate
if [ $? -ne 0 ]; then
echo "ERROR in climate extraction!"
    exit 1
fi
# Step 11: Spectral Engineering
echo
echo "========================================"
echo "Step 11: Spectral Engineering (Post-processing)"
echo "========================================"
uv run python -m src.s11_spectral_eng
if [ $? -ne 0 ]; then
echo "ERROR in spectral engineering!"
    exit 1
fi
# Step 12: GLCM
echo
echo "========================================"
echo "Step 12: GLCM Texture Extraction"
echo "========================================"
uv run python -m src.s12_glcm
if [ $? -ne 0 ]; then
echo "ERROR in GLCM extraction!"
    exit 1
fi
# Step 10: Semivariogram (can run independently)
echo
echo "========================================"
echo "Step 10: Semivariogram Analysis"
echo "========================================"
uv run python -m src.s10_semivariogram
if [ $? -ne 0 ]; then
echo "WARNING: Semivariogram analysis failed (non-critical)"
fi
# Step 8: Merge
echo
echo "========================================"
echo "Step 8: Feature Merging"
echo "========================================"
uv run python -m src.s08_merge_features
if [ $? -ne 0 ]; then
echo "ERROR in feature merging!"
    exit 1
fi
echo
echo "========================================"
echo "PIPELINE COMPLETE!"
echo "========================================"
echo
echo "All extraction steps completed successfully."
echo "Final dataset saved to: data\features\full_dataset.csv"
echo "Database updated: data\soil_analysis.db"
echo
