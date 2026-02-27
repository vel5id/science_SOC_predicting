#!/bin/bash
cd "$(dirname "$0")/.."

# Extract SoilGrids 250m v2.0 features (sand/silt/clay/soc/ph/cec/bdod/N)
echo "========================================"
echo "SoilGrids 250m - Soil Texture ^& Properties"
echo "========================================"
uv run python -m src.s06_soil_maps
echo
echo "========================================"
echo "SoilGrids extraction complete!"
echo "========================================"
