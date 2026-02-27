#!/bin/bash
cd "$(dirname "$0")/.."

# Extract Landsat 8 features
echo "========================================"
echo "Landsat 8 Feature Extraction"
echo "========================================"
uv run python -m src.s03_landsat8
echo
echo "========================================"
echo "Landsat 8 extraction complete!"
echo "========================================"
