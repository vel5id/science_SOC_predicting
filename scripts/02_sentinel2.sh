#!/bin/bash
cd "$(dirname "$0")/.."

# Extract Sentinel-2 features
echo "========================================"
echo "Sentinel-2 Feature Extraction"
echo "========================================"
uv run python -m src.s02_sentinel2
echo
echo "========================================"
echo "Sentinel-2 extraction complete!"
echo "========================================"
