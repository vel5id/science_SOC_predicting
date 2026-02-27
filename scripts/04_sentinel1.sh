#!/bin/bash
cd "$(dirname "$0")/.."

# Extract Sentinel-1 features
echo "========================================"
echo "Sentinel-1 SAR Feature Extraction"
echo "========================================"
uv run python -m src.s04_sentinel1
echo
echo "========================================"
echo "Sentinel-1 extraction complete!"
echo "========================================"
