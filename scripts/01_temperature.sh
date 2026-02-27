#!/bin/bash
cd "$(dirname "$0")/.."

# Extract ERA5 temperature data
echo "========================================"
echo "ERA5 Temperature Extraction"
echo "========================================"
uv run python -m src.s01_temperature
echo
echo "========================================"
echo "Temperature extraction complete!"
echo "========================================"
