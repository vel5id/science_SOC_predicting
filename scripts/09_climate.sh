#!/bin/bash
cd "$(dirname "$0")/.."

# Climate Feature Extraction (ERA5-Land)
echo "========================================"
echo "Climate Feature Extraction"
echo "========================================"
uv run python -m src.s09_climate
echo
echo "Climate extraction complete!"
