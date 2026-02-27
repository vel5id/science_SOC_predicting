#!/bin/bash
cd "$(dirname "$0")/.."

# Extract topographic features
echo "========================================"
echo "Topographic Feature Extraction"
echo "========================================"
uv run python -m src.s05_topography
echo
echo "========================================"
echo "Topographic extraction complete!"
echo "========================================"
