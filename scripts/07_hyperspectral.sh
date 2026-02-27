#!/bin/bash
cd "$(dirname "$0")/.."

# Investigate hyperspectral data availability
echo "========================================"
echo "Hyperspectral Data Investigation"
echo "========================================"
uv run python -m src.s07_hyperspectral
echo
echo "========================================"
echo "Hyperspectral investigation complete!"
echo "========================================"
