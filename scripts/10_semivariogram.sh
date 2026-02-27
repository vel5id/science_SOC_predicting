#!/bin/bash
cd "$(dirname "$0")/.."

# Semivariogram Analysis
echo "========================================"
echo "Semivariogram Analysis"
echo "========================================"
uv run python -m src.s10_semivariogram
echo
echo "Semivariogram analysis complete!"
