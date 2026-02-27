#!/bin/bash
cd "$(dirname "$0")/.."

# Spectral Engineering (Post-processing)
echo "========================================"
echo "Spectral Engineering"
echo "========================================"
uv run python -m src.s11_spectral_eng
echo
echo "Spectral engineering complete!"
