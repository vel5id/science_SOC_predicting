#!/bin/bash
cd "$(dirname "$0")/.."

# GLCM Texture Extraction
echo "========================================"
echo "GLCM Texture Extraction"
echo "========================================"
uv run python -m src.s12_glcm
echo
echo "GLCM extraction complete!"
