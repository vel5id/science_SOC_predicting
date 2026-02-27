#!/bin/bash
cd "$(dirname "$0")/.."

# Merge all features
echo "========================================"
echo "Feature Merging and Database Integration"
echo "========================================"
uv run python -m src.s08_merge_features
echo
echo "========================================"
echo "Feature merging complete!"
echo "========================================"
