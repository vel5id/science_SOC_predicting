#!/bin/bash
cd "$(dirname "$0")/.."

# Run pytest test suite
echo "========================================"
echo "Running Test Suite"
echo "========================================"
uv run pytest tests/ -v
echo
echo "========================================"
echo "Test run complete!"
echo "========================================"
