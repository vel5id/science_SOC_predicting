#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"

echo "============================================================"
echo " ML Pipeline: ResNet Transfer Learning + SoilGrids Baseline"
echo "============================================================"
echo

# ── Step 1: Transfer Learning ResNet ────────────────────────────────────────
echo "[1/2] transfer_learning_resnet.py ..."
echo
"$PYTHON" "$PROJECT_DIR/ML/transfer_learning_resnet.py"
echo
echo "[OK] transfer_learning_resnet.py завершён."
echo

# ── Step 2: SoilGrids Baseline ──────────────────────────────────────────────
echo "[2/2] soilgrids_baseline.py ..."
echo
"$PYTHON" "$PROJECT_DIR/ML/soilgrids_baseline.py"
echo
echo "[OK] soilgrids_baseline.py завершён."
echo
echo "============================================================"
echo " [DONE] Оба скрипта выполнены успешно."
echo "============================================================"
