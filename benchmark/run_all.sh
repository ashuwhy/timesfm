#!/bin/bash
# Run all TimesFM 2.5 benchmarks
# Usage: bash benchmark/run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${PROJECT_DIR}/.venv/bin/python"

echo "=============================================="
echo "TimesFM 2.5 200M Benchmark Suite"
echo "=============================================="
echo ""

# 1. Install dependencies
echo "[1/3] Installing dependencies..."
uv pip install --python .venv/bin/python gluonts datasetsforecast utilsforecast pandas scipy 2>/dev/null || \
  $PYTHON -m pip install gluonts datasetsforecast utilsforecast pandas scipy
echo "Dependencies installed."
echo ""

# 2. ETT benchmark (faster)
echo "[2/3] Running ETT long-horizon benchmark..."
$PYTHON benchmark/run_base_ett.py "$@"
echo ""

# 3. Monash benchmark (slower)
echo "[3/3] Running Monash extended benchmark..."
$PYTHON benchmark/run_base_monash.py "$@"
echo ""

echo "=============================================="
echo "All benchmarks complete!"
echo "Results:"
echo "  ETT:    results/ett/results.json"
echo "  Monash: results/monash/<run_id>/results.csv"
echo "=============================================="
