#!/bin/bash
set -e

echo "=================================================="
echo "DCSS-CDI Production GPU Training & Evaluation Runner"
echo "=================================================="

echo "[1/4] Installing dependencies (PyTorch, Hugging Face Datasets)..."
# Use existing torch if available, otherwise install
python3 -c "import torch" 2>/dev/null || pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets pytest

echo "[2/4] Verifying mathematical test suite and syntax..."
# Syntax check first
find cdi/v3 -name "*.py" -exec python3 -m py_compile {} +
# Run repository tests
pytest -q

echo "[3/4] Launching GPU training & fine-tuning pipeline on WikiText-103 & SciQ..."
python3 -c "from cdi.v3.production.train_production import run_production_pipeline; run_production_pipeline()"

echo "[4/4] Finalizing results..."
if [ -f "results/production/production_report.json" ]; then
    echo "SUCCESS: Production run completed."
    cat results/production/production_report.json
else
    echo "ERROR: Production report not found."
    exit 1
fi
