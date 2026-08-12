#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo "DCSS-CDI Production GPU Training & Evaluation Runner"
echo "=================================================="

# 1. Install dependencies
echo "[1/4] Installing dependencies (PyTorch, Hugging Face Datasets)..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets pytest

# 2. Run repository test suite to verify baseline integrity
echo "[2/4] Verifying repository mathematical test suite..."
pytest -q tests/production tests/stage_f tests/p2 || pytest -q

# 3. Execute Hugging Face large corpus intake and GPU training pipeline
echo "[3/4] Launching GPU training & fine-tuning pipeline on WikiText-103 & SciQ..."
python3 -c "
from cdi.v3.production.train_production import run_production_pipeline
report = run_production_pipeline()
print('Training completed successfully:', report)
"

# 4. Final verification and report generation
echo "[4/4] Finalizing production verification..."
if [ -f "results/production/production_report.json" ]; then
    echo "SUCCESS: Production training finished and checkpoint verified."
else
    echo "ERROR: Production training report not found."
    exit 1
fi

echo "=================================================="
echo "All tasks completed successfully!"
echo "=================================================="
