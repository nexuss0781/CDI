#!/bin/bash
set -e

echo "=================================================="
echo "DCSS-CDI Production GPU Training & Evaluation Runner"
echo "=================================================="

echo "[1/4] Installing/Updating dependencies..."
pip install --upgrade datasets pytest
# Ensure torch is installed with CUDA support if not present
python3 -c "import torch; print(f'Torch version: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "[2/4] Verifying mathematical test suite and syntax..."
find cdi/v3 -name "*.py" -exec python3 -m py_compile {} +
pytest -q

echo "[3/4] Launching GPU training & fine-tuning pipeline..."
# We use real datasets now; synthetic fallback is removed to ensure real training.
python3 -c "from cdi.v3.production.train_production import run_production_pipeline; run_production_pipeline()"

echo "[4/4] Finalizing results..."
if [ -f "results/production/production_report.json" ]; then
    echo "SUCCESS: Production run completed."
    cat results/production/production_report.json
else
    echo "ERROR: Production report not found."
    exit 1
fi
