#!/bin/bash
set -e

echo "=================================================="
echo "DCSS-CDI Production GPU Training & Evaluation Runner"
echo "=================================================="

# Ensure the root directory is in the Python path
export PYTHONPATH=$PYTHONPATH:.

# Check for HF_TOKEN
if [ -n "$HF_TOKEN" ]; then
    echo "[INFO] HF_TOKEN detected in environment."
    export HF_TOKEN=$HF_TOKEN
else
    echo "[INFO] No HF_TOKEN detected. Public datasets will be accessed without authentication."
fi

echo "[1/4] Installing/Updating dependencies..."
python3 -m pip install --upgrade -r requirements.txt
# Ensure torch is installed with CUDA support if not present
python3 -c "import torch; print(f'Torch version: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "[2/4] Verifying mathematical test suite and syntax..."
find cdi/v3 -name "*.py" -exec python3 -m py_compile {} +
pytest -q

echo "[3/4] Launching GPU training & fine-tuning pipeline..."
# Run as a module to ensure correct package resolution
python3 -m cdi.v3.production.train_production --config benchmarks/configs/production_large.json

echo "[4/4] Finalizing results..."
if [ -f "results/production/production_report.json" ]; then
    echo "SUCCESS: Production run completed."
    cat results/production/production_report.json
else
    echo "ERROR: Production report not found."
    exit 1
fi
