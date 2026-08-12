#!/bin/bash
set -e

echo "=================================================="
echo "DCSS-CDI Production GPU Training & Evaluation Runner"
echo "=================================================="

# Check for HF_TOKEN
if [ -n "$HF_TOKEN" ]; then
    echo "[INFO] HF_TOKEN detected in environment."
    # We'll pass this to the python environment
    export HF_TOKEN=$HF_TOKEN
else
    echo "[INFO] No HF_TOKEN detected. Public datasets will be accessed without authentication."
fi

echo "[1/4] Installing/Updating dependencies..."
# In Colab, we want to ensure we have the latest datasets and huggingface_hub
pip install --upgrade datasets huggingface_hub pytest
# Ensure torch is installed with CUDA support if not present
python3 -c "import torch; print(f'Torch version: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "[2/4] Verifying mathematical test suite and syntax..."
find cdi/v3 -name "*.py" -exec python3 -m py_compile {} +
pytest -q

echo "[3/4] Launching GPU training & fine-tuning pipeline..."
# The token is picked up from the environment by the python script
python3 -c "from cdi.v3.production.train_production import run_production_pipeline; run_production_pipeline()"

echo "[4/4] Finalizing results..."
if [ -f "results/production/production_report.json" ]; then
    echo "SUCCESS: Production run completed."
    cat results/production/production_report.json
else
    echo "ERROR: Production report not found."
    exit 1
fi
