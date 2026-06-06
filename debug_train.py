"""
Debug script to diagnose the backward pass error.
Runs only the first lap, first epoch, with detailed logging.
"""

import sys
import torch

from cdi.config import CDIConfig
from cdi.engine import CDIEngine
from cdi.tokenizer import CDITokenizer
from dataset import download_wikitext, make_dataloader
from train import run_lm_epoch

print("="*70)
print("🔬 CDI TRAINING DEBUG SESSION")
print("="*70)
print("\nThis script will run 1 batch with detailed diagnostic output")
print("to identify where the backward pass error occurs.\n")

# Configure for tiny test
config = CDIConfig.tiny()
config.observation_dim = 32
config.output_dim = 32
config.validate()

print(f"Config: {config.n_points} points, {config.total_state_dim} state dim\n")

# Tokenizer
print("📚 Loading tokenizer...")
tokenizer = CDITokenizer(
    embed_dim=config.observation_dim,
    max_len=config.n_points,
    dtype=config.dtype,
)
print(f"✓ Vocab size: {tokenizer.vocab_size:,}\n")

# Dataset
print("📊 Loading dataset...")
train_ds = download_wikitext(seq_len=config.n_points)
train_loader = make_dataloader(train_ds, batch_size=config.batch_size, shuffle=True)
print(f"✓ Dataset loaded: {len(train_ds)} samples\n")

# Build engine
print("🏗️  Building CDI engine...")
engine = CDIEngine(config)
engine.build()
print(f"✓ Engine built")
print(f"  • Engine params: {sum(p.numel() for p in engine.get_parameters()):,}")
print(f"  • Tokenizer params: {sum(p.numel() for p in tokenizer.get_parameters()):,}")
print(f"  • Spectral gap: {engine.laplacian.spectral_gap().item():.6f}\n")

# Optimizer
all_params = engine.get_parameters() + tokenizer.get_parameters()
optimizer = torch.optim.Adam(all_params, lr=config.learning_rate)
print(f"✓ Optimizer created: Adam with lr={config.learning_rate}\n")

# Run ONE epoch with debug enabled, max 2 batches
print("="*70)
print("🚀 STARTING TRAINING WITH DEBUG OUTPUT")
print("="*70)

try:
    metrics = run_lm_epoch(
        engine, tokenizer, train_loader, optimizer,
        max_batches=2,  # Only 2 batches
        debug=True,     # Enable full diagnostic output
    )
    
    print("\n" + "="*70)
    print("✅ SUCCESS! Training completed without errors")
    print("="*70)
    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  • {key}: {value}")
        
except Exception as e:
    print("\n" + "="*70)
    print("❌ ERROR OCCURRED")
    print("="*70)
    print(f"\nException type: {type(e).__name__}")
    print(f"Exception message: {str(e)}")
    print("\n" + "="*70)
    import traceback
    traceback.print_exc()
    sys.exit(1)
