"""
CDI Training Pipeline — Next-Gen LLM
======================================

Interleaved training on real text data:

    for each lap:
        1. TRAIN   on Wikipedia knowledge corpus (wikitext-2)
        2. FINE-TUNE on science QA (SciQ)
        3. TEST    on hand-crafted science questions
        4. LOG     perplexity, δ², spectral gap, etc.
        5. Continue → next lap

This is a LANGUAGE MODEL, not a regression engine.
CDI replaces every transformer component with mathematical structures.

Complexity Constraints (enforced)
---------------------------------
- O(1) per-token point evaluation
- O(n) learning via heat equation Euler steps
- O(n log n) abstraction via spectral sequence
- NO torch.nn modules, NO softmax in the engine
  (logsumexp used ONLY in cross-entropy loss computation)
- NO transformer architecture components

Usage::

    python train.py --config tiny  --laps 3 --lap-epochs 5   # quick test
    python train.py --config small --laps 5 --lap-epochs 15  # standard
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from cdi.config import CDIConfig
from cdi.engine import CDIEngine
from cdi.tokenizer import CDITokenizer
from dataset import (
    download_wikitext,
    download_sciq,
    make_test_set,
    make_dataloader,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Manual Gradient Clipping — NO torch.nn
# ═══════════════════════════════════════════════════════════════════════════

def clip_grad_norm_(params: List[torch.Tensor], max_norm: float) -> float:
    """Pure tensor gradient clipping.  O(n_params)."""
    total_sq = 0.0
    grads = [p for p in params if p.grad is not None]
    for p in grads:
        total_sq += p.grad.data.norm(2).item() ** 2
    total = math.sqrt(total_sq)
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for p in grads:
            p.grad.data.mul_(scale)
    return total


# ═══════════════════════════════════════════════════════════════════════════
#  Training Epoch — Language Modeling
# ═══════════════════════════════════════════════════════════════════════════

def run_lm_epoch(
    engine: CDIEngine,
    tokenizer: CDITokenizer,
    dataloader,
    optimizer: torch.optim.Optimizer,
    rebuild_every: int = 10,
    max_batches: int = None,
) -> Dict[str, float]:
    """One epoch of next-token prediction training.

    For each batch:
        1. Embed token IDs → (batch, seq_len, embed_dim)
        2. CDI forward_sequence_batch → (batch, seq_len, embed_dim)
        3. Cross-entropy loss via weight-tied logits
        4. Backward + clip + step

    Complexity per batch: O(batch × n_points × heat_steps)
    """
    total_loss = 0.0
    total_ce = 0.0
    total_consistency = 0.0
    total_perplexity = 0.0
    total_grad_norm = 0.0
    n_batches = 0

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        # Rebuild operators periodically
        if batch_idx % rebuild_every == 0:
            engine.rebuild_operators()

        optimizer.zero_grad()

        # Embed tokens → (batch, seq_len, embed_dim)
        embeddings = tokenizer.embed(input_ids)  # (batch, seq_len, embed_dim)

        # CDI forward — each manifold point = one token position
        output = engine.forward_sequence_batch(embeddings)  # (batch, seq_len, embed_dim)

        # Cross-entropy loss with weight tying
        loss, loss_dict = engine.compute_lm_loss(
            output, target_ids, tokenizer.embedding
        )

        # Backward
        loss.backward(retain_graph=True)

        # Gradient clipping — no torch.nn
        all_params = engine.get_parameters() + tokenizer.get_parameters()
        grad_norm = clip_grad_norm_(all_params, max_norm=1.0)

        optimizer.step()

        total_loss += loss_dict["total"]
        total_ce += loss_dict["ce"]
        total_consistency += loss_dict["consistency"]
        total_perplexity += loss_dict["perplexity"]
        total_grad_norm += grad_norm
        n_batches += 1

    d = max(n_batches, 1)
    return {
        "loss": total_loss / d,
        "ce": total_ce / d,
        "perplexity": total_perplexity / d,
        "consistency": total_consistency / d,
        "grad_norm": total_grad_norm / d,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_lm(
    engine: CDIEngine,
    tokenizer: CDITokenizer,
    dataloader,
) -> Dict[str, float]:
    """Evaluate perplexity on a dataset."""
    total_ce = 0.0
    n_tokens = 0

    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            embeddings = tokenizer.embed(input_ids)
            output = engine.forward_sequence_batch(embeddings)
            logits = output @ tokenizer.embedding.T  # (B, S, V)

            B, S, V = logits.shape
            logits_flat = logits.reshape(B * S, V)
            targets_flat = target_ids.reshape(B * S)

            log_probs = logits_flat - logits_flat.logsumexp(dim=-1, keepdim=True)
            ce = -log_probs[torch.arange(B * S), targets_flat].sum()

            total_ce += ce.item()
            n_tokens += B * S

    avg_ce = total_ce / max(n_tokens, 1)
    return {
        "ce": avg_ce,
        "perplexity": math.exp(min(avg_ce, 20)),  # cap to avoid overflow
        "n_tokens": n_tokens,
    }


def generate_sample(
    engine: CDIEngine,
    tokenizer: CDITokenizer,
    prompt: str,
    max_tokens: int = 20,
) -> str:
    """Generate text from a prompt using greedy decoding.

    This is a simple autoregressive generation loop.
    """
    ids = tokenizer.encode(prompt)
    seq_len = tokenizer.max_len

    with torch.no_grad():
        for _ in range(max_tokens):
            # Take last seq_len tokens
            window = ids[-seq_len:]
            embeddings = tokenizer.embed(window).unsqueeze(0)  # (1, S, E)
            output = engine.forward_sequence_batch(embeddings)  # (1, S, E)
            logits = output[0, -1] @ tokenizer.embedding.T  # (V,)
            next_id = logits.argmax().item()
            ids = torch.cat([ids, torch.tensor([next_id], dtype=torch.long)])

            # Stop at EOS
            if next_id == tokenizer.hf_tokenizer.eos_token_id:
                break

    return tokenizer.decode_ids(ids)


# ═══════════════════════════════════════════════════════════════════════════
#  INTERLEAVED TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_interleaved_training(
    config: CDIConfig,
    n_laps: int = 5,
    lap_epochs: int = 15,
    ft_epochs: int = 5,
    max_batches_per_epoch: int = 200,
    output_dir: str = "./results",
) -> Dict:
    """
    Interleaved LLM training::

        for lap in 1..n_laps:
            TRAIN      lap_epochs on Wikipedia (wikitext-2)
            FINE-TUNE   ft_epochs  on science QA (SciQ)
            TEST       science questions → log perplexity
            GENERATE   sample outputs
            continue
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    seq_len = config.n_points  # context length = manifold points

    # ── Header ────────────────────────────────────────────────────────
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  𝕮𝔇𝕴 — COHOMODYNAMIC INTELLIGENCE".center(68) + "║")
    print("║" + "  Next-Generation Language Model Training".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"  Context length  : {seq_len} tokens (= n_points)")
    print(f"  State dimension : {config.total_state_dim}")
    print(f"  Laps            : {n_laps} × ({lap_epochs} train + {ft_epochs} ft)")
    print(f"  Complexity      : O(1) reflex | O(n) learn | O(n log n) abstract")
    print()

    # ── 1. Tokenizer ─────────────────────────────────────────────────
    print("─" * 70)
    print("  PHASE 1: Tokenizer + Datasets")
    print("─" * 70)

    embed_dim = config.observation_dim
    tokenizer = CDITokenizer(
        tokenizer_name="gpt2",
        embed_dim=embed_dim,
        max_len=seq_len,
        dtype=config.dtype,
    )
    print(f"  Tokenizer    : GPT-2 (vocab={tokenizer.vocab_size:,})")
    print(f"  Embed dim    : {embed_dim}")

    # ── 2. Datasets ──────────────────────────────────────────────────
    train_ds = download_wikitext(seq_len=seq_len)
    ft_ds = download_sciq(seq_len=seq_len)
    test_ds = make_test_set(seq_len=seq_len)

    train_loader = make_dataloader(train_ds, batch_size=config.batch_size, shuffle=True)
    ft_loader = make_dataloader(ft_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = make_dataloader(test_ds, batch_size=len(test_ds), shuffle=False)

    print(f"\n  Train     : {train_ds}")
    print(f"  Fine-tune : {ft_ds}")
    print(f"  Test      : {test_ds}")

    # ── 3. Build engine ──────────────────────────────────────────────
    print()
    print("─" * 70)
    print("  PHASE 2: Engine Construction")
    print("─" * 70)

    engine = CDIEngine(config)
    engine.build()

    engine_params = engine.get_parameters()
    tok_params = tokenizer.get_parameters()
    all_params = engine_params + tok_params

    n_engine = sum(p.numel() for p in engine_params)
    n_tok = sum(p.numel() for p in tok_params)
    print(f"  Engine params : {n_engine:,}")
    print(f"  Embedding     : {n_tok:,} ({tokenizer.vocab_size} × {embed_dim})")
    print(f"  Total params  : {n_engine + n_tok:,}")
    print(f"  Spectral gap  : {engine.laplacian.spectral_gap().item():.6f}")
    print(f"  Learning time : {engine.heat.learning_time().item():.4f}")

    # ── 4. Optimizer ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(all_params, lr=config.learning_rate)

    # ── 5. Interleaved training ──────────────────────────────────────
    print()
    print("─" * 70)
    print("  PHASE 3: Interleaved Training")
    print("─" * 70)

    history = {"laps": [], "train": [], "ft": []}
    best_perplexity = float("inf")
    global_epoch = 0

    for lap in range(1, n_laps + 1):
        lap_start = time.time()

        print(f"\n  ┌─── LAP {lap}/{n_laps} ──────────────────────────────────────────┐")

        # ── TRAIN on Wikipedia ───────────────────────────────────
        print(f"  │  TRAINING on Wikipedia ({lap_epochs} epochs)")
        for ep in range(1, lap_epochs + 1):
            global_epoch += 1
            t0 = time.time()
            metrics = run_lm_epoch(
                engine, tokenizer, train_loader, optimizer,
                rebuild_every=15, max_batches=max_batches_per_epoch,
            )
            dt = time.time() - t0
            history["train"].append({"global_epoch": global_epoch, "lap": lap, **metrics})

            if ep % max(lap_epochs // 3, 1) == 0 or ep == lap_epochs:
                print(f"  │    Epoch {global_epoch:4d} │ "
                      f"CE={metrics['ce']:.4f} │ "
                      f"PPL={metrics['perplexity']:.1f} │ "
                      f"δ²={metrics['consistency']:.2e} │ "
                      f"∇={metrics['grad_norm']:.3f} │ "
                      f"{dt:.1f}s")

        # ── FINE-TUNE on Science QA ──────────────────────────────
        print(f"  │  FINE-TUNING on SciQ ({ft_epochs} epochs)")
        for ep in range(1, ft_epochs + 1):
            global_epoch += 1
            metrics = run_lm_epoch(
                engine, tokenizer, ft_loader, optimizer,
                rebuild_every=5,
            )
            history["ft"].append({"global_epoch": global_epoch, "lap": lap, **metrics})
            print(f"  │    FT {ep}/{ft_epochs}      │ "
                  f"CE={metrics['ce']:.4f} │ "
                  f"PPL={metrics['perplexity']:.1f} │ "
                  f"δ²={metrics['consistency']:.2e}")

        # ── TEST on science questions ────────────────────────────
        engine.rebuild_operators()
        test_metrics = evaluate_lm(engine, tokenizer, test_loader)

        # Track best
        if test_metrics["perplexity"] < best_perplexity:
            best_perplexity = test_metrics["perplexity"]
            torch.save(
                [p.data.clone() for p in all_params],
                Path(output_dir) / "best_params.pt",
            )

        # Mathematical diagnostics
        diag = {
            "spectral_gap": engine.laplacian.spectral_gap().item(),
            "learning_time": engine.heat.learning_time().item(),
            "harmonic_dim": engine.hodge.harmonic_dimension(),
            "delta_sq": engine.belief.consistency_penalty().item(),
        }

        lap_time = time.time() - lap_start

        lap_record = {
            "lap": lap, "global_epoch": global_epoch,
            **test_metrics, **diag, "lap_time": lap_time,
        }
        history["laps"].append(lap_record)

        print(f"  │  TEST on science questions:")
        print(f"  │    CE         = {test_metrics['ce']:.4f}")
        print(f"  │    Perplexity = {test_metrics['perplexity']:.1f}")
        print(f"  │    λ₁ = {diag['spectral_gap']:.6f}  "
              f"τ = {diag['learning_time']:.4f}  "
              f"ℋ = {diag['harmonic_dim']}  "
              f"δ² = {diag['delta_sq']:.2e}")

        # ── SAMPLE GENERATION ────────────────────────────────────
        print(f"  │  Sample generation:")
        prompts = [
            "Q: What is the basic unit of life? A:",
            "Q: What force keeps planets in orbit? A:",
            "Q: What is the chemical symbol for water? A:",
        ]
        for prompt in prompts:
            try:
                generated = generate_sample(engine, tokenizer, prompt, max_tokens=15)
                display = generated[len(prompt):].strip()[:60]
                print(f"  │    {prompt} {display}")
            except Exception as e:
                print(f"  │    {prompt} [error: {e}]")

        print(f"  └─── LAP {lap} done in {lap_time:.1f}s "
              f"(best PPL: {best_perplexity:.1f}) ──────────┘")

    # ── Final summary ────────────────────────────────────────────────
    print()
    print("─" * 70)
    print("  PHASE 4: Final Summary")
    print("─" * 70)

    print(f"\n  Best test perplexity : {best_perplexity:.1f}")

    print("\n  Lap-by-Lap Progress:")
    print("  " + "─" * 62)
    print(f"  {'Lap':>4} │ {'Epoch':>6} │ {'CE':>8} │ {'PPL':>10} │ "
          f"{'λ₁':>10} │ {'δ²':>10}")
    print("  " + "─" * 62)
    for r in history["laps"]:
        print(f"  {r['lap']:4d} │ {r['global_epoch']:6d} │ "
              f"{r['ce']:8.4f} │ {r['perplexity']:10.1f} │ "
              f"{r['spectral_gap']:10.6f} │ {r['delta_sq']:10.2e}")
    print("  " + "─" * 62)

    # Save
    results_path = Path(output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "best_perplexity": best_perplexity,
            "laps": history["laps"],
            "config": {
                "n_points": config.n_points,
                "manifold_dim": config.manifold_dim,
                "belief_dims": list(config.belief_dims),
                "total_state_dim": config.total_state_dim,
                "embed_dim": config.observation_dim,
            },
        }, f, indent=2, default=str)

    print(f"\n  Results saved to {results_path}")
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + f"  Training complete.  Best PPL: {best_perplexity:.1f}".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    return history


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CDI Language Model Training")
    parser.add_argument("--config", type=str, default="small",
                        choices=["tiny", "small", "medium"])
    parser.add_argument("--laps", type=int, default=5)
    parser.add_argument("--lap-epochs", type=int, default=15)
    parser.add_argument("--ft-epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=200,
                        help="Max batches per training epoch (for speed)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None,
                        help="Token embedding dimension")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config_map = {"tiny": CDIConfig.tiny, "small": CDIConfig.small, "medium": CDIConfig.medium}
    config = config_map[args.config]()
    config.seed = args.seed

    # For LM mode: obs_dim = output_dim = embed_dim
    if args.embed_dim:
        config.observation_dim = args.embed_dim
        config.output_dim = args.embed_dim
    else:
        # Default embed dims per config
        embed_defaults = {"tiny": 32, "small": 48, "medium": 64}
        dim = embed_defaults[args.config]
        config.observation_dim = dim
        config.output_dim = dim

    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    config.validate()

    run_interleaved_training(
        config=config,
        n_laps=args.laps,
        lap_epochs=args.lap_epochs,
        ft_epochs=args.ft_epochs,
        max_batches_per_epoch=args.max_batches,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
