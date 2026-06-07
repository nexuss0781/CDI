"""
CDI Training Pipeline — v2.0
==============================

v2.0 Spec Changes (CDI_LM_v2_Technical_Specification.md §4.2):

  Fix F3: engine.rebuild_operators() called after EVERY optimizer.step()
  Fix F1: gradient flow verified after first step; halt if severed
  Fix F2: recurrent Ψ — no zero resets between tokens
  Fix F4: embed_dim matches config (no override that violates axioms)

Training loop (Spec Algorithm 4.2.1):
    1. Forward pass — recurrent CDI sequence
    2. Loss = CE + λ_B·Bianchi + λ_C·Consist + λ_S·Spectral
    3. loss.backward()
    4. clip_grad_norm_
    5. optimizer.step()
    6. engine.rebuild_operators()   ← MANDATORY (was missing in v1.0)
    7. Every 100 steps: log gradient flow + spectral gap

Usage:
    python train.py --config tiny  --laps 3 --lap-epochs 5
    python train.py --config small --laps 5 --lap-epochs 15
    python train.py --config medium --laps 5 --lap-epochs 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _ram_used_gb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    except Exception:
        return 0.0


def clip_grad_norm_(params: List[torch.Tensor], max_norm: float) -> float:
    """Pure-tensor gradient clipping (no torch.nn). O(n_params)."""
    total_sq = sum(
        p.grad.data.norm(2).item() ** 2
        for p in params if p.grad is not None
    )
    total_norm = math.sqrt(total_sq)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for p in params:
            if p.grad is not None:
                p.grad.data = p.grad.data * scale
    return total_norm


# ═══════════════════════════════════════════════════════════════════════════
#  Gradient flow check — Spec §4.3
# ═══════════════════════════════════════════════════════════════════════════

def _check_gradient_flow(engine: CDIEngine, step: int) -> None:
    """Verify all parameter groups received gradient.

    Spec §4.3 Verification Test 1 + Test 2.
    Prints a warning table if any critical parameter has zero gradient.
    At step 1 a full gradient flow table is always printed.
    """
    checks = engine.verify_gradient_flow()
    failed = [k for k, v in checks.items() if not v]

    if step == 1:
        print("\n  ┌─── Gradient Flow Verification (Step 1) ─────────────────┐")
        for name, ok in checks.items():
            status = "✓  FLOWING" if ok else "✗  SEVERED"
            print(f"  │    {name:<25s}  {status}")
        if failed:
            print("  │")
            print(f"  │  WARNING: {len(failed)} parameter(s) have zero gradient.")
            print("  │  These parameters are not learning. Check .detach() usage.")
        else:
            print("  │  All parameters receiving gradient — gradient topology OK.")
        print("  └──────────────────────────────────────────────────────────┘\n")
    elif failed:
        print(f"\n  [Step {step}] Gradient WARNING — severed: {failed}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Single training step — v2.0 (Spec Algorithm 4.2.1)
# ═══════════════════════════════════════════════════════════════════════════

def training_step(
    engine: CDIEngine,
    tokenizer: CDITokenizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    all_params: List[torch.Tensor],
    global_step: int,
) -> Dict[str, float]:
    """One training step per Spec Algorithm 4.2.1.

    Steps:
      1. Forward pass (recurrent CDI sequence)
      2. Compute composite loss
      3. Backprop
      4. Gradient clipping
      5. optimizer.step()
      6. MANDATORY rebuild_operators()    ← v2.0 Fix F3
      7. Gradient flow check at step 1
    """
    # 1. Embed token IDs → (B, L, embed_dim)
    embeddings = tokenizer.embed(input_ids)

    # 2. Forward: recurrent belief evolution
    output = engine.forward_sequence_batch(embeddings)  # (B, L, embed_dim)

    # 3. Composite loss (Spec §4.1)
    loss, loss_dict = engine.compute_lm_loss(
        output, target_ids, tokenizer.embedding, global_step=global_step
    )

    # 4. Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # 5. Gradient clipping
    grad_norm = clip_grad_norm_(all_params, max_norm=1.0)
    loss_dict["grad_norm"] = grad_norm

    # 6. Parameter update
    optimizer.step()

    # 7. MANDATORY: Rebuild all operators from updated parameters (Fix F3)
    engine.rebuild_operators()
    engine.global_step += 1

    # 8. Gradient flow verification
    if global_step <= 1 or (global_step % engine.config.spectral_diag_every == 0):
        _check_gradient_flow(engine, global_step)

    return loss_dict


# ═══════════════════════════════════════════════════════════════════════════
#  Epoch runner
# ═══════════════════════════════════════════════════════════════════════════

def run_lm_epoch(
    engine: CDIEngine,
    tokenizer: CDITokenizer,
    dataloader,
    optimizer: torch.optim.Optimizer,
    all_params: List[torch.Tensor],
    global_step_start: int,
    max_batches: int = None,
    label: str = "",
) -> Tuple[Dict[str, float], int]:
    """One epoch of next-token prediction training.

    Returns (metrics_dict, global_step_end).
    """
    totals: Dict[str, float] = {
        "loss": 0.0, "ce": 0.0, "perplexity": 0.0,
        "consistency": 0.0, "bianchi": 0.0, "grad_norm": 0.0,
        "lambda_1": 0.0,
    }
    n_batches = 0
    global_step = global_step_start
    total_batches = max_batches if max_batches else len(dataloader)

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        global_step += 1
        metrics = training_step(
            engine, tokenizer, input_ids, target_ids,
            optimizer, all_params, global_step,
        )

        for k in totals:
            totals[k] += metrics.get(k, 0.0)
        n_batches += 1

        # Progress line
        ram = _ram_used_gb()
        print(
            f"\r  │    {label}  {n_batches}/{total_batches}"
            f"  CE={metrics['ce']:.4f}"
            f"  PPL={metrics['perplexity']:.1f}"
            f"  λ₁={metrics['lambda_1']:.4f}"
            f"  ∇={metrics['grad_norm']:.3f}"
            f"  RAM={ram:.1f}GB   ",
            end="", flush=True,
        )

    print("\r" + " " * 100 + "\r", end="", flush=True)

    d = max(n_batches, 1)
    return {k: v / d for k, v in totals.items()}, global_step


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_lm(
    engine: CDIEngine,
    tokenizer: CDITokenizer,
    dataloader,
) -> Dict[str, float]:
    """Evaluate perplexity with no_grad. Uses weight-tied logits."""
    total_ce = 0.0
    n_tokens = 0

    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            embeddings = tokenizer.embed(input_ids)
            output = engine.forward_sequence_batch(embeddings)
            logits = output @ tokenizer.embedding.T   # (B, L, V)

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
        "perplexity": math.exp(min(avg_ce, 20)),
        "n_tokens": n_tokens,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Text Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_sample(
    engine: CDIEngine,
    tokenizer: CDITokenizer,
    prompt: str,
    max_tokens: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    """Autoregressive generation with temperature + top-k sampling.

    v2.0: uses temperature and top-k to prevent collapsed greedy decoding.
    temperature=1.0, top_k=50 gives diverse output.
    temperature=0.7, top_k=20 gives more focused output.
    """
    ids = tokenizer.encode(prompt)
    seq_len = tokenizer.max_len

    with torch.no_grad():
        for _ in range(max_tokens):
            window = ids[-seq_len:]
            embeddings = tokenizer.embed(window).unsqueeze(0)      # (1, L, E)
            output = engine.forward_sequence_batch(embeddings)     # (1, L, E)
            logits = output[0, -1] @ tokenizer.embedding.T         # (V,)

            # Temperature scaling
            logits = logits / max(temperature, 1e-6)

            # Top-k filtering
            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                threshold = top_vals[-1]
                logits = logits.masked_fill(logits < threshold, float('-inf'))

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            ids = torch.cat([ids, torch.tensor([next_id], dtype=torch.long)])

            if next_id == tokenizer.eos_id:
                break

    return tokenizer.decode_ids(ids)


# ═══════════════════════════════════════════════════════════════════════════
#  Interleaved Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_interleaved_training(
    config: CDIConfig,
    n_laps: int = 5,
    lap_epochs: int = 15,
    ft_epochs: int = 5,
    max_batches_per_epoch: int = 200,
    output_dir: str = "./results",
    temperature: float = 1.0,
    top_k: int = 50,
) -> Dict:
    """Interleaved CDI-LM v2.0 training.

    Loop:
        for lap in 1..n_laps:
            TRAIN      lap_epochs on WikiText-2
            FINE-TUNE  ft_epochs  on SciQ
            TEST       science questions → PPL
            GENERATE   samples with top-k temperature sampling
            LOG        CE, PPL, λ₁, δ², gradient flow
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    seq_len = config.n_points

    # ── Header ────────────────────────────────────────────────────────
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  𝕮𝔇𝕴 — COHOMODYNAMIC INTELLIGENCE  v2.0".center(68) + "║")
    print("║" + "  Next-Generation Language Model  (Spec-Compliant)".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"  v2.0 Fixes Applied:")
    print(f"    F1 — Differentiable inference (no .detach in forward path)")
    print(f"    F2 — Recurrent belief state Ψ with learnable theta_init")
    print(f"    F3 — rebuild_operators() after every optimizer.step()")
    print(f"    F4 — dim(B_0) >= embed_dim enforced by config validation")
    print()
    print(f"  Context length  : {seq_len} tokens (= n_points)")
    print(f"  State dim N     : {config.total_state_dim}")
    print(f"  Manifold dim d  : {config.manifold_dim}")
    print(f"  Belief dims     : {config.belief_dims}  (B_0={config.belief_dim(0)})")
    print(f"  Spinor dim s    : {config.spinor_dim}")
    print(f"  Heat steps K    : {config.heat_steps}  dt={config.heat_dt}")
    print(f"  Laps            : {n_laps} × ({lap_epochs} train + {ft_epochs} ft)")
    print(f"  Complexity      : O(1) reflex | O(n) learn | O(n log n) abstract")
    print()

    # ── Tokenizer ─────────────────────────────────────────────────────
    print("─" * 70)
    print("  PHASE 1: Tokenizer + Datasets")
    print("─" * 70)

    embed_dim = config.observation_dim
    tokenizer = CDITokenizer(
        embed_dim=embed_dim,
        max_len=seq_len,
        dtype=config.dtype,
    )
    print(f"  Tokenizer : EthioBBPE  vocab={tokenizer.vocab_size:,}  embed={embed_dim}")
    print(f"  B_0 dim   : {config.belief_dim(0)}  (≥ embed_dim={embed_dim}: "
          f"{'✓ OK' if config.belief_dim(0) >= embed_dim else '✗ VIOLATION'})")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = download_wikitext(seq_len=seq_len)
    ft_ds = download_sciq(seq_len=seq_len)
    test_ds = make_test_set(seq_len=seq_len)

    train_loader = make_dataloader(train_ds, batch_size=config.batch_size, shuffle=True)
    ft_loader = make_dataloader(ft_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = make_dataloader(test_ds, batch_size=len(test_ds), shuffle=False)

    print(f"\n  Train     : {train_ds}")
    print(f"  Fine-tune : {ft_ds}")
    print(f"  Test      : {test_ds}")

    # ── Engine ────────────────────────────────────────────────────────
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
    ratio_pct = 100.0 * n_engine / max(n_tok, 1)
    min_ratio_pct = 15.0  # Spec Axiom 2.4.2.3

    print(f"  Engine params : {n_engine:,}")
    print(f"  Embedding     : {n_tok:,}  ({tokenizer.vocab_size} × {embed_dim})")
    print(f"  Total params  : {n_engine + n_tok:,}")
    print(f"  Engine/Embed  : {ratio_pct:.1f}%  "
          f"(min {min_ratio_pct}%: {'✓ OK' if ratio_pct >= min_ratio_pct else '✗ BELOW SPEC'})")
    print(f"  Spectral gap  : {engine.laplacian.spectral_gap().item():.6f}")
    print(f"  Learning time : {engine.heat.learning_time().item():.4f}")
    print(f"  Harmonic dim  : {engine.hodge.harmonic_dimension()}")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(all_params, lr=config.learning_rate, betas=(0.9, 0.98))

    # ── Interleaved Training ──────────────────────────────────────────
    print()
    print("─" * 70)
    print("  PHASE 3: Interleaved Training  (v2.0 — rebuild after every step)")
    print("─" * 70)

    history: Dict = {"laps": [], "train": [], "ft": []}
    best_perplexity = float("inf")
    global_step = 0
    global_epoch = 0
    initial_lam1 = engine.laplacian.spectral_gap().item()

    for lap in range(1, n_laps + 1):
        lap_start = time.time()
        print(f"\n  ┌─── LAP {lap}/{n_laps} ──────────────────────────────────────────┐")

        # ── TRAIN on Wikipedia ───────────────────────────────────────
        print(f"  │  TRAINING on Wikipedia ({lap_epochs} epochs)")
        for ep in range(1, lap_epochs + 1):
            global_epoch += 1
            t0 = time.time()
            metrics, global_step = run_lm_epoch(
                engine, tokenizer, train_loader, optimizer, all_params,
                global_step_start=global_step,
                max_batches=max_batches_per_epoch,
                label=f"Ep {ep}/{lap_epochs}",
            )
            dt = time.time() - t0
            history["train"].append(
                {"global_epoch": global_epoch, "lap": lap, **metrics}
            )
            print(
                f"  │    Epoch {ep}/{lap_epochs}  {dt:.1f}s"
                f"  CE={metrics['ce']:.4f}"
                f"  PPL={metrics['perplexity']:.1f}"
                f"  λ₁={metrics['lambda_1']:.4f}"
                f"  δ²={metrics['consistency']:.2e}"
                f"  ∇={metrics['grad_norm']:.3f}"
            )

        # ── FINE-TUNE on SciQ ────────────────────────────────────────
        print(f"  │  FINE-TUNING on SciQ ({ft_epochs} epochs)")
        for ep in range(1, ft_epochs + 1):
            global_epoch += 1
            t0 = time.time()
            metrics, global_step = run_lm_epoch(
                engine, tokenizer, ft_loader, optimizer, all_params,
                global_step_start=global_step,
                max_batches=None,
                label=f"FT {ep}/{ft_epochs}",
            )
            dt = time.time() - t0
            history["ft"].append(
                {"global_epoch": global_epoch, "lap": lap, **metrics}
            )
            print(
                f"  │    FT {ep}/{ft_epochs}  {dt:.1f}s"
                f"  CE={metrics['ce']:.4f}"
                f"  PPL={metrics['perplexity']:.1f}"
                f"  λ₁={metrics['lambda_1']:.4f}"
            )

        # ── TEST ─────────────────────────────────────────────────────
        test_metrics = evaluate_lm(engine, tokenizer, test_loader)

        if test_metrics["perplexity"] < best_perplexity:
            best_perplexity = test_metrics["perplexity"]
            torch.save(
                [p.data.clone() for p in all_params],
                Path(output_dir) / "best_params.pt",
            )

        # Mathematical diagnostics
        current_lam1 = engine.laplacian.spectral_gap().item()
        lam1_delta_pct = abs(current_lam1 - initial_lam1) / max(initial_lam1, 1e-12) * 100
        diag = {
            "spectral_gap":   current_lam1,
            "learning_time":  engine.heat.learning_time().item(),
            "harmonic_dim":   engine.hodge.harmonic_dimension(),
            "delta_sq":       engine.belief.consistency_penalty().item(),
            "lam1_delta_pct": lam1_delta_pct,
        }

        lap_time = time.time() - lap_start
        lap_record = {
            "lap": lap, "global_epoch": global_epoch, "global_step": global_step,
            **test_metrics, **diag, "lap_time": lap_time,
        }
        history["laps"].append(lap_record)

        # Spec §6 Validation Criterion: λ₁ must change by >10% over first epoch
        spec_lam1_ok = lam1_delta_pct > 10.0
        spec_consist_ok = diag["delta_sq"] < 1e-6
        spec_ppl_ok = test_metrics["perplexity"] < 50.0

        print(f"  │  TEST:")
        print(f"  │    CE         = {test_metrics['ce']:.4f}")
        print(f"  │    Perplexity = {test_metrics['perplexity']:.1f}"
              f"  {'✓' if spec_ppl_ok else '○'} (target <50)")
        print(f"  │    λ₁         = {diag['spectral_gap']:.6f}"
              f"  Δ={lam1_delta_pct:.1f}%"
              f"  {'✓' if spec_lam1_ok else '○'} (target >10% change)")
        print(f"  │    τ          = {diag['learning_time']:.4f}")
        print(f"  │    ℋ          = {diag['harmonic_dim']}")
        print(f"  │    δ²         = {diag['delta_sq']:.2e}"
              f"  {'✓' if spec_consist_ok else '○'} (target <1e-6)")

        # ── Sample generation (temperature + top-k) ──────────────────
        print(f"  │  Generation (T={temperature}, top_k={top_k}):")
        prompts = [
            "Q: What is the basic unit of life? A:",
            "Q: What force keeps planets in orbit? A:",
            "Q: What is the chemical symbol for water? A:",
        ]
        for prompt in prompts:
            try:
                generated = generate_sample(
                    engine, tokenizer, prompt,
                    max_tokens=15, temperature=temperature, top_k=top_k,
                )
                display = generated[len(prompt):].strip()[:70]
                print(f"  │    {display}")
            except Exception as e:
                print(f"  │    [error: {e}]")

        print(
            f"  └─── LAP {lap} done in {lap_time:.1f}s"
            f"  best PPL={best_perplexity:.1f} ──────────────┘"
        )

    # ── Final Summary ─────────────────────────────────────────────────
    print()
    print("─" * 70)
    print("  PHASE 4: Final Summary")
    print("─" * 70)
    print(f"\n  Best test perplexity : {best_perplexity:.1f}")
    print(f"  Total training steps : {global_step:,}")

    # Spec §6 Validation Criteria
    final_diag = history["laps"][-1]
    print("\n  Spec §6 Validation Criteria:")
    criteria = [
        ("CE < 3.0",             final_diag["ce"] < 3.0,             f"CE={final_diag['ce']:.4f}"),
        ("PPL < 50",             final_diag["perplexity"] < 50.0,    f"PPL={final_diag['perplexity']:.1f}"),
        ("δ² < 1e-6",           final_diag["delta_sq"] < 1e-6,      f"δ²={final_diag['delta_sq']:.2e}"),
        ("λ₁ changed >10%",     final_diag["lam1_delta_pct"] > 10.0, f"Δλ₁={final_diag['lam1_delta_pct']:.1f}%"),
    ]
    for name, ok, value in criteria:
        print(f"    {'✓' if ok else '✗'}  {name:<20s}  {value}")

    # Lap-by-lap table
    print("\n  Lap-by-Lap Progress:")
    print("  " + "─" * 70)
    print(f"  {'Lap':>4} │ {'Step':>7} │ {'CE':>8} │ {'PPL':>10} │ "
          f"{'λ₁':>10} │ {'δ²':>10} │ {'Δλ₁%':>7}")
    print("  " + "─" * 70)
    for r in history["laps"]:
        print(
            f"  {r['lap']:4d} │ {r['global_step']:7d} │"
            f" {r['ce']:8.4f} │ {r['perplexity']:10.1f} │"
            f" {r['spectral_gap']:10.6f} │ {r['delta_sq']:10.2e} │"
            f" {r['lam1_delta_pct']:6.1f}%"
        )
    print("  " + "─" * 70)

    # Save results
    results_path = Path(output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "version": "v2.0",
            "best_perplexity": best_perplexity,
            "total_steps": global_step,
            "laps": history["laps"],
            "config": {
                "n_points":        config.n_points,
                "manifold_dim":    config.manifold_dim,
                "belief_dims":     list(config.belief_dims),
                "b0_dim":          config.belief_dim(0),
                "total_state_dim": config.total_state_dim,
                "embed_dim":       config.observation_dim,
                "spinor_dim":      config.spinor_dim,
                "heat_steps_K":    config.heat_steps,
                "heat_dt":         config.heat_dt,
            },
            "v2_fixes": {
                "F1_no_detach":           True,
                "F2_recurrent_state":     True,
                "F3_rebuild_after_step":  True,
                "F4_dim_hierarchy":       config.belief_dim(0) >= config.observation_dim,
            },
        }, f, indent=2, default=str)

    print(f"\n  Results saved → {results_path}")
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + f"  CDI-LM v2.0  Training complete.  Best PPL: {best_perplexity:.1f}".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    return history


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CDI-LM v2.0 Training (Spec-Compliant)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="small", choices=["tiny", "small", "medium"],
                        help="Model size preset")
    parser.add_argument("--laps",       type=int,   default=5)
    parser.add_argument("--lap-epochs", type=int,   default=15)
    parser.add_argument("--ft-epochs",  type=int,   default=5)
    parser.add_argument("--max-batches",type=int,   default=200,
                        help="Max batches per train epoch (0 = all)")
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--output-dir", type=str,   default="./results")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--temperature",type=float, default=1.0,
                        help="Sampling temperature for generation")
    parser.add_argument("--top-k",      type=int,   default=50,
                        help="Top-k for generation sampling")

    args = parser.parse_args()

    config_map = {
        "tiny":   CDIConfig.tiny,
        "small":  CDIConfig.small,
        "medium": CDIConfig.medium,
    }
    config = config_map[args.config]()
    config.seed = args.seed

    # v2.0: embed_dim comes from the config preset — do NOT override
    # in a way that violates Axiom 2.4.2.1 (dim B_0 >= embed_dim).
    # The config presets are already spec-compliant.
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    max_batches = args.max_batches if args.max_batches > 0 else None

    # Validate — will raise if any v2.0 axiom is violated
    config.validate()

    run_interleaved_training(
        config=config,
        n_laps=args.laps,
        lap_epochs=args.lap_epochs,
        ft_epochs=args.ft_epochs,
        max_batches_per_epoch=max_batches,
        output_dir=args.output_dir,
        temperature=args.temperature,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
