"""
CDI v2.0 Debug / Smoke Test
=============================

Runs a minimal forward-backward pass to verify all four v2.0 fixes:

  Fix F1 — Gradient reaches manifold.points, connection, Dirac, Laplacian
  Fix F2 — Belief state Ψ changes across tokens (not stuck at zeros)
  Fix F3 — rebuild_operators() called after step; operator matrices rebuild
  Fix F4 — dim(B_0) >= embed_dim; engine params >= 15% of embedding params

Usage:
    python debug_train.py
    python debug_train.py --config small
    python debug_train.py --steps 5 --verbose
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List

import torch

from cdi.config import CDIConfig
from cdi.engine import CDIEngine
from cdi.tokenizer import CDITokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ok(cond: bool, msg: str) -> bool:
    icon = "✓" if cond else "✗"
    print(f"  {icon}  {msg}")
    return cond


def clip_grad(params: List[torch.Tensor], max_norm: float = 1.0) -> float:
    total_sq = sum(
        p.grad.data.norm(2).item() ** 2 for p in params if p.grad is not None
    )
    norm = math.sqrt(total_sq)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for p in params:
            if p.grad is not None:
                p.grad.data = p.grad.data * scale
    return norm


# ─────────────────────────────────────────────────────────────────────────────
# Main debug runner
# ─────────────────────────────────────────────────────────────────────────────

def run_debug(config_name: str = "tiny", n_steps: int = 3, verbose: bool = False) -> bool:
    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + "  CDI v2.0 — Debug / Smoke Test".center(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print(f"\n  Config: {config_name}  Steps: {n_steps}")

    # ── Config ────────────────────────────────────────────────────────
    cfg_map = {
        "tiny":   CDIConfig.tiny,
        "small":  CDIConfig.small,
        "medium": CDIConfig.medium,
    }
    config = cfg_map[config_name]()

    print(f"\n  Manifold dim     : {config.manifold_dim}")
    print(f"  n_points         : {config.n_points}")
    print(f"  belief_dims      : {config.belief_dims}")
    print(f"  B_0 dim          : {config.belief_dim(0)}")
    print(f"  embed_dim        : {config.observation_dim}")
    print(f"  state dim N      : {config.total_state_dim}")
    print(f"  spinor dim s     : {config.spinor_dim}")
    print(f"  heat steps K     : {config.heat_steps}")

    # ── Fix F4 check (before build) ───────────────────────────────────
    print("\n  ─── Fix F4: Dimensional Hierarchy ─────────────────────────")
    b0 = config.belief_dim(0)
    ed = config.observation_dim
    total_b = config.total_belief_dim
    all_ok = True
    all_ok &= _ok(b0 >= ed,
                  f"dim(B_0)={b0} >= embed_dim={ed}  (Axiom 2.4.2.1)")
    all_ok &= _ok(total_b >= 4 * ed,
                  f"total_belief={total_b} >= 4×embed={4*ed}  (Axiom 2.4.2.2)")

    # ── Build ─────────────────────────────────────────────────────────
    print("\n  ─── Building engine ────────────────────────────────────────")
    try:
        config.validate()
        engine = CDIEngine(config)
        engine.build()
        print(f"  ✓  Engine built successfully")
    except Exception as e:
        print(f"  ✗  Engine build FAILED: {e}")
        return False

    tokenizer = CDITokenizer(
        embed_dim=config.observation_dim,
        max_len=config.n_points,
        dtype=config.dtype,
    )
    engine_params = engine.get_parameters()
    tok_params = tokenizer.get_parameters()
    all_params = engine_params + tok_params

    n_engine = sum(p.numel() for p in engine_params)
    n_tok = sum(p.numel() for p in tok_params)
    ratio_pct = 100.0 * n_engine / max(n_tok, 1)
    all_ok &= _ok(ratio_pct >= 15.0,
                  f"Engine/Embed ratio = {ratio_pct:.1f}%  (Axiom 2.4.2.3 min 15%)")
    print(f"  ─  Engine: {n_engine:,}  Embed: {n_tok:,}  Total: {n_engine+n_tok:,}")

    # ── Fix F2 pre-check: theta_init is NOT zeros ─────────────────────
    print("\n  ─── Fix F2: Learnable Initial State ───────────────────────")
    all_ok &= _ok(
        engine.theta_init.requires_grad,
        "theta_init.requires_grad == True",
    )
    all_ok &= _ok(
        engine.theta_init.abs().max().item() > 0,
        "theta_init is not all-zeros (has random init)",
    )

    # ── Fix F3 pre-check: operators exist after build ─────────────────
    print("\n  ─── Fix F3: Operator Rebuild ───────────────────────────────")
    D_before = engine.dirac.matrix.data.clone()
    L_before = engine.laplacian.matrix.data.clone()

    # ── Synthetic batch ───────────────────────────────────────────────
    B_batch = 2
    L = config.n_points
    V = tokenizer.vocab_size
    optimizer = torch.optim.Adam(all_params, lr=1e-3)

    initial_lam1 = engine.laplacian.spectral_gap().item()
    initial_psi_norm = engine.theta_init.norm().item()

    print(f"\n  ─── Training {n_steps} steps ────────────────────────────────")
    print(f"  Initial λ₁    = {initial_lam1:.6f}")
    print(f"  Initial ‖Ψ₀‖  = {initial_psi_norm:.6f}")

    step_metrics = []
    psi_evolution = []  # track belief state changes per step

    for step in range(1, n_steps + 1):
        # Synthetic token IDs
        input_ids = torch.randint(0, V, (B_batch, L))
        target_ids = torch.randint(0, V, (B_batch, L))

        embeddings = tokenizer.embed(input_ids)

        # Record Ψ state before forward
        psi_before = engine.theta_init.detach().norm().item()

        # Forward
        output = engine.forward_sequence_batch(embeddings)

        # Loss
        loss, loss_dict = engine.compute_lm_loss(
            output, target_ids, tokenizer.embedding, global_step=step
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad(all_params, max_norm=1.0)
        optimizer.step()

        # Fix F3: rebuild operators
        engine.rebuild_operators()

        # Track Ψ change (theta_init should be updating)
        psi_after = engine.theta_init.detach().norm().item()
        psi_delta = abs(psi_after - psi_before)
        psi_evolution.append(psi_delta)

        if verbose:
            print(
                f"  Step {step}: CE={loss_dict['ce']:.4f}"
                f"  PPL={loss_dict['perplexity']:.1f}"
                f"  λ₁={loss_dict['lambda_1']:.4f}"
                f"  δ²={loss_dict['consistency']:.2e}"
                f"  ∇={grad_norm:.4f}"
                f"  ΔΨ={psi_delta:.2e}"
            )
        step_metrics.append(loss_dict)

    print(f"\n  ─── v2.0 Fix Verification ──────────────────────────────────")

    # ── Fix F1: Gradient flow ─────────────────────────────────────────
    print("\n  Fix F1 — Gradient Connectivity:")
    grad_checks = engine.verify_gradient_flow()
    for name, ok in grad_checks.items():
        all_ok &= _ok(ok, f"grad flowing to {name}")

    # ── Fix F2: Recurrent state evolves ──────────────────────────────
    print("\n  Fix F2 — Recurrent State Evolution:")
    all_ok &= _ok(
        engine.theta_init.requires_grad,
        "theta_init.requires_grad == True",
    )
    all_ok &= _ok(
        any(d > 1e-12 for d in psi_evolution),
        f"theta_init norm changed across steps (max ΔΨ={max(psi_evolution):.2e})",
    )
    # Check that outputs differ per token (state evolves across sequence)
    with torch.no_grad():
        test_seq = torch.randn(L, config.observation_dim, dtype=config.dtype)
        out = engine.forward_sequence(test_seq)
        token_var = out.var(dim=0).mean().item()
    all_ok &= _ok(
        token_var > 1e-12,
        f"Output varies across tokens (var={token_var:.2e}) — state is recurrent",
    )

    # ── Fix F3: Operators actually rebuild ───────────────────────────
    print("\n  Fix F3 — Operator Rebuild:")
    D_after = engine.dirac.matrix.data
    L_after = engine.laplacian.matrix.data
    dirac_changed = not torch.allclose(D_before, D_after, atol=1e-12)
    lap_changed = not torch.allclose(L_before, L_after, atol=1e-12)
    all_ok &= _ok(dirac_changed,   "Dirac matrix changed after rebuild_operators()")
    all_ok &= _ok(lap_changed,     "Laplacian matrix changed after rebuild_operators()")

    # Spectral gap should change
    final_lam1 = engine.laplacian.spectral_gap().item()
    lam1_changed = abs(final_lam1 - initial_lam1) > 1e-12
    _ok(lam1_changed,  f"λ₁ changed: {initial_lam1:.6f} → {final_lam1:.6f}")

    # ── Fix F4: Dimensional hierarchy ────────────────────────────────
    print("\n  Fix F4 — Dimensional Hierarchy (recap):")
    _ok(b0 >= ed,          f"dim(B_0)={b0} >= embed_dim={ed}")
    _ok(total_b >= 4 * ed, f"total_belief={total_b} >= 4×{ed}={4*ed}")
    _ok(ratio_pct >= 15.0, f"Engine/Embed={ratio_pct:.1f}% >= 15%")

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("─" * 62)
    last = step_metrics[-1]
    print(f"  Final step:  CE={last['ce']:.4f}  PPL={last['perplexity']:.1f}"
          f"  λ₁={last['lambda_1']:.4f}  δ²={last['consistency']:.2e}")
    print("─" * 62)

    if all_ok:
        print("\n  ✓  ALL v2.0 fixes verified — CDI architecture is spec-compliant.\n")
    else:
        print("\n  ✗  Some checks FAILED — review output above.\n")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CDI v2.0 Debug / Smoke Test")
    parser.add_argument("--config",  default="tiny", choices=["tiny", "small", "medium"])
    parser.add_argument("--steps",   type=int, default=3, help="Training steps to run")
    parser.add_argument("--verbose", action="store_true", help="Print per-step metrics")
    args = parser.parse_args()

    ok = run_debug(
        config_name=args.config,
        n_steps=args.steps,
        verbose=args.verbose,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
