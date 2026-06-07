#!/usr/bin/env python
"""
CDI v2.0 Test Orchestrator
============================

Single entry point for all CDI tests.

Phases:
  1. Syntax validation  — all .py files compile clean
  2. v2.0 Fix checks    — the four critical fixes verified without pytest
  3. Integration tests  — forward/backward/loss with a real engine
  4. Pytest suite       — if pytest available, run tests/

Usage:
    python run_tests.py                      # full suite
    python run_tests.py --phase syntax       # syntax only
    python run_tests.py --phase fixes        # v2.0 fix checks only
    python run_tests.py --phase integration  # integration only
"""

import sys
import py_compile
import ast
import time
import argparse
from pathlib import Path
from typing import List, Tuple


# ─── Colour helpers ──────────────────────────────────────────────────────────

class C:
    G = '\033[92m'   # green
    R = '\033[91m'   # red
    Y = '\033[93m'   # yellow
    B = '\033[96m'   # cyan/blue
    W = '\033[1m'    # bold
    E = '\033[0m'    # end


def ok(cond: bool, msg: str) -> bool:
    icon = f"{C.G}✓{C.E}" if cond else f"{C.R}✗{C.E}"
    print(f"  {icon}  {msg}")
    return cond


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 1 — Syntax
# ═══════════════════════════════════════════════════════════════════════════

def phase_syntax() -> Tuple[bool, List]:
    print(f"\n{C.W}{C.B}{'='*70}{C.E}")
    print(f"{C.W}PHASE 1: Syntax Validation{C.E}")
    print(f"{C.W}{C.B}{'='*70}{C.E}\n")

    root = Path(__file__).parent
    files = list((root / "cdi").rglob("*.py"))
    files += [root / "dataset.py", root / "train.py",
              root / "debug_train.py", root / "run_tests.py"]
    files = [f for f in files if f.exists()]

    passed = failed = 0
    errors = []
    for fp in sorted(files):
        try:
            py_compile.compile(str(fp), doraise=True)
            with open(fp, encoding="utf-8") as f:
                ast.parse(f.read(), str(fp))
            print(f"  {C.G}✓{C.E} {fp.relative_to(root)}")
            passed += 1
        except SyntaxError as e:
            print(f"  {C.R}✗{C.E} {fp.relative_to(root)}: line {e.lineno}: {e.msg}")
            failed += 1
            errors.append((str(fp), str(e)))
        except Exception as e:
            print(f"  {C.R}✗{C.E} {fp.relative_to(root)}: {e}")
            failed += 1
            errors.append((str(fp), str(e)))

    print(f"\n  {C.G}{passed} passed{C.E}  {C.R}{failed} failed{C.E}")
    return failed == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2 — v2.0 Fix checks (fast, no tokenizer needed)
# ═══════════════════════════════════════════════════════════════════════════

def phase_fixes() -> Tuple[bool, List]:
    print(f"\n{C.W}{C.B}{'='*70}{C.E}")
    print(f"{C.W}PHASE 2: v2.0 Fix Verification{C.E}")
    print(f"{C.W}{C.B}{'='*70}{C.E}\n")

    errors = []
    try:
        import torch
        from cdi.config import CDIConfig
        from cdi.engine import CDIEngine
    except ImportError as e:
        print(f"  {C.R}✗{C.E}  Import failed: {e}")
        return False, [(None, str(e))]

    all_ok = True
    cfg = CDIConfig.tiny()

    # ── Config axioms ─────────────────────────────────────────────────
    print("  Config axioms (v2.0 §2.4):")
    b0 = cfg.belief_dim(0)
    ed = cfg.observation_dim
    tb = cfg.total_belief_dim
    all_ok &= ok(b0 >= ed,       f"dim(B_0)={b0} >= embed={ed}   (Axiom 2.4.2.1)")
    all_ok &= ok(tb >= 4 * ed,   f"total_belief={tb} >= 4×embed={4*ed}  (Axiom 2.4.2.2)")
    try:
        cfg.validate()
        all_ok &= ok(True, "CDIConfig.validate() passes")
    except AssertionError as e:
        all_ok &= ok(False, f"CDIConfig.validate() FAILED: {e}")
        errors.append(("config.validate", str(e)))

    # ── Build ─────────────────────────────────────────────────────────
    print("\n  Engine build:")
    try:
        engine = CDIEngine(cfg)
        engine.build()
        all_ok &= ok(True, "CDIEngine.build() succeeded")
    except Exception as e:
        all_ok &= ok(False, f"CDIEngine.build() FAILED: {e}")
        errors.append(("engine.build", str(e)))
        return False, errors

    # ── Fix F2: theta_init ────────────────────────────────────────────
    print("\n  Fix F2 — Learnable initial state:")
    all_ok &= ok(engine.theta_init.requires_grad, "theta_init.requires_grad == True")
    all_ok &= ok(engine.theta_init.norm().item() > 0, "theta_init != zeros (has random init)")
    all_ok &= ok(engine.W_iota.requires_grad, "W_iota.requires_grad == True")
    all_ok &= ok(engine.W_out.requires_grad,  "W_out.requires_grad == True")

    # ── Fix F2: recurrent output varies across tokens ─────────────────
    print("\n  Fix F2 — Recurrent state evolution:")
    with torch.no_grad():
        seq = torch.randn(cfg.n_points, cfg.observation_dim, dtype=cfg.dtype)
        out = engine.forward_sequence(seq)
    var = out.var(dim=0).mean().item()
    all_ok &= ok(var > 1e-12,
                 f"Token outputs vary (var={var:.2e}) — state is recurrent")

    # ── Fix F1: gradient connectivity ────────────────────────────────
    print("\n  Fix F1 — Gradient connectivity (no .detach in forward):")
    V = 50
    emb = torch.randn(V, cfg.observation_dim, dtype=cfg.dtype, requires_grad=True)
    batch = torch.randn(1, cfg.n_points, cfg.observation_dim, dtype=cfg.dtype)
    target = torch.randint(0, V, (1, cfg.n_points))
    out2 = engine.forward_sequence_batch(batch)
    total, ld = engine.compute_lm_loss(out2, target, emb, global_step=0)
    opt = torch.optim.SGD(engine.get_parameters() + [emb], lr=0.0)
    opt.zero_grad()
    total.backward()

    gf = engine.verify_gradient_flow()
    critical = ["manifold.points", "manifold.metric_L",
                "theta_init", "W_iota", "W_out", "belief.deltas", "connection"]
    for name in critical:
        all_ok &= ok(gf.get(name, False), f"grad → {name}")
        if not gf.get(name, False):
            errors.append(("gradient_flow", f"severed at {name}"))

    # ── Fix F3: rebuild operators ─────────────────────────────────────
    print("\n  Fix F3 — Operator rebuild after step:")
    D_before = engine.dirac.matrix.data.clone()
    L_before = engine.laplacian.matrix.data.clone()
    opt2 = torch.optim.Adam(engine.get_parameters() + [emb], lr=1e-2)
    out3 = engine.forward_sequence_batch(batch)
    loss2, _ = engine.compute_lm_loss(out3, target, emb)
    opt2.zero_grad()
    loss2.backward()
    opt2.step()
    engine.rebuild_operators()                  # v2.0 mandatory

    D_after = engine.dirac.matrix.data
    L_after = engine.laplacian.matrix.data
    d_changed = not torch.allclose(D_before, D_after, atol=1e-12)
    l_changed = not torch.allclose(L_before, L_after, atol=1e-12)
    all_ok &= ok(d_changed, "Dirac matrix updated after rebuild_operators()")
    all_ok &= ok(l_changed, "Laplacian matrix updated after rebuild_operators()")
    if not d_changed:
        errors.append(("rebuild", "Dirac unchanged"))
    if not l_changed:
        errors.append(("rebuild", "Laplacian unchanged"))

    # ── Fix F4: parameter budget ──────────────────────────────────────
    print("\n  Fix F4 — Parameter budget:")
    n_eng = sum(p.numel() for p in engine.get_parameters())
    n_emb = 16000 * cfg.observation_dim
    ratio = 100.0 * n_eng / n_emb
    all_ok &= ok(ratio >= 15.0, f"Engine/Embed={ratio:.1f}% >= 15%  (Axiom 2.4.2.3)")
    if ratio < 15.0:
        errors.append(("param_budget", f"ratio={ratio:.1f}% < 15%"))

    print(f"\n  {'All v2.0 fixes verified ✓' if all_ok else 'Some fixes FAILED ✗'}")
    return all_ok, errors


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 3 — Integration (requires ethiobbpe)
# ═══════════════════════════════════════════════════════════════════════════

def phase_integration() -> Tuple[bool, List]:
    print(f"\n{C.W}{C.B}{'='*70}{C.E}")
    print(f"{C.W}PHASE 3: Integration Tests (tokenizer + engine){C.E}")
    print(f"{C.W}{C.B}{'='*70}{C.E}\n")

    errors = []

    try:
        import torch
        from cdi.config import CDIConfig
        from cdi.engine import CDIEngine
        from cdi.tokenizer import CDITokenizer
        print(f"  {C.G}✓{C.E}  imports OK")
    except ImportError as e:
        print(f"  {C.R}✗{C.E}  Import failed: {e}")
        return False, [(None, str(e))]

    try:
        from ethiobbpe import EthioBBPETokenizer
        print(f"  {C.G}✓{C.E}  ethiobbpe available")
    except ImportError:
        print(f"  {C.Y}⚠{C.E}  ethiobbpe not installed — skipping integration")
        print(f"       Install: pip install -r requirements.txt")
        return True, []  # not a hard failure

    passed = failed = 0
    cfg = CDIConfig.tiny()
    engine = CDIEngine(cfg)
    engine.build()

    tokenizer = CDITokenizer(
        embed_dim=cfg.observation_dim,
        max_len=cfg.n_points,
        dtype=cfg.dtype,
    )
    all_params = engine.get_parameters() + tokenizer.get_parameters()
    optimizer = torch.optim.Adam(all_params, lr=1e-3)

    # 1. Encode
    try:
        ids = tokenizer.encode("What is the basic unit of life?")
        assert ids.shape == (cfg.n_points,)
        print(f"  {C.G}✓{C.E}  tokenizer.encode  → {ids.shape}")
        passed += 1
    except Exception as e:
        print(f"  {C.R}✗{C.E}  tokenizer.encode: {e}")
        failed += 1; errors.append(("encode", str(e)))

    # 2. Embed
    try:
        emb = tokenizer.embed(ids)
        assert emb.shape == (cfg.n_points, cfg.observation_dim)
        print(f"  {C.G}✓{C.E}  tokenizer.embed   → {emb.shape}")
        passed += 1
    except Exception as e:
        print(f"  {C.R}✗{C.E}  tokenizer.embed: {e}")
        failed += 1; errors.append(("embed", str(e)))
        emb = None

    # 3. Forward sequence batch
    output = None
    try:
        if emb is None: raise RuntimeError("skipped")
        batch = emb.unsqueeze(0)  # (1, L, E)
        output = engine.forward_sequence_batch(batch)
        assert output.shape == (1, cfg.n_points, cfg.output_dim)
        assert torch.isfinite(output).all()
        print(f"  {C.G}✓{C.E}  forward_sequence_batch → {output.shape}")
        passed += 1
    except Exception as e:
        print(f"  {C.R}✗{C.E}  forward_sequence_batch: {e}")
        failed += 1; errors.append(("forward", str(e)))

    # 4. LM loss
    loss = None
    try:
        if output is None: raise RuntimeError("skipped")
        tgt = ids.unsqueeze(0)
        loss, ld = engine.compute_lm_loss(output, tgt, tokenizer.embedding)
        assert torch.isfinite(loss)
        print(f"  {C.G}✓{C.E}  compute_lm_loss    CE={ld['ce']:.4f}  PPL={ld['perplexity']:.1f}  λ₁={ld['lambda_1']:.4f}")
        passed += 1
    except Exception as e:
        print(f"  {C.R}✗{C.E}  compute_lm_loss: {e}")
        failed += 1; errors.append(("loss", str(e)))

    # 5. Full training step (v2.0: with rebuild_operators)
    try:
        if loss is None: raise RuntimeError("skipped")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        engine.rebuild_operators()      # v2.0 mandatory
        print(f"  {C.G}✓{C.E}  backward + step + rebuild_operators()")
        passed += 1
    except Exception as e:
        print(f"  {C.R}✗{C.E}  training step: {e}")
        failed += 1; errors.append(("step", str(e)))

    # 6. Gradient flow
    try:
        gf = engine.verify_gradient_flow()
        critical = ["manifold.points", "theta_init", "W_iota", "W_out"]
        all_grad = all(gf.get(k, False) for k in critical)
        if all_grad:
            print(f"  {C.G}✓{C.E}  gradient flow — all critical params have grad")
        else:
            severed = [k for k in critical if not gf.get(k, False)]
            print(f"  {C.R}✗{C.E}  gradient SEVERED at: {severed}")
            failed += 1
        passed += int(all_grad)
    except Exception as e:
        print(f"  {C.R}✗{C.E}  gradient check: {e}")
        failed += 1; errors.append(("gradient", str(e)))

    # 7. Generation
    try:
        from train import generate_sample
        gen = generate_sample(engine, tokenizer, "Q: What is DNA? A:", max_tokens=10)
        tokens = gen.split()
        diverse = len(set(tokens)) > 1
        print(f"  {C.G}✓{C.E}  generate_sample  → \"{gen[:60]}\"")
        if not diverse:
            print(f"  {C.Y}⚠{C.E}  generation collapsed (all same token) — more training needed")
        passed += 1
    except Exception as e:
        print(f"  {C.R}✗{C.E}  generate_sample: {e}")
        failed += 1; errors.append(("generate", str(e)))

    print(f"\n  Integration: {C.G}{passed} passed{C.E}  {C.R}{failed} failed{C.E}")
    return failed == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="CDI v2.0 Test Orchestrator")
    parser.add_argument("--phase", choices=["all", "syntax", "fixes", "integration"],
                        default="all")
    args = parser.parse_args()

    print(f"\n{C.W}")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  CDI v2.0 — Test Orchestrator".center(68) + "║")
    print("║" + "  Verifying all four architectural fixes".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"{C.E}")

    t0 = time.time()
    results = {}

    if args.phase in ("all", "syntax"):
        results["syntax"], _ = phase_syntax()

    if args.phase in ("all", "fixes"):
        results["fixes"], _ = phase_fixes()

    if args.phase in ("all", "integration"):
        results["integration"], _ = phase_integration()

    # ── Pytest optional ───────────────────────────────────────────────
    if args.phase == "all":
        try:
            import pytest
            print(f"\n{C.W}{C.B}{'='*70}{C.E}")
            print(f"{C.W}PHASE 4: pytest tests/{C.E}")
            print(f"{C.W}{C.B}{'='*70}{C.E}")
            ret = pytest.main([
                "tests/", "-v", "--tb=short",
                "-x",          # stop on first failure
                "--no-header",
            ])
            results["pytest"] = ret == 0
        except ImportError:
            print(f"\n  {C.Y}⚠{C.E}  pytest not installed — skipping unit tests")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{C.W}{C.B}{'='*70}{C.E}")
    print(f"{C.W}SUMMARY  ({elapsed:.1f}s){C.E}")
    print(f"{C.W}{C.B}{'='*70}{C.E}")

    phase_names = {
        "syntax":      "Syntax validation",
        "fixes":       "v2.0 Fix verification",
        "integration": "Integration tests",
        "pytest":      "pytest suite",
    }
    all_ok = True
    for key, passed in results.items():
        icon = f"{C.G}✓ PASS{C.E}" if passed else f"{C.R}✗ FAIL{C.E}"
        print(f"  {phase_names.get(key, key):<28s}  {icon}")
        all_ok = all_ok and passed

    print()
    if all_ok:
        print(f"{C.G}{C.W}  ✓  CDI v2.0 — all checks passed{C.E}")
        print(f"\n  Ready to train:")
        print(f"    python train.py --config tiny  --laps 3 --lap-epochs 5")
        print(f"    python train.py --config small --laps 5 --lap-epochs 15")
    else:
        print(f"{C.R}{C.W}  ✗  Some checks failed — review output above{C.E}")

    print(f"{C.W}{C.B}{'='*70}{C.E}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
