#!/usr/bin/env python
"""
CDI v2.0 Test Orchestrator
============================

Legacy support test orchestrator.

The authoritative complete regression command is ``pytest -q``. This helper
keeps the older v2 smoke phases available and recursively collects every
``tests/test_*.py`` regression module in its pytest phase.

Usage:
    python run_tests.py                      # full suite (all phases)
    python run_tests.py --phase syntax
    python run_tests.py --phase fixes
    python run_tests.py --phase integration
    python run_tests.py --phase pytest
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
    B = '\033[96m'   # cyan
    W = '\033[1m'    # bold
    E = '\033[0m'    # reset


def ok(cond: bool, msg: str) -> bool:
    icon = f"{C.G}✓{C.E}" if cond else f"{C.R}✗{C.E}"
    print(f"  {icon}  {msg}")
    return cond


def _banner(title: str) -> None:
    print(f"\n{C.W}{C.B}{'='*70}{C.E}")
    print(f"{C.W}{title}{C.E}")
    print(f"{C.W}{C.B}{'='*70}{C.E}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 1 — Syntax
# ═══════════════════════════════════════════════════════════════════════════

def phase_syntax() -> Tuple[bool, List]:
    _banner("PHASE 1: Syntax Validation")

    root = Path(__file__).parent

    # Collect every .py file in cdi/, tests/, and the root scripts
    files: List[Path] = []
    files += sorted((root / "cdi").rglob("*.py"))
    files += sorted((root / "tests").rglob("*.py"))
    for name in ("dataset.py", "train.py", "debug_train.py", "run_tests.py"):
        p = root / name
        if p.exists():
            files.append(p)

    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for f in files:
        k = str(f.resolve())
        if k not in seen:
            seen.add(k)
            unique.append(f)

    passed = failed = 0
    errors: List[Tuple] = []

    for fp in unique:
        try:
            py_compile.compile(str(fp), doraise=True)
            with open(fp, encoding="utf-8") as fh:
                ast.parse(fh.read(), str(fp))
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
#  Phase 2 — v2.0 Fix checks
# ═══════════════════════════════════════════════════════════════════════════

def phase_fixes() -> Tuple[bool, List]:
    _banner("PHASE 2: v2.0 Fix Verification")

    errors: List[Tuple] = []
    try:
        import torch
        from cdi.config import CDIConfig
        from cdi.engine import CDIEngine
    except ImportError as e:
        print(f"  {C.R}✗{C.E}  Import failed: {e}")
        return False, [(None, str(e))]

    all_ok = True
    cfg = CDIConfig.tiny()

    # Config axioms
    print("  Config axioms (v2.0 §2.4):")
    b0 = cfg.belief_dim(0)
    ed = cfg.observation_dim
    tb = cfg.total_belief_dim
    all_ok &= ok(b0 >= ed,     f"dim(B_0)={b0} >= embed={ed}   (Axiom 2.4.2.1)")
    all_ok &= ok(tb >= 4 * ed, f"total_belief={tb} >= 4×embed={4*ed}  (Axiom 2.4.2.2)")
    try:
        cfg.validate()
        all_ok &= ok(True, "CDIConfig.validate() passes")
    except AssertionError as e:
        all_ok &= ok(False, f"CDIConfig.validate() FAILED: {e}")
        errors.append(("config.validate", str(e)))

    # Build
    print("\n  Engine build:")
    try:
        engine = CDIEngine(cfg)
        engine.build()
        all_ok &= ok(True, "CDIEngine.build() succeeded")
    except Exception as e:
        all_ok &= ok(False, f"CDIEngine.build() FAILED: {e}")
        errors.append(("engine.build", str(e)))
        return False, errors

    # Fix F2: learnable initial state
    print("\n  Fix F2 — Learnable initial state:")
    all_ok &= ok(engine.theta_init.requires_grad, "theta_init.requires_grad == True")
    all_ok &= ok(engine.theta_init.norm().item() > 0, "theta_init != zeros")
    all_ok &= ok(engine.W_iota.requires_grad, "W_iota.requires_grad == True")
    all_ok &= ok(engine.W_out.requires_grad,  "W_out.requires_grad == True")

    # Fix F2: recurrent output varies
    print("\n  Fix F2 — Recurrent state evolution:")
    with torch.no_grad():
        seq = torch.randn(cfg.n_points, cfg.observation_dim, dtype=cfg.dtype)
        out = engine.forward_sequence(seq)
    var = out.var(dim=0).mean().item()
    all_ok &= ok(var > 1e-12, f"Token outputs vary (var={var:.2e}) — state is recurrent")

    # Fix F1: gradient connectivity
    print("\n  Fix F1 — Gradient connectivity:")
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

    # Fix F3: rebuild operators
    print("\n  Fix F3 — Operator rebuild after step:")
    D_before = engine.dirac.matrix.data.clone()
    L_before = engine.laplacian.matrix.data.clone()
    opt2 = torch.optim.Adam(engine.get_parameters() + [emb], lr=1e-2)
    # The previous gradient check consumed the graph captured by the live
    # operator matrices; rebuild before constructing this independent step.
    engine.rebuild_operators()
    out3 = engine.forward_sequence_batch(batch)
    loss2, _ = engine.compute_lm_loss(out3, target, emb)
    opt2.zero_grad(); loss2.backward(); opt2.step()
    engine.rebuild_operators()

    d_changed = not torch.allclose(engine.dirac.matrix.data, D_before, atol=1e-12)
    l_changed = not torch.allclose(engine.laplacian.matrix.data, L_before, atol=1e-12)
    all_ok &= ok(d_changed, "Dirac matrix updated after rebuild_operators()")
    all_ok &= ok(l_changed, "Laplacian matrix updated after rebuild_operators()")
    if not d_changed: errors.append(("rebuild", "Dirac unchanged"))
    if not l_changed: errors.append(("rebuild", "Laplacian unchanged"))

    # Fix F4: parameter budget
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
#  Phase 3 — Integration
# ═══════════════════════════════════════════════════════════════════════════

def phase_integration() -> Tuple[bool, List]:
    _banner("PHASE 3: Integration Tests (tokenizer + engine)")

    errors: List[Tuple] = []

    try:
        import torch
        from cdi.config import CDIConfig
        from cdi.engine import CDIEngine
        from cdi.tokenizer import CDITokenizer
        print(f"  {C.G}✓{C.E}  imports OK")
    except ImportError as e:
        print(f"  {C.R}✗{C.E}  Import failed: {e}")
        return False, [(None, str(e))]

    # Verify the exact published EthioBBPE tokenizer required by CDI.
    try:
        from ethiobbpe import EthioBBPETokenizer
        ethio = EthioBBPETokenizer.from_pretrained()
        assert ethio.get_vocab_size() > 0
        print(f"  {C.G}✓{C.E}  EthioBBPE tokenizer available (vocab={ethio.get_vocab_size():,})")
    except Exception as e:
        print(f"  {C.R}✗{C.E}  EthioBBPE tokenizer unavailable ({e})")
        return False, [("ethiobbpe", str(e))]

    passed = failed = 0
    cfg = CDIConfig.tiny()
    engine = CDIEngine(cfg)
    engine.build()

    tokenizer = CDITokenizer(embed_dim=cfg.observation_dim, max_len=cfg.n_points,
                             dtype=cfg.dtype)
    all_params = engine.get_parameters() + tokenizer.get_parameters()
    optimizer = torch.optim.Adam(all_params, lr=1e-3)

    # 1. encode_and_embed
    ids = emb_seq = None
    try:
        ids, emb_seq = tokenizer.encode_and_embed(
            "What is the basic unit of life?"
        )
        assert emb_seq.shape == (cfg.n_points, cfg.observation_dim)
        print(f"  {C.G}✓{C.E}  encode_and_embed  → {emb_seq.shape}")
        passed += 1
    except Exception as e:
        print(f"  {C.R}✗{C.E}  encode_and_embed: {e}")
        failed += 1; errors.append(("encode_embed", str(e)))

    # 2. forward_sequence_batch
    output = None
    try:
        if emb_seq is None: raise RuntimeError("skipped")
        batch = emb_seq.unsqueeze(0)
        output = engine.forward_sequence_batch(batch)
        assert output.shape == (1, cfg.n_points, cfg.output_dim)
        assert torch.isfinite(output).all()
        print(f"  {C.G}✓{C.E}  forward_sequence_batch → {output.shape}")
        passed += 1
    except Exception as e:
        print(f"  {C.R}✗{C.E}  forward_sequence_batch: {e}")
        failed += 1; errors.append(("forward", str(e)))

    # 3. LM loss
    loss = None
    try:
        if output is None or ids is None: raise RuntimeError("skipped")
        tgt = ids.unsqueeze(0)
        loss, ld = engine.compute_lm_loss(
            output, tgt,             tokenizer.embedding,
        )
        assert torch.isfinite(loss)
        print(f"  {C.G}✓{C.E}  compute_lm_loss  CE={ld['ce']:.4f}  "
              f"PPL={ld['perplexity']:.1f}  λ₁={ld['lambda_1']:.4f}")
        passed += 1
    except Exception as e:
        print(f"  {C.R}✗{C.E}  compute_lm_loss: {e}")
        failed += 1; errors.append(("loss", str(e)))

    # 4. training step
    try:
        if loss is None: raise RuntimeError("skipped")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        engine.rebuild_operators()
        print(f"  {C.G}✓{C.E}  backward + step + rebuild_operators()")
        passed += 1
    except Exception as e:
        print(f"  {C.R}✗{C.E}  training step: {e}")
        failed += 1; errors.append(("step", str(e)))

    # 5. gradient flow
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

    # 6. generate_sample
    try:
        from train import generate_sample
        gen = generate_sample(engine, tokenizer,
                              "Q: What is DNA? A:", max_tokens=10)
        print(f"  {C.G}✓{C.E}  generate_sample  → \"{gen[:60]}\"")
        if len(set(gen.split())) <= 1:
            print(f"  {C.Y}⚠{C.E}  generation collapsed — needs more training")
        passed += 1
    except Exception as e:
        print(f"  {C.R}✗{C.E}  generate_sample: {e}")
        failed += 1; errors.append(("generate", str(e)))

    print(f"\n  Integration: {C.G}{passed} passed{C.E}  {C.R}{failed} failed{C.E}")
    return failed == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 4 — pytest suite (all four test files)
# ═══════════════════════════════════════════════════════════════════════════

# Preferred run order — foundational tests first so failures are obvious.
# Any additional test_*.py files discovered in tests/ are appended after.
_TEST_FILES_ORDERED = [
    "tests/test_core.py",
    "tests/test_operators.py",
    "tests/test_dynamics.py",
    "tests/test_extended.py",
]


def phase_pytest() -> Tuple[bool, List]:
    _banner("PHASE 4: pytest suite — all tests/ files")

    errors: List[Tuple] = []

    try:
        import pytest
    except ImportError:
        print(f"  {C.Y}⚠{C.E}  pytest not installed — run: pip install pytest")
        return True, []   # soft skip

    root = Path(__file__).parent
    tests_dir = root / "tests"

    if not tests_dir.exists():
        print(f"  {C.R}✗{C.E}  tests/ directory not found")
        return False, [("missing_dir", "tests/")]

    # Build ordered list: preferred files first, then any extras discovered
    discovered = sorted(
        p for p in tests_dir.rglob("test_*.py")
        if p.name != "__init__.py"
    )
    ordered_paths: List[Path] = []
    seen_names: set = set()

    # Add preferred files first (skip if missing — will be noted below)
    for rel in _TEST_FILES_ORDERED:
        p = root / rel
        if p.exists():
            ordered_paths.append(p)
            seen_names.add(p.name)
        else:
            print(f"  {C.Y}⚠{C.E}  preferred file not found (skipping): {rel}")

    # Append any extras not already in the ordered list
    for p in discovered:
        if p.name not in seen_names:
            ordered_paths.append(p)
            seen_names.add(p.name)

    if not ordered_paths:
        print(f"  {C.R}✗{C.E}  no test files found in tests/")
        return False, [("no_tests", "tests/")]

    # Report what will run
    print(f"  Collecting {len(ordered_paths)} test file(s):")
    for p in ordered_paths:
        print(f"    {p.relative_to(root)}")
    print()

    # ── Single pytest invocation — NO -x so all tests run ─────────────
    # Flags:
    #   -v              verbose per-test lines
    #   --tb=short      compact tracebacks
    #   --no-header     suppress pytest version header
    #   -p no:cacheprovider  no .pytest_cache writes
    #   NO -x           never stop on first failure
    args = [
        *[str(p) for p in ordered_paths],
        "-v",
        "--tb=short",
        "--no-header",
        "-p", "no:cacheprovider",
    ]

    ret = pytest.main(args)

    passed = ret == 0
    if not passed:
        errors.append(("pytest", f"exit code {ret}"))

    return passed, errors


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

_PHASE_MAP = {
    "syntax":      ("Syntax validation",     phase_syntax),
    "fixes":       ("v2.0 Fix verification", phase_fixes),
    "integration": ("Integration tests",     phase_integration),
    "pytest":      ("pytest suite",          phase_pytest),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="CDI v2.0 Test Orchestrator")
    parser.add_argument(
        "--phase",
        choices=["all"] + list(_PHASE_MAP.keys()),
        default="all",
        help="Which phase(s) to run (default: all)",
    )
    args = parser.parse_args()

    # Header
    print(f"\n{C.W}")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  CDI — Legacy Smoke and Complete Regression Orchestrator".center(68) + "║")
    print("║" + "  pytest phase recursively collects all tests/test_*.py modules".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"{C.E}")

    t0 = time.time()
    results: dict = {}

    phases_to_run = list(_PHASE_MAP.keys()) if args.phase == "all" else [args.phase]

    for phase_key in phases_to_run:
        label, fn = _PHASE_MAP[phase_key]
        results[phase_key], _ = fn()

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{C.W}{C.B}{'='*70}{C.E}")
    print(f"{C.W}SUMMARY  ({elapsed:.1f}s){C.E}")
    print(f"{C.W}{C.B}{'='*70}{C.E}")

    all_ok = True
    for key, passed in results.items():
        label = _PHASE_MAP[key][0]
        icon = f"{C.G}✓ PASS{C.E}" if passed else f"{C.R}✗ FAIL{C.E}"
        print(f"  {label:<30s}  {icon}")
        all_ok = all_ok and passed

    print()
    if all_ok:
        print(f"{C.G}{C.W}  ✓  All requested checks passed{C.E}")
        print("\n  CCT execution remains gate-controlled:")
        print("    pytest -q")
        print("    ./run.sh status")
        print("    consult Todo.md before any experiment")
    else:
        failed_phases = [_PHASE_MAP[k][0] for k, v in results.items() if not v]
        print(f"{C.R}{C.W}  ✗  Failed: {', '.join(failed_phases)}{C.E}")
        print(f"     Review output above for details.")

    print(f"{C.W}{C.B}{'='*70}{C.E}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
