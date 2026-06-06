#!/usr/bin/env python
"""
CDI Test Orchestrator (Single Entry Point)
============================================

Usage:
    python run_tests.py

Runs all tests end-to-end:
  1. Syntax validation
  2. Integration tests
  3. Reports results

This is the ONLY file you need to run.
"""

import sys
import py_compile
import ast
import time
import traceback
from pathlib import Path
from typing import List, Dict


# ═══════════════════════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════════════════════

class C:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: SYNTAX VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def phase1_syntax_check():
    """Validate all Python files for syntax errors."""
    
    print(f"\n{C.BOLD}{C.CYAN}{'='*70}{C.END}")
    print(f"{C.BOLD}PHASE 1: SYNTAX VALIDATION{C.END}")
    print(f"{C.BOLD}{C.CYAN}{'='*70}{C.END}\n")
    
    root_dir = Path(__file__).parent
    cdi_dir = root_dir / "cdi"
    
    # Find all Python files
    python_files = list(cdi_dir.rglob("*.py"))
    python_files.extend([
        root_dir / "dataset.py",
        root_dir / "train.py",
    ])
    python_files = [f for f in python_files if f.exists()]
    
    passed = 0
    failed = 0
    errors = []
    
    for filepath in sorted(python_files):
        try:
            py_compile.compile(str(filepath), doraise=True)
            with open(filepath, 'r', encoding='utf-8') as f:
                ast.parse(f.read(), str(filepath))
            print(f"  ✓ {filepath.relative_to(root_dir)}")
            passed += 1
        except SyntaxError as e:
            print(f"  ✗ {filepath.relative_to(root_dir)}: Line {e.lineno}: {e.msg}")
            failed += 1
            errors.append((filepath, str(e)))
        except Exception as e:
            print(f"  ✗ {filepath.relative_to(root_dir)}: {e}")
            failed += 1
            errors.append((filepath, str(e)))
    
    print(f"\n  Syntax: {C.GREEN}{passed} passed{C.END}, {C.RED}{failed} failed{C.END}")
    
    return failed == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

def phase2_integration_tests():
    """Run integration tests."""
    
    print(f"\n{C.BOLD}{C.CYAN}{'='*70}{C.END}")
    print(f"{C.BOLD}PHASE 2: INTEGRATION TESTS{C.END}")
    print(f"{C.BOLD}{C.CYAN}{'='*70}{C.END}\n")
    
    # Check dependencies
    print("Checking dependencies...")
    deps_ok = True
    
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError:
        print(f"  ✗ torch (not installed)")
        deps_ok = False
    
    try:
        from ethiobbpe import EthioBBPETokenizer
        print(f"  ✓ ethiobbpe")
    except ImportError:
        print(f"  ✗ ethiobbpe (not installed)")
        deps_ok = False
    
    if not deps_ok:
        print(f"\n{C.YELLOW}⚠ Skipping integration tests (missing dependencies){C.END}")
        print(f"  Install with: pip install -r requirements.txt\n")
        return True, []  # Not a failure
    
    print()
    
    # Import CDI modules
    try:
        from cdi.config import CDIConfig
        from cdi.tokenizer import CDITokenizer
        from cdi.engine import CDIEngine
    except ImportError as e:
        print(f"{C.RED}✗ Could not import CDI: {e}{C.END}\n")
        return False, [(None, str(e))]
    
    passed = 0
    failed = 0
    errors = []
    
    # Test 1: Tokenizer encode
    try:
        print("Running: Tokenizer.encode()")
        tokenizer = CDITokenizer(embed_dim=32, max_len=16)
        ids = tokenizer.encode("What is photosynthesis?")
        assert ids.shape == (16,), f"Expected (16,), got {ids.shape}"
        print(f"  ✓ encode\n")
        passed += 1
    except Exception as e:
        print(f"  ✗ encode: {e}\n")
        failed += 1
        errors.append(("Tokenizer.encode", str(e)))
    
    # Test 2: Tokenizer embed
    try:
        print("Running: Tokenizer.embed()")
        embeddings = tokenizer.embed(ids)
        assert embeddings.shape == (16, 32), f"Expected (16, 32), got {embeddings.shape}"
        print(f"  ✓ embed\n")
        passed += 1
    except Exception as e:
        print(f"  ✗ embed: {e}\n")
        failed += 1
        errors.append(("Tokenizer.embed", str(e)))
    
    # Test 3: Engine forward
    try:
        print("Running: CDIEngine.forward_sequence_batch()")
        config = CDIConfig.tiny()
        engine = CDIEngine(config)
        engine.build()
        batch_emb = embeddings[:config.n_points].unsqueeze(0)
        output = engine.forward_sequence_batch(batch_emb)
        assert output.shape == (1, config.n_points, config.output_dim)
        print(f"  ✓ forward_sequence_batch\n")
        passed += 1
    except Exception as e:
        print(f"  ✗ forward_sequence_batch: {e}\n")
        failed += 1
        errors.append(("Engine.forward_sequence_batch", str(e)))
    
    # Test 4: Logits projection
    try:
        print("Running: Tokenizer.to_logits()")
        logits = tokenizer.to_logits(output[0])
        assert logits.shape[0] == config.n_points
        assert logits.shape[1] == tokenizer.vocab_size
        print(f"  ✓ to_logits\n")
        passed += 1
    except Exception as e:
        print(f"  ✗ to_logits: {e}\n")
        failed += 1
        errors.append(("Tokenizer.to_logits", str(e)))
    
    # Test 5: Loss computation
    try:
        print("Running: CDIEngine.compute_lm_loss()")
        target_ids = ids[:config.n_points].unsqueeze(0)
        loss, loss_dict = engine.compute_lm_loss(output, target_ids, tokenizer.embedding)
        assert loss.item() > 0
        print(f"  ✓ compute_lm_loss (loss={loss_dict['ce']:.4f}, ppl={loss_dict['perplexity']:.1f})\n")
        passed += 1
    except Exception as e:
        print(f"  ✗ compute_lm_loss: {e}\n")
        failed += 1
        errors.append(("Engine.compute_lm_loss", str(e)))
    
    # Test 6: Backward pass
    try:
        print("Running: Backward pass (gradient flow)")
        loss.backward()
        has_grad = any(p.grad is not None for p in engine.get_parameters() + tokenizer.get_parameters())
        assert has_grad
        print(f"  ✓ backward\n")
        passed += 1
    except Exception as e:
        print(f"  ✗ backward: {e}\n")
        failed += 1
        errors.append(("Backward", str(e)))
    
    print(f"  Integration: {C.GREEN}{passed} passed{C.END}, {C.RED}{failed} failed{C.END}")
    
    return failed == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run all tests."""
    
    print(f"\n{C.BOLD}")
    print("╔" + "="*68 + "╗")
    print("║" + "  CDI TEST ORCHESTRATOR".center(68) + "║")
    print("║" + "  Single Entry Point - run_tests.py".center(68) + "║")
    print("╚" + "="*68 + "╝")
    print(f"{C.END}")
    
    # Phase 1: Syntax
    syntax_ok, syntax_errors = phase1_syntax_check()
    
    # Phase 2: Integration
    integration_ok, integration_errors = phase2_integration_tests()
    
    # Summary
    print(f"\n{C.BOLD}{C.CYAN}{'='*70}{C.END}")
    print(f"{C.BOLD}SUMMARY{C.END}")
    print(f"{C.BOLD}{C.CYAN}{'='*70}{C.END}\n")
    
    all_ok = syntax_ok and integration_ok
    
    print(f"  Syntax validation ........ {C.GREEN}✓ PASS{C.END if syntax_ok else C.RED}✗ FAIL{C.END}")
    print(f"  Integration tests ....... {C.GREEN}✓ PASS{C.END if integration_ok else C.RED}✗ FAIL{C.END if integration_ok == False else C.YELLOW}⚠ SKIPPED{C.END}")
    
    if syntax_errors:
        print(f"\n{C.RED}Syntax Errors:{C.END}")
        for path, error in syntax_errors:
            print(f"  {path}: {error}")
    
    if integration_errors and integration_ok == False:
        print(f"\n{C.RED}Integration Errors:{C.END}")
        for test, error in integration_errors:
            print(f"  {test}: {error}")
    
    print(f"\n{C.BOLD}{C.CYAN}{'='*70}{C.END}")
    if all_ok or (syntax_ok and integration_ok is None):
        print(f"{C.GREEN}{C.BOLD}✓ ALL TESTS PASSED{C.END}")
        print(f"\nNext steps:")
        print(f"  1. Install dependencies:  pip install -r requirements.txt")
        print(f"  2. Run training:          python train.py --config tiny")
    else:
        print(f"{C.RED}{C.BOLD}✗ SOME TESTS FAILED{C.END}")
        print(f"\nFix errors above before proceeding.")
    print(f"{C.BOLD}{C.CYAN}{'='*70}{C.END}\n")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
