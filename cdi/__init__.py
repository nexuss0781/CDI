"""
Cohomodynamic Intelligence (CDI) — v2.0
=========================================

A post-neural intelligence engine based on sheaf cohomology, Dirac operators,
and heat equation dynamics on cognitive manifolds.

This is NOT a neural network. It uses:
  - Riemannian geometry for the cognitive manifold (M, g)
  - Sheaf theory for local-to-global information propagation
  - Hodge theory for inference (replaces attention/softmax)
  - Recurrent heat equation dynamics (replaces backprop + positional encoding)
  - Spectral sequences for hierarchical abstraction (O(n log n))

v2.0 Corrections (CDI_LM_v2_Technical_Specification.md):
  F1 — Differentiable inference: no .detach() in forward path
  F2 — Recurrent belief state Ψ with learnable initial condition theta_init
  F3 — Mandatory rebuild_operators() after every optimizer.step()
  F4 — Dimensional hierarchy: dim(B_0) >= embed_dim enforced by config

Complexity guarantees:
  O(1)       Reflex (point evaluation at inference)
  O(n)       Learning (heat equation Euler steps)
  O(n log n) Abstraction (spectral sequence algorithm)

Mathematical Framework: CDI Mathematical Specification v1.0 + v2.0 Corrective Spec
"""

from cdi.config import CDIConfig
from cdi.engine import CDIEngine
from cdi.tokenizer import CDITokenizer

__version__ = "2.0.0"
__all__ = ["CDIConfig", "CDIEngine", "CDITokenizer"]
