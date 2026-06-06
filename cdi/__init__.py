"""
Cohomodynamic Intelligence (CDI)
================================

A novel intelligence engine based on sheaf cohomology, Dirac operators,
and heat equation dynamics on cognitive manifolds.

This is NOT a neural network. It uses:
- Riemannian geometry for the cognitive manifold
- Sheaf theory for local-to-global information
- Hodge theory for inference (replaces attention/softmax)
- Heat equation for learning (replaces backpropagation)
- Spectral sequences for hierarchical abstraction

Complexity guarantees:
- Reflex (point evaluation): O(1)
- Learning (Čech update): O(n)
- Abstraction (spectral sequence): O(n log n)

Mathematical Framework: CDI Mathematical Specification v1.0
"""

from cdi.config import CDIConfig
from cdi.engine import CDIEngine
from cdi.tokenizer import CDITokenizer

__version__ = "1.0.0"
__all__ = ["CDIConfig", "CDIEngine", "CDITokenizer"]
