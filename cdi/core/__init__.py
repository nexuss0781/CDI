"""
CDI Core Modules
================

Core mathematical structures for the Cohomodynamic Intelligence engine.

- §1 CognitiveManifold: Discretised Riemannian manifold (M, g)
- §2 GoodCover: Čech-type open cover from k-NN balls
- §2 ObservationSheaf: Sheaf of observations over the cover
- §3 BeliefComplex: Graded cochain complex B^• with coboundary δ
"""

from .manifold import CognitiveManifold
from .cover import GoodCover
from .sheaf import ObservationSheaf
from .belief import BeliefComplex

__all__ = [
    "CognitiveManifold",
    "GoodCover",
    "ObservationSheaf",
    "BeliefComplex",
]
