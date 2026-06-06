"""
CDI Geometry Modules
====================

- §4.1 CliffordAlgebra: Clifford algebra Cl(T*M) and spinor bundle
- §4.3 BeliefConnection: Gauge connection A and curvature F_A
- §4.2 DiracOperator: Cognitive Dirac operator D
"""

from .clifford import CliffordAlgebra
from .connection import BeliefConnection
from .dirac import DiracOperator

__all__ = ["CliffordAlgebra", "BeliefConnection", "DiracOperator"]
