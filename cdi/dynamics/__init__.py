"""
CDI Dynamics Modules
====================

- §6   HeatEquation: ∂Ψ/∂t = −Δ_ℬ Ψ + 𝒥
- §6.2 SpectralDecomposition: eigenanalysis and heat semigroup
- §10  EnergyFunctional: E[Ψ] and dissipation
"""

from .heat_equation import HeatEquation
from .spectral import SpectralDecomposition
from .energy import EnergyFunctional

__all__ = ["HeatEquation", "SpectralDecomposition", "EnergyFunctional"]
