"""
CDI Operators
=============

- §5.1 BeliefLaplacian: Δ_ℬ = D² + Δ_δ + coupling + A²
- §5.2 HodgeDecomposition: Γ(𝔹) = ℋ(𝔹) ⊕ im(Δ_ℬ)
- §5.3 GreenOperator: G_ℬ — pseudo-inverse of Δ_ℬ
- §5.3 InferenceOperator: ℱ(s) = H(ι(s)) + δ* G_ℬ D* ι(s)
"""

from .laplacian import BeliefLaplacian
from .hodge import HodgeDecomposition
from .green import GreenOperator
from .inference import InferenceOperator

__all__ = ["BeliefLaplacian", "HodgeDecomposition", "GreenOperator", "InferenceOperator"]
