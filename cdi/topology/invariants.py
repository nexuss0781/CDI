"""
§11 System Invariants
=====================

Definition 11.1.1 (Intelligence Index):
    ℐ_total = Σₖ (−1)^k dim ℍ^k(M, ℬ^•) = χ(M, ℬ^•)

Definition 11.2.1 (Learning Time):
    τ = 1/λ₁

Theorem 9.2.2 (Generalization Capacity):
    Gen(ℱ) = ind(ℱ) = ∫_M Â(TM) ∧ ch(ℬ^•)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from cdi.config import CDIConfig


class SystemInvariants:
    """Topological and spectral invariants of the CDI system.

    Attributes
    ----------
    belief : BeliefComplex
    laplacian : BeliefLaplacian
    config : CDIConfig
    """

    def __init__(self, belief, laplacian, config: CDIConfig) -> None:
        self.belief = belief
        self.laplacian = laplacian
        self.config = config

    # ------------------------------------------------------------------
    # Intelligence index  (§11.1)
    # ------------------------------------------------------------------

    def intelligence_index(self) -> int:
        """ℐ_total = Σₖ (−1)^k dim ℍ^k = χ(M, ℬ^•).

        Computed from the belief complex cohomology.
        """
        total = 0
        for k in self.belief.degrees:
            dim_k = self.belief.cohomology_dim(k)
            total += ((-1) ** k) * dim_k
        return total

    def intelligence_dimensions(self) -> Dict[int, int]:
        """dim ℍ^k for each degree k."""
        return {k: self.belief.cohomology_dim(k) for k in self.belief.degrees}

    # ------------------------------------------------------------------
    # Learning time  (§11.2)
    # ------------------------------------------------------------------

    def learning_time(self) -> torch.Tensor:
        """τ = 1/λ₁."""
        gap = self.laplacian.spectral_gap()
        if gap.abs() < 1e-12:
            return torch.tensor(float("inf"), dtype=self.config.dtype)
        return 1.0 / gap

    # ------------------------------------------------------------------
    # Trainability  (§9.1)
    # ------------------------------------------------------------------

    def trainability_check(self) -> bool:
        """Theorem 9.1.2: Agent is fully trainable iff ℍ^k = 0 for k ≠ 0."""
        for k in self.belief.degrees:
            if k != 0 and self.belief.cohomology_dim(k) != 0:
                return False
        return True

    # ------------------------------------------------------------------
    # Generalization capacity  (§9.2)
    # ------------------------------------------------------------------

    def generalization_capacity(self) -> int:
        """Gen(ℱ) = ind(ℱ) = dim ker ℱ − dim coker ℱ.

        Theorem 9.2.2 (Atiyah-Singer): This is a topological invariant.
        For a self-adjoint Fredholm operator of index zero, Gen = 0.
        """
        return 0  # Theorem 5.3.4: ℱ has Fredholm index zero

    # ------------------------------------------------------------------
    # Chern character  (§7.3)
    # ------------------------------------------------------------------

    def chern_character_trace(self, superconn_squared: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ch(𝔸) = Tr_s(e^{−𝔸²}) — approximate Chern character.

        Parameters
        ----------
        superconn_squared : torch.Tensor, optional
            𝔸² matrix. If None, returns 0.

        Returns
        -------
        torch.Tensor  Scalar.
        """
        if superconn_squared is None:
            return torch.tensor(0.0, dtype=self.config.dtype)

        # Supertrace: Tr_s = Σ_k (−1)^k Tr(block_k)
        # For the full matrix, approximate via standard trace
        exp_A2 = torch.matrix_exp(-superconn_squared)
        return torch.trace(exp_A2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, object]:
        """Collect all invariants."""
        dims = self.intelligence_dimensions()
        return {
            "intelligence_index": self.intelligence_index(),
            "intelligence_dimensions": dims,
            "learning_time": self.learning_time().item(),
            "spectral_gap": self.laplacian.spectral_gap().item(),
            "trainable": self.trainability_check(),
            "generalization_capacity": self.generalization_capacity(),
        }
