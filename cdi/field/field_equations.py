"""
§7.2 Cohomodynamic Field Equations
===================================

Definition 7.2.1:
    𝔸Ψ = 𝒥

Theorem 7.2.2 (Decomposition):
    D·ψₖ + δ^{k−1}·ψ_{k−1} + δ^{k*}·ψ_{k+1} + A·ψₖ = 𝒥ₖ

for each degree k.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from cdi.config import CDIConfig


class FieldEquations:
    """Cohomodynamic field equations 𝔸Ψ = 𝒥.

    Attributes
    ----------
    superconnection : Superconnection
    config : CDIConfig
    """

    def __init__(self, superconnection, config: CDIConfig) -> None:
        self.superconnection = superconnection
        self.config = config

    def solve(self, J: torch.Tensor) -> torch.Tensor:
        """Solve 𝔸Ψ = 𝒥 via least-squares.

        Parameters
        ----------
        J : torch.Tensor  Shape ``(N,)``.

        Returns
        -------
        torch.Tensor  Shape ``(N,)`` — solution Ψ.
        """
        A_mat = self.superconnection.matrix()
        # Least-squares: minimise ‖𝔸Ψ − 𝒥‖²
        result = torch.linalg.lstsq(A_mat, J.unsqueeze(-1))
        return result.solution.squeeze(-1)

    def residual(self, state: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """‖𝔸Ψ − 𝒥‖.

        Parameters
        ----------
        state : torch.Tensor  Shape ``(N,)``.
        J     : torch.Tensor  Shape ``(N,)``.

        Returns
        -------
        torch.Tensor  Scalar.
        """
        return torch.norm(self.superconnection.apply(state) - J)

    def decompose_by_degree(
        self, state: torch.Tensor, J: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Per-degree residuals of the field equation.

        Returns {k: residual_k} where residual_k = ‖(𝔸Ψ)_k − 𝒥_k‖.
        """
        belief = self.superconnection.belief
        n = self.config.n_points
        s = self.config.spinor_dim
        B = self.config.total_belief_dim

        A_psi = self.superconnection.apply(state)
        diff = A_psi - J

        # Reshape to (n, s, B) and split by degree
        diff_3d = diff.reshape(n, s, B)

        residuals = {}
        for k in belief.degrees:
            offset = self.config.belief_offset(k)
            dim_k = self.config.belief_dim(k)
            block = diff_3d[:, :, offset : offset + dim_k]
            residuals[k] = torch.norm(block)

        return residuals
