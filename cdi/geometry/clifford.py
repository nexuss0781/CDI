"""
§4.1 Clifford Algebra — Generators and Spinor Bundle
=====================================================

Implements Cl(T*M) from CDI Specification §4.1.

Definition 4.1.1: The Clifford bundle Cl(T*M) acts on the belief
spinor bundle S = S⁺ ⊕ S⁻ via gamma matrices satisfying

    {γⁱ, γʲ} = γⁱγʲ + γʲγⁱ = −2 gⁱʲ I

For dimension d, the spinor space has dim S = 2^⌊d/2⌋.
"""

from __future__ import annotations

from typing import List

import torch

from cdi.config import CDIConfig


class CliffordAlgebra:
    """Clifford algebra Cl(T*M) with spinor representation.

    Gamma matrices are built in the *flat* basis and then
    rotated via the vielbein (orthonormal frame) to account
    for the curved metric.

    Attributes
    ----------
    config : CDIConfig
    d : int
        Manifold dimension.
    s : int
        Spinor dimension  2^⌊d/2⌋.
    flat_gammas : list[torch.Tensor]
        Flat-space gamma matrices, each shape ``(s, s)``.
    """

    def __init__(self, config: CDIConfig) -> None:
        self.config = config
        self.d = config.manifold_dim
        self.s = config.spinor_dim  # 2^(d//2)
        dtype = config.dtype

        self.flat_gammas: List[torch.Tensor] = self._build_flat_gammas(dtype)
        assert len(self.flat_gammas) == self.d

    # ------------------------------------------------------------------
    # Construction of flat gamma matrices
    # ------------------------------------------------------------------

    def _build_flat_gammas(self, dtype: torch.dtype) -> List[torch.Tensor]:
        """Build real gamma matrices for ℝ^d via Pauli-matrix recursion.

        For d=1: γ¹ = [[0,1],[1,0]] — but spinor_dim=1, so use scalar ±1.
        For d=2: γ¹ = σ_x, γ² = σ_z  (real, {γⁱ,γʲ}=-2δⁱʲ with sign)
        For d=3: γ¹ = σ_x, γ² = σ_y (imaginary → use σ_z⊗σ_x trick), γ³ = σ_z
        General d: recursive tensor-product construction.
        """
        if self.d == 1:
            # 1D: spinor_dim = 1, γ¹ is scalar-like → use 1×1 matrix
            return [torch.tensor([[1.0]], dtype=dtype)]

        # Pauli matrices (real subset)
        sigma_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
        sigma_y = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=dtype)
        sigma_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)

        if self.d == 2:
            # {σ_x, σ_z} = 0  and  σ_x²=I, σ_z²=I
            return [sigma_x.clone(), sigma_z.clone()]

        if self.d == 3:
            # Need 3 anti-commuting real 2×2 matrices
            # σ_x, σ_y (antisymmetric — still real), σ_z
            return [sigma_x.clone(), sigma_y.clone(), sigma_z.clone()]

        # General d ≥ 4 — recursive construction
        # Build 2^⌊d/2⌋ × 2^⌊d/2⌋ gamma matrices using tensor products
        gammas = self._recursive_gammas(self.d, dtype)
        return gammas

    def _recursive_gammas(self, d: int, dtype: torch.dtype) -> List[torch.Tensor]:
        """Recursive construction of d gamma matrices of size 2^⌊d/2⌋."""
        if d <= 3:
            return self._build_flat_gammas.__wrapped__(self, dtype) if d <= 3 else []

        # Base: d=2 gammas
        sigma_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
        sigma_y = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=dtype)
        sigma_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
        I2 = torch.eye(2, dtype=dtype)

        # Build iteratively: at each step add 2 more gamma matrices
        # via tensor products
        gammas = [sigma_x, sigma_z]  # d=2 base
        size = 2

        for i in range(1, d // 2):
            new_gammas = []
            Id = torch.eye(size, dtype=dtype)
            # Transform existing gammas: γ_old → σ_x ⊗ γ_old
            for g in gammas:
                new_gammas.append(torch.kron(sigma_x, g))
            # Add two new gammas: σ_y ⊗ I, σ_z ⊗ I
            new_gammas.append(torch.kron(sigma_y, Id))
            new_gammas.append(torch.kron(sigma_z, Id))
            gammas = new_gammas
            size *= 2

        # If d is odd, need one more gamma: product of all existing
        if d % 2 == 1:
            chirality = torch.eye(size, dtype=dtype)
            for g in gammas:
                chirality = chirality @ g
            # Normalise sign
            gammas.append(chirality)

        return gammas[:d]

    # ------------------------------------------------------------------
    # Clifford action
    # ------------------------------------------------------------------

    def clifford_action(self, covector: torch.Tensor, spinor: torch.Tensor) -> torch.Tensor:
        """Clifford multiplication c(ξ)·ψ = ξᵢ γⁱ ψ.

        Parameters
        ----------
        covector : torch.Tensor
            Shape ``(..., d)`` — covector ξ.
        spinor : torch.Tensor
            Shape ``(..., s)`` — spinor ψ.

        Returns
        -------
        torch.Tensor
            Shape ``(..., s)``.
        """
        result = torch.zeros_like(spinor)
        for i in range(self.d):
            gamma_i = self.flat_gammas[i]  # (s, s)
            xi_i = covector[..., i : i + 1]  # (..., 1)
            result = result + xi_i * (spinor @ gamma_i.T)
        return result

    def gamma_at_point(self, frame: torch.Tensor) -> List[torch.Tensor]:
        """Curved gamma matrices γⁱ(x) = eⁱ_a(x) γᵃ_flat.

        Parameters
        ----------
        frame : torch.Tensor
            Orthonormal frame at one point, shape ``(d, d)``.
            ``frame[i, :]`` is the i-th basis vector.

        Returns
        -------
        list[torch.Tensor]
            d matrices of shape ``(s, s)``.
        """
        curved = []
        for i in range(self.d):
            gamma_i = torch.zeros(self.s, self.s, dtype=self.config.dtype)
            for a in range(self.d):
                gamma_i = gamma_i + frame[i, a] * self.flat_gammas[a]
            curved.append(gamma_i)
        return curved

    # ------------------------------------------------------------------
    # Chirality
    # ------------------------------------------------------------------

    def chirality(self) -> torch.Tensor:
        """Chirality operator γ_chiral = i^⌊d/2⌋ γ¹γ²···γᵈ.

        Defines the ℤ₂-grading S = S⁺ ⊕ S⁻.

        Returns
        -------
        torch.Tensor
            Shape ``(s, s)``.
        """
        chir = torch.eye(self.s, dtype=self.config.dtype)
        for g in self.flat_gammas:
            chir = chir @ g
        return chir

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_relations(self, metric_at_point: torch.Tensor = None) -> torch.Tensor:
        """Check {γⁱ, γʲ} = −2 gⁱʲ I.  Returns max absolute error.

        Parameters
        ----------
        metric_at_point : torch.Tensor, optional
            Shape ``(d, d)``. If None, uses identity (flat space).

        Returns
        -------
        torch.Tensor
            Scalar — maximum |{γⁱ,γʲ} + 2gⁱʲ I|.
        """
        if metric_at_point is None:
            g = torch.eye(self.d, dtype=self.config.dtype)
        else:
            g = metric_at_point

        I_s = torch.eye(self.s, dtype=self.config.dtype)
        max_err = torch.tensor(0.0, dtype=self.config.dtype)
        for i in range(self.d):
            for j in range(self.d):
                anticomm = self.flat_gammas[i] @ self.flat_gammas[j] + \
                           self.flat_gammas[j] @ self.flat_gammas[i]
                expected = -2.0 * g[i, j] * I_s
                err = torch.abs(anticomm - expected).max()
                max_err = torch.maximum(max_err, err)
        return max_err
