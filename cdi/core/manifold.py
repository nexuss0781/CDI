"""
§1 Cognitive Site — Discretised Riemannian Manifold
====================================================

Implements the cognitive site (M, T) from CDI Specification §1.1.1.

M is a compact oriented Riemannian manifold of dimension d with metric g.
Discretisation: M → {p₁, …, p_n} ⊂ ℝᵈ.
The metric g_i at each point p_i is a d×d SPD matrix,
parameterised as g = L·Lᵀ (Cholesky) for guaranteed SPD.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from cdi.config import CDIConfig


class CognitiveManifold:
    """Discretised Riemannian manifold (M, g).

    §1.1.1: A cognitive site is (M, T) where M is a compact oriented
    Riemannian manifold of dimension d with metric g.

    Discretisation: M → {p₁, …, p_n} ⊂ ℝᵈ
    The metric g_i at each point p_i is a d×d SPD matrix,
    parameterised as g = L·Lᵀ (Cholesky) for guaranteed SPD.

    Attributes
    ----------
    config : CDIConfig
        Engine-wide configuration dataclass.
    d : int
        Manifold dimension.
    n : int
        Number of discretisation points.
    points : torch.Tensor
        Learnable point cloud, shape ``(n, d)``.
    metric_L : torch.Tensor
        Lower-triangular Cholesky factors for the metric, shape ``(n, d, d)``.
    """

    def __init__(self, config: CDIConfig) -> None:
        self.config = config
        self.d: int = config.manifold_dim
        self.n: int = config.n_points
        dtype = config.dtype

        assert self.d >= 1, f"manifold_dim must be ≥ 1, got {self.d}"
        assert self.n >= 2, f"n_points must be ≥ 2, got {self.n}"

        # ------------------------------------------------------------------
        # Manifold points — initialised on unit cube, learnable
        # ------------------------------------------------------------------
        torch.manual_seed(config.seed)
        self.points: torch.Tensor = torch.randn(self.n, self.d, dtype=dtype) * 0.5
        self.points.requires_grad_(True)

        # ------------------------------------------------------------------
        # Metric: g = L @ Lᵀ,  L is lower-triangular per point
        # Initialise as identity: L = I
        # ------------------------------------------------------------------
        L_init = (
            torch.eye(self.d, dtype=dtype)
            .unsqueeze(0)
            .expand(self.n, -1, -1)
            .clone()
        )
        self.metric_L: torch.Tensor = L_init.requires_grad_(True)  # (n, d, d)

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def metric(self) -> torch.Tensor:
        """Riemannian metric g = L Lᵀ.

        Returns
        -------
        torch.Tensor
            Shape ``(n, d, d)`` — SPD matrices.
        """
        L = torch.tril(self.metric_L)  # enforce lower-triangular
        # Ensure strictly positive diagonal for numerical stability
        diag_mask = torch.eye(
            self.d, dtype=self.config.dtype, device=self.metric_L.device
        )
        L = L * (1 - diag_mask) + torch.abs(L * diag_mask) + 1e-6 * diag_mask
        return L @ L.transpose(-2, -1)

    def inverse_metric(self) -> torch.Tensor:
        """Inverse metric g⁻¹.

        Returns
        -------
        torch.Tensor
            Shape ``(n, d, d)``.
        """
        return torch.linalg.inv(self.metric())

    def volume_element(self) -> torch.Tensor:
        """Volume element √det(g) at each point.

        Returns
        -------
        torch.Tensor
            Shape ``(n,)``.
        """
        return torch.sqrt(torch.linalg.det(self.metric()).clamp(min=1e-12))

    # ------------------------------------------------------------------
    # Christoffel symbols  Γᵃ_{bc}
    # ------------------------------------------------------------------

    def christoffel_symbols(self) -> torch.Tensor:
        r"""Christoffel symbols Γᵃ_{bc} via weighted finite differences.

        .. math::

            \Gamma^a_{bc} = \tfrac12\, g^{ad}\!\bigl(
                \partial_b g_{dc} + \partial_c g_{db} - \partial_d g_{bc}
            \bigr)

        Returns
        -------
        torch.Tensor
            Shape ``(n, d, d, d)`` — ``Gamma[point, a, b, c]``.
        """
        g = self.metric()          # (n, d, d)
        g_inv = self.inverse_metric()  # (n, d, d)
        pts = self.points          # (n, d)

        # ── approximate metric derivatives ──────────────────────────
        # ∂g_{ab}/∂xᶜ at point i ≈ weighted average of
        #   (g_j − g_i) / (x_j − x_i)ᶜ  over neighbours j
        dists = torch.cdist(pts, pts)  # (n, n)

        # Gaussian kernel weights
        sigma = dists.mean() * 0.5 + 1e-12
        weights = torch.exp(-dists ** 2 / (2 * sigma ** 2))  # (n, n)
        weights.fill_diagonal_(0)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        dx = pts.unsqueeze(1) - pts.unsqueeze(0)  # (n, n, d)
        dg = g.unsqueeze(1) - g.unsqueeze(0)      # (n, n, d, d)
        dx_norm_sq = (dx ** 2).sum(dim=-1).clamp(min=1e-10)  # (n, n)

        # partial_g[i, a, b, c] = Σ_j w_ij · dg[i,j,a,b] · dx[i,j,c] / |dx|²
        partial_g = torch.zeros(
            self.n, self.d, self.d, self.d, dtype=self.config.dtype
        )
        for c in range(self.d):
            w_dx_c = weights * dx[:, :, c] / dx_norm_sq.clamp(min=1e-10)
            partial_g[:, :, :, c] = torch.einsum("ij,ijab->iab", w_dx_c, dg)

        # ── assemble Christoffel symbols ────────────────────────────
        # Γᵃ_{bc} = ½ g^{ad} (∂_b g_{dc} + ∂_c g_{db} − ∂_d g_{bc})
        Gamma = torch.zeros(
            self.n, self.d, self.d, self.d, dtype=self.config.dtype
        )
        for a in range(self.d):
            for b in range(self.d):
                for c in range(self.d):
                    val = torch.zeros(self.n, dtype=self.config.dtype)
                    for dd in range(self.d):
                        val = val + g_inv[:, a, dd] * (
                            partial_g[:, dd, c, b]
                            + partial_g[:, dd, b, c]
                            - partial_g[:, b, c, dd]
                        )
                    Gamma[:, a, b, c] = 0.5 * val
        return Gamma

    # ------------------------------------------------------------------
    # Scalar curvature
    # ------------------------------------------------------------------

    def scalar_curvature(self) -> torch.Tensor:
        r"""Scalar curvature *R* at each point.

        Approximates the Riemann tensor via the Christoffel-product terms
        (dominant for slowly-varying metrics):

        .. math::

            R^a_{bcd} \approx \Gamma^a_{ce}\,\Gamma^e_{bd}
                              - \Gamma^a_{de}\,\Gamma^e_{bc}

        Then the Ricci tensor R_{ac} = R^b_{abc} and R = g^{ac} R_{ac}.

        Returns
        -------
        torch.Tensor
            Shape ``(n,)`` — scalar curvature per point.
        """
        g_inv = self.inverse_metric()
        Gamma = self.christoffel_symbols()

        # Approximate Ricci tensor via Γ-product terms
        R_ricci = torch.zeros(self.n, self.d, self.d, dtype=self.config.dtype)
        for a in range(self.d):
            for c in range(self.d):
                for e in range(self.d):
                    for b in range(self.d):
                        R_ricci[:, a, c] += (
                            Gamma[:, b, a, e] * Gamma[:, e, c, b]
                            - Gamma[:, b, c, e] * Gamma[:, e, a, b]
                        )

        # Scalar curvature R = g^{ac} R_{ac}
        R = torch.einsum("iac,iac->i", g_inv, R_ricci)
        return R

    # ------------------------------------------------------------------
    # Orthonormal frame
    # ------------------------------------------------------------------

    def orthonormal_frame(self) -> torch.Tensor:
        """Orthonormal frame {eᵢ} at each point via Cholesky of g⁻¹.

        g⁻¹ = E Eᵀ  ⟹  columns of E are orthonormal w.r.t. g.

        Returns
        -------
        torch.Tensor
            Shape ``(n, d, d)`` — ``frame[p, i, :]`` is the i-th
            basis vector at point *p*.
        """
        g_inv = self.inverse_metric()
        # Add small diagonal jitter for numerical stability
        jitter = 1e-8 * torch.eye(self.d, dtype=self.config.dtype, device=g_inv.device)
        E = torch.linalg.cholesky(g_inv + jitter)
        return E

    # ------------------------------------------------------------------
    # Inner products
    # ------------------------------------------------------------------

    def inner_product(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        point_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Riemannian inner product ⟨u, v⟩_g.

        Parameters
        ----------
        u, v : torch.Tensor
            Tangent vectors — shape ``(d,)`` (single point) or ``(n, d)``
            (all points simultaneously).
        point_idx : int, optional
            If given, evaluate at a single point; *u* and *v* are ``(d,)``.

        Returns
        -------
        torch.Tensor
            Scalar (single point) or ``(n,)`` (all points).
        """
        g = self.metric()  # (n, d, d)
        if point_idx is not None:
            assert 0 <= point_idx < self.n, f"point_idx {point_idx} out of range"
            return u @ g[point_idx] @ v
        if u.dim() > 1:
            return torch.einsum("...i,nij,...j->...", u, g, v)
        return u @ g @ v

    def l2_inner_product(
        self, psi: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        r"""L² inner product ⟨ψ, φ⟩ = ∫_M h(ψ, φ)\, \mathrm{dvol}_g.

        Parameters
        ----------
        psi, phi : torch.Tensor
            Sections over M — shape ``(n,)`` or ``(n, *)``.

        Returns
        -------
        torch.Tensor
            Scalar.
        """
        vol = self.volume_element()  # (n,)
        if psi.dim() == 1 and phi.dim() == 1:
            return (psi * phi * vol).sum()
        # Multi-dim sections: flatten trailing dims
        flat_psi = psi.reshape(self.n, -1)
        flat_phi = phi.reshape(self.n, -1)
        pointwise = (flat_psi * flat_phi).sum(dim=-1)  # (n,)
        return (pointwise * vol).sum()

    # ------------------------------------------------------------------
    # Learnable parameters
    # ------------------------------------------------------------------

    def get_parameters(self) -> List[torch.Tensor]:
        """Learnable parameters: points and metric Cholesky factors."""
        return [self.points, self.metric_L]
