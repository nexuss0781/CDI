"""Tests for §1 CognitiveManifold."""

import torch
import pytest


class TestCognitiveManifold:

    def test_points_shape(self, manifold, tiny_config):
        assert manifold.points.shape == (tiny_config.n_points, tiny_config.manifold_dim)

    def test_metric_spd(self, manifold, tiny_config):
        """Metric must be symmetric positive definite."""
        g = manifold.metric()
        assert g.shape == (tiny_config.n_points, tiny_config.manifold_dim, tiny_config.manifold_dim)
        # Symmetric
        assert torch.allclose(g, g.transpose(-2, -1), atol=1e-12)
        # Positive definite: all eigenvalues > 0
        evals = torch.linalg.eigvalsh(g)
        assert (evals > 0).all(), f"Non-positive eigenvalue: {evals.min()}"

    def test_inverse_metric(self, manifold, tiny_config):
        """g · g⁻¹ = I."""
        g = manifold.metric()
        g_inv = manifold.inverse_metric()
        I = torch.eye(tiny_config.manifold_dim, dtype=tiny_config.dtype)
        product = g @ g_inv
        for i in range(tiny_config.n_points):
            assert torch.allclose(product[i], I, atol=1e-8)

    def test_volume_positive(self, manifold, tiny_config):
        vol = manifold.volume_element()
        assert vol.shape == (tiny_config.n_points,)
        assert (vol > 0).all()

    def test_christoffel_shape(self, manifold, tiny_config):
        d = tiny_config.manifold_dim
        Gamma = manifold.christoffel_symbols()
        assert Gamma.shape == (tiny_config.n_points, d, d, d)

    def test_scalar_curvature_shape(self, manifold, tiny_config):
        R = manifold.scalar_curvature()
        assert R.shape == (tiny_config.n_points,)
        assert torch.isfinite(R).all()

    def test_orthonormal_frame(self, manifold, tiny_config):
        frame = manifold.orthonormal_frame()
        d = tiny_config.manifold_dim
        assert frame.shape == (tiny_config.n_points, d, d)

    def test_l2_inner_product(self, manifold, tiny_config):
        N = tiny_config.n_points
        psi = torch.randn(N, dtype=tiny_config.dtype)
        ip = manifold.l2_inner_product(psi, psi)
        assert ip > 0  # non-negative for non-zero vector

    def test_parameters(self, manifold):
        params = manifold.get_parameters()
        assert len(params) == 2
        assert all(p.requires_grad for p in params)


class TestBeliefComplex:

    def test_dims(self, belief, tiny_config):
        assert len(belief.dims) == tiny_config.n_degrees
        assert len(belief.deltas) == tiny_config.n_degrees - 1

    def test_coboundary_shapes(self, belief, tiny_config):
        for i in range(len(belief.deltas)):
            assert belief.deltas[i].shape == (belief.dims[i + 1], belief.dims[i])

    def test_coboundary_apply(self, belief, tiny_config):
        for k in belief.degrees[:-1]:
            idx = belief.degree_to_index(k)
            x = torch.randn(belief.dims[idx], dtype=tiny_config.dtype)
            y = belief.coboundary(k, x)
            assert y.shape == (belief.dims[idx + 1],)

    def test_combinatorial_laplacian_symmetric(self, belief, tiny_config):
        for k in belief.degrees:
            lap = belief.combinatorial_laplacian_at_degree(k)
            assert torch.allclose(lap, lap.T, atol=1e-12)

    def test_full_laplacian_shape(self, belief, tiny_config):
        B = tiny_config.total_belief_dim
        lap = belief.full_combinatorial_laplacian()
        assert lap.shape == (B, B)

    def test_consistency_penalty_finite(self, belief):
        p = belief.consistency_penalty()
        assert torch.isfinite(p)
        assert p >= 0

    def test_split_assemble_roundtrip(self, belief, tiny_config):
        B = tiny_config.total_belief_dim
        state = torch.randn(B, dtype=tiny_config.dtype)
        parts = belief.split_state(state)
        reassembled = belief.assemble_state(parts)
        assert torch.allclose(state, reassembled)

    def test_parameters(self, belief, tiny_config):
        params = belief.get_parameters()
        assert len(params) == tiny_config.n_degrees - 1


class TestCliffordAlgebra:

    def test_gamma_count(self, clifford, tiny_config):
        assert len(clifford.flat_gammas) == tiny_config.manifold_dim

    def test_gamma_shapes(self, clifford, tiny_config):
        s = tiny_config.spinor_dim
        for g in clifford.flat_gammas:
            assert g.shape == (s, s)

    def test_clifford_relations_flat(self, clifford):
        """Anti-commutation: {γⁱ, γʲ} = −2δⁱʲ I  (flat space)."""
        err = clifford.verify_relations()
        assert err < 1e-10, f"Clifford relation error: {err}"

    def test_clifford_action(self, clifford, tiny_config):
        s = tiny_config.spinor_dim
        d = tiny_config.manifold_dim
        xi = torch.randn(d, dtype=tiny_config.dtype)
        psi = torch.randn(s, dtype=tiny_config.dtype)
        result = clifford.clifford_action(xi, psi)
        assert result.shape == (s,)


class TestCover:

    def test_patches(self, cover, tiny_config):
        assert len(cover.patches) == tiny_config.n_points
        for p in cover.patches:
            assert len(p) == tiny_config.cover_k

    def test_adjacency(self, cover, tiny_config):
        n = tiny_config.n_points
        assert cover.adjacency.shape == (n, n)

    def test_edges_nonempty(self, cover):
        assert len(cover.edges) > 0

    def test_intersection(self, cover):
        if cover.edges:
            i, j = cover.edges[0]
            inter = cover.intersection(i, j)
            assert len(inter) > 0

    def test_hierarchical_tree(self, cover):
        tree = cover.build_hierarchical_tree()
        assert "levels" in tree
        assert "root" in tree
        assert tree["root"] is not None
