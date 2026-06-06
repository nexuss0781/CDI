"""Tests for operators (§5): Laplacian, Hodge, Green, Inference."""

import torch
import pytest


class TestDiracOperator:

    def test_matrix_shape(self, dirac, tiny_config):
        N = tiny_config.total_state_dim
        assert dirac.matrix().shape == (N, N)

    def test_self_adjoint(self, dirac):
        err = dirac.check_self_adjoint()
        assert err < 1e-8, f"Dirac not self-adjoint: ‖D−Dᵀ‖ = {err}"

    def test_apply(self, dirac, tiny_config):
        N = tiny_config.total_state_dim
        psi = torch.randn(N, dtype=tiny_config.dtype)
        result = dirac.apply(psi)
        assert result.shape == (N,)


class TestBeliefLaplacian:

    def test_matrix_shape(self, laplacian, tiny_config):
        N = tiny_config.total_state_dim
        assert laplacian.matrix().shape == (N, N)

    def test_self_adjoint(self, laplacian):
        err = laplacian.check_self_adjoint()
        assert err < 1e-8, f"Laplacian not self-adjoint: {err}"

    def test_positive_semidefinite(self, laplacian):
        assert laplacian.check_positive_semidefinite()

    def test_eigendecompose(self, laplacian, tiny_config):
        evals, evecs = laplacian.eigendecompose()
        N = tiny_config.total_state_dim
        assert evals.shape == (N,)
        assert evecs.shape == (N, N)
        assert (evals >= -1e-8).all()

    def test_spectral_gap(self, laplacian):
        gap = laplacian.spectral_gap()
        assert gap >= 0


class TestHodgeDecomposition:

    def test_decomposition_is_orthogonal(self, hodge, tiny_config):
        N = tiny_config.total_state_dim
        psi = torch.randn(N, dtype=tiny_config.dtype)
        h, nh = hodge.decompose(psi)
        # Orthogonality: h · nh ≈ 0
        assert torch.abs(torch.dot(h, nh)) < 1e-6
        # Reconstruction
        assert torch.allclose(h + nh, psi, atol=1e-8)

    def test_harmonic_projector_idempotent(self, hodge, tiny_config):
        H = hodge.harmonic_projector()
        H2 = H @ H
        assert torch.allclose(H, H2, atol=1e-8)

    def test_harmonic_dimension(self, hodge):
        dim = hodge.harmonic_dimension()
        assert dim >= 0


class TestGreenOperator:

    def test_pseudo_inverse(self, green):
        """G·Δ + H ≈ I."""
        err = green.verify()
        assert err < 1e-6, f"Green pseudo-inverse error: {err}"

    def test_apply(self, green, tiny_config):
        N = tiny_config.total_state_dim
        f = torch.randn(N, dtype=tiny_config.dtype)
        result = green.apply(f)
        assert result.shape == (N,)


class TestInferenceOperator:

    def test_infer_shape(self, built_engine):
        cfg = built_engine.config
        obs = torch.randn(cfg.n_points, cfg.observation_dim, dtype=cfg.dtype)
        result = built_engine.inference_op.infer(obs)
        assert result.shape == (cfg.n_points, cfg.output_dim)

    def test_embed_shape(self, built_engine):
        cfg = built_engine.config
        obs = torch.randn(cfg.n_points, cfg.observation_dim, dtype=cfg.dtype)
        embedded = built_engine.inference_op.embed_observation(obs)
        assert embedded.shape == (cfg.total_state_dim,)
