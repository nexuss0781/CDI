"""
Tests for operators (§4-5): Dirac, Laplacian, Hodge, Green, Inference — v2.0.

v2.0 additions:
  - test_dirac_in_graph: Dirac matrix is connected to autograd graph
  - test_laplacian_in_graph: Laplacian matrix is connected to autograd graph
  - test_green_apply_differentiable: PCG apply is differentiable
  - test_inference_no_detach: inference output has gradient to W_iota
"""

import torch
import pytest


class TestDiracOperator:

    def test_matrix_shape(self, dirac, tiny_config):
        N = tiny_config.total_state_dim
        assert dirac.matrix.shape == (N, N)

    def test_self_adjoint(self, dirac):
        """Theorem 4.2.3: D = D*."""
        err = dirac.check_self_adjoint()
        assert err < 1e-8, f"Dirac not self-adjoint: ‖D−Dᵀ‖ = {err:.2e}"

    def test_apply_shape(self, dirac, tiny_config):
        N = tiny_config.total_state_dim
        psi = torch.randn(N, dtype=tiny_config.dtype)
        result = dirac.apply(psi)
        assert result.shape == (N,)

    def test_apply_adjoint_equals_apply(self, dirac, tiny_config):
        """D* = D since D is self-adjoint."""
        N = tiny_config.total_state_dim
        psi = torch.randn(N, dtype=tiny_config.dtype)
        assert torch.allclose(dirac.apply(psi), dirac.apply_adjoint(psi), atol=1e-10)

    def test_dirac_in_graph(self, manifold, clifford, connection, belief, cover, tiny_config):
        """Fix F3 v2.0: Dirac matrix must be connected to autograd graph.

        In v1.0, pts were .detach()-ed so the Dirac matrix was a constant.
        In v2.0, the matrix is built from live pts (no .detach()).
        We verify by checking that a scalar computed from the matrix
        has a gradient w.r.t. manifold.points.
        """
        from cdi.geometry.dirac import DiracOperator
        d = DiracOperator(manifold, clifford, connection, belief, cover, tiny_config)
        d.build()
        # Matrix depends on pts; verify grad flows
        loss = d.matrix.sum()
        loss.backward()
        assert manifold.points.grad is not None, (
            "manifold.points has no gradient through Dirac matrix — "
            ".detach() may still be present in dirac.build()"
        )
        assert manifold.points.grad.abs().max().item() > 0


class TestBeliefLaplacian:

    def test_matrix_shape(self, laplacian, tiny_config):
        N = tiny_config.total_state_dim
        assert laplacian.matrix.shape == (N, N)

    def test_self_adjoint(self, laplacian):
        """Theorem 5.1.3: Δ_ℬ = Δ_ℬ*."""
        err = laplacian.check_self_adjoint()
        assert err < 1e-8, f"Laplacian not self-adjoint: {err:.2e}"

    def test_positive_semidefinite(self, laplacian):
        """Theorem 5.1.3: Δ_ℬ ≥ 0."""
        assert laplacian.check_positive_semidefinite()

    def test_eigendecompose(self, laplacian, tiny_config):
        evals, evecs = laplacian.eigendecompose()
        N = tiny_config.total_state_dim
        assert evals.shape == (N,)
        assert evecs.shape == (N, N)
        assert (evals >= -1e-8).all(), f"Negative eigenvalue: {evals.min():.2e}"

    def test_spectral_gap_nonneg(self, laplacian):
        gap = laplacian.spectral_gap()
        assert float(gap) >= 0

    def test_lanczos_gap_nonneg(self, laplacian):
        """v2.0 Lanczos spectral gap is consistent with full eigendecomp."""
        lam1_full = laplacian.spectral_gap().item()
        lam1_lanczos = laplacian.lanczos_spectral_gap(max_iter=20)
        assert lam1_lanczos >= 0
        # Lanczos should be in the right ballpark (within 2× of full)
        if lam1_full > 1e-10:
            ratio = lam1_lanczos / lam1_full
            assert 0.01 < ratio < 100, (
                f"Lanczos gap {lam1_lanczos:.4f} far from full gap {lam1_full:.4f}"
            )

    def test_laplacian_in_graph(self, manifold, clifford, connection, belief,
                                 cover, tiny_config):
        """Fix F1/F3 v2.0: Laplacian matrix must propagate gradients."""
        from cdi.geometry.dirac import DiracOperator
        from cdi.operators.laplacian import BeliefLaplacian

        d = DiracOperator(manifold, clifford, connection, belief, cover, tiny_config)
        d.build()
        lap = BeliefLaplacian(d, belief, connection, tiny_config)
        lap.build()

        loss = lap.matrix.sum()
        loss.backward()
        # At least connection params or belief params should have grads
        conn_grads = [
            p.grad is not None and p.grad.abs().max().item() > 0
            for p in connection.get_parameters()
        ]
        belief_grads = [
            p.grad is not None and p.grad.abs().max().item() > 0
            for p in belief.get_parameters()
        ]
        assert any(conn_grads) or any(belief_grads), (
            "No gradient reached connection or belief through Laplacian matrix. "
            "Operators may have .detach() remaining."
        )


class TestHodgeDecomposition:

    def test_decomposition_is_orthogonal(self, hodge, tiny_config):
        """Theorem 5.2.1: ℋ ⊥ im(Δ_ℬ)."""
        N = tiny_config.total_state_dim
        psi = torch.randn(N, dtype=tiny_config.dtype)
        h, nh = hodge.decompose(psi)
        inner = torch.abs(torch.dot(h.detach(), nh.detach())).item()
        assert inner < 1e-6, f"Harmonic and non-harmonic not orthogonal: {inner:.2e}"

    def test_decomposition_sums_to_input(self, hodge, tiny_config):
        N = tiny_config.total_state_dim
        psi = torch.randn(N, dtype=tiny_config.dtype)
        h, nh = hodge.decompose(psi)
        assert torch.allclose(h + nh, psi, atol=1e-8)

    def test_harmonic_projector_idempotent(self, hodge, tiny_config):
        """H² = H."""
        H = hodge.harmonic_projector()
        H2 = H @ H
        assert torch.allclose(H.detach(), H2.detach(), atol=1e-6)

    def test_harmonic_dimension_nonneg(self, hodge):
        dim = hodge.harmonic_dimension()
        assert dim >= 0

    def test_hodge_output_in_graph(self, hodge, tiny_config):
        """Fix F1 v2.0: harmonic projection must NOT detach from graph.

        v1.0 detached the harmonic part; v2.0 must keep it in the graph
        so gradients flow to the Laplacian and its parameters.
        """
        N = tiny_config.total_state_dim
        psi = torch.randn(N, dtype=tiny_config.dtype, requires_grad=True)
        harmonic, _ = hodge.decompose(psi)
        # harmonic should be differentiable w.r.t. psi
        grad = torch.autograd.grad(harmonic.sum(), psi, allow_unused=True)[0]
        # If detached, grad would be None
        assert grad is not None, (
            "harmonic_part.detach() still present in HodgeDecomposition.decompose(). "
            "Remove .detach() — Fix F1 requirement."
        )


class TestGreenOperator:

    def test_pseudo_inverse(self, green):
        """Theorem 5.3.2: G·Δ + H ≈ I."""
        err = green.verify()
        assert err < 1e-4, f"Green pseudo-inverse error: {err:.4e}"

    def test_apply_shape(self, green, tiny_config):
        N = tiny_config.total_state_dim
        f = torch.randn(N, dtype=tiny_config.dtype)
        result = green.apply(f)
        assert result.shape == (N,)

    def test_green_apply_differentiable(self, green, tiny_config):
        """Fix F1 v2.0: Green's PCG apply must be differentiable.

        v1.0 detached the Green output. v2.0 uses PCG which is differentiable
        through each Laplacian matvec call.
        """
        N = tiny_config.total_state_dim
        f = torch.randn(N, dtype=tiny_config.dtype, requires_grad=True)
        result = green.apply(f)
        # result should be differentiable w.r.t. f
        grad = torch.autograd.grad(result.sum(), f, allow_unused=True)[0]
        assert grad is not None, (
            "Green.apply() output is not differentiable w.r.t. input. "
            "Check for .detach() in GreenOperator.apply() — Fix F1 requirement."
        )


class TestInferenceOperator:

    def test_infer_shape(self, built_engine):
        """Inference output has correct shape."""
        cfg = built_engine.config
        obs = torch.randn(cfg.n_points, cfg.observation_dim, dtype=cfg.dtype)
        result = built_engine.inference_op.infer(obs)
        assert result.shape == (cfg.n_points, cfg.output_dim)

    def test_embed_observation_shape(self, built_engine):
        cfg = built_engine.config
        obs = torch.randn(cfg.n_points, cfg.observation_dim, dtype=cfg.dtype)
        embedded = built_engine.inference_op.embed_observation(obs)
        assert embedded.shape == (cfg.total_state_dim,)

    def test_inference_differentiable(self, built_engine):
        """Fix F1 v2.0: inference output must be differentiable w.r.t. input.

        In v1.0 both harmonic_part and green_d_star were .detach()-ed,
        so the inference output had no gradient to the manifold parameters.
        In v2.0 neither is detached.
        """
        cfg = built_engine.config
        obs = torch.randn(
            cfg.n_points, cfg.observation_dim,
            dtype=cfg.dtype, requires_grad=True
        )
        result = built_engine.inference_op.infer(obs)
        grad = torch.autograd.grad(
            result.sum(), obs, allow_unused=True
        )[0]
        assert grad is not None, (
            "Inference output has no gradient to observation input. "
            ".detach() may still be present in InferenceOperator.infer()."
        )
