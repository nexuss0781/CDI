"""Verification tests for the DCSS-CDI Stage B sparse substrate."""
from __future__ import annotations

import pytest
import torch

from benchmarks.stage_b import (
    device_gate,
    diagnostics_gate,
    equivalence_gate,
    gradient_gate,
    production_guard_gate,
    serialization_gate,
    sparse_scaling_gate,
    topology_gate,
)
from cdi.config import CDIConfig
from cdi.v3 import (
    DCSSConfig,
    DenseReferenceOperators,
    MatrixFreeLaplacian,
    OperatorDiagnostics,
    SparseCochainMap,
    SparseIncidence,
    SparseTopology,
)


def build_nano(ablation: bool = False):
    config = DCSSConfig.nano(seed=42, geometry_ablation=ablation)
    topology = SparseTopology.from_config(config)
    return config, topology, MatrixFreeLaplacian(topology, config)


def test_nano_configuration_and_deterministic_topology() -> None:
    config, topology, _ = build_nano()
    assert config.total_state_dim < 64
    result = topology_gate("nano", 42)
    assert result["passed"], result
    assert result["details"]["fingerprint"] == topology.fingerprint()


def test_incidence_adjoint_for_leading_batch_dimensions() -> None:
    config, topology, _ = build_nano()
    incidence = SparseIncidence(topology)
    torch.manual_seed(43)
    x = torch.randn(2, topology.n_vertices, config.state_width)
    y = torch.randn(2, topology.n_edges, config.state_width)
    left = (incidence.apply(x) * y).sum()
    right = (x * incidence.transpose_apply(y)).sum()
    assert torch.allclose(left, right, rtol=1e-5, atol=1e-6)


def test_sparse_dense_equivalence_random_basis_batch_and_ablation() -> None:
    result = equivalence_gate("nano", trials=6, seed=42)
    assert result["passed"], result
    assert result["details"]["modes"]["false"]["basis_trials"] == 32
    assert result["details"]["modes"]["true"]["passed"]


def test_gradient_equivalence_and_geometry_ablation_activity() -> None:
    result = gradient_gate("nano", seed=42)
    assert result["passed"], result
    assert result["details"]["modes"]["false"]["sparse_gradient_norm"] > 0.0
    assert result["details"]["modes"]["true"]["sparse_gradient"] is None


def test_structural_cochain_identity_and_health_score() -> None:
    _, topology, laplacian = build_nano()
    diagnostics = OperatorDiagnostics(laplacian, SparseCochainMap(topology))
    report = diagnostics.full_report()
    assert report["cochain"]["passed"], report
    assert report["cochain"]["max_relative_residual"] <= 1e-5
    assert 0.0 <= report["cohomological_health_score"]["score"] <= 1.0
    result = diagnostics_gate("nano", 42)
    assert result["passed"], result


def test_dense_reference_hard_limit() -> None:
    _, _, laplacian = build_nano()
    with pytest.raises(ValueError, match="refused"):
        DenseReferenceOperators.build_small(laplacian, max_state_dim=1)


def test_production_guard_rejects_dense_behavior_and_exercises_both_modes() -> None:
    result = production_guard_gate("nano", 42)
    assert result["passed"], result
    for mode in ("false", "true"):
        details = result["details"]["modes"][mode]
        assert not details["large_allocations"]
        assert not details["forbidden_operations"]


def test_cpu_device_serialization_and_sparse_scaling() -> None:
    assert device_gate("nano", 42)["passed"]
    assert serialization_gate("nano", 42)["passed"]
    scaling = sparse_scaling_gate([4, 8, 16], 42)
    assert scaling["passed"], scaling
    assert all(record["storage_to_dense_ratio"] < 1.0 for record in scaling["details"]["records"])


def test_legacy_config_exposes_geometry_ablation_without_default_change() -> None:
    config = CDIConfig()
    assert config.geometry_ablation is False
