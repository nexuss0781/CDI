from __future__ import annotations

import json

import pytest
import torch

from cdi.v3.production.config import ProductionRunConfig
from cdi.v3.ssm import CohomodynamicState, StageCConfig, SelectiveCohomodynamicSSM


def test_dissipation_scale_and_edge_weight_bounds_fail_closed() -> None:
    model = SelectiveCohomodynamicSSM(StageCConfig.nano(seed=3))
    state = model.initial_state(batch_shape=(1,))
    x = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="dissipation_scale"):
        model.cell.step(x, state, dissipation_scale=-0.01)
    with torch.no_grad():
        model.cell.geometry.edge_log_weights.fill_(10.0)
    weights = model.cell.geometry.edge_weights
    assert torch.isfinite(weights).all()
    assert bool((weights > 0.0).all())
    assert bool((weights < model.cell.geometry.config.max_geometry_edge_weight).all())
    applied = model.cell.geometry.apply(state.fast)
    applied.square().sum().backward()
    assert model.cell.geometry.edge_log_weights.grad is not None
    assert torch.isfinite(model.cell.geometry.edge_log_weights.grad).all()


def test_bounded_geometry_weights_preserve_historical_effective_initialization() -> None:
    model = SelectiveCohomodynamicSSM(StageCConfig.nano(seed=7))
    historical_raw = torch.linspace(-0.2, 0.2, model.cell.topology.n_edges)
    expected = torch.nn.functional.softplus(historical_raw) + 1.0e-6
    assert torch.allclose(model.cell.geometry.edge_weights.detach().cpu(), expected, atol=1.0e-6, rtol=1.0e-6)


def test_state_norm_limit_is_enforced_at_runtime() -> None:
    config = StageCConfig(state_norm_bound=1.0, seed=5)
    model = SelectiveCohomodynamicSSM(config)
    state = CohomodynamicState(
        torch.full((1, 4, 4), 100.0),
        torch.full((1, 4, 4), 100.0),
        torch.full((1, 4, 4), 100.0),
    )
    with pytest.raises(FloatingPointError, match="state norm"):
        model.step(torch.zeros(1, 4), state)


def test_offline_config_is_ethiobbpe_bound_and_rejects_unknown_json_fields(tmp_path) -> None:
    config = ProductionRunConfig()
    config.validate()
    assert config.tokenizer_version == "EthioBBPE==2.0.0"
    with pytest.raises(ValueError, match="CPU float32"):
        ProductionRunConfig(device="cuda").validate()
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"seed": 7, "unexpected": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        ProductionRunConfig.from_json(path)
