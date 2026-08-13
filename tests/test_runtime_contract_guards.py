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
    with pytest.raises(FloatingPointError, match="edge weight"):
        model.cell.geometry.apply(state.fast)


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
