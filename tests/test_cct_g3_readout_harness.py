from __future__ import annotations

from typing import Any

from benchmarks.cct_g3_2_readout_ablation import G3_2_MODEL_NAMES, readout_decision


def _report(full_loss: float, mean_control_loss: float, geometry_free_loss: float) -> dict[str, Any]:
    records = []
    for seed in (11, 29, 47):
        for model, loss in (
            ("dcss_cdi", full_loss),
            ("dcss_mean_readout_control", mean_control_loss),
            ("dcss_geometry_free", geometry_free_loss),
            ("gru_baseline", 0.9),
            ("transformer", 0.9),
        ):
            records.append(
                {
                    "seed": seed,
                    "model": model,
                    "train_loss_decreased": True,
                    "validation": {"loss": loss},
                }
            )
    return {
        "config": {"seeds": [11, 29, 47]},
        "records": records,
        "summary": {name: {"parameter_count": 80_500} for name in G3_2_MODEL_NAMES},
        "decision": {"finite_values_gate": True},
    }


def test_g3_readout_decision_requires_per_seed_full_improvement() -> None:
    decision = readout_decision(_report(0.80, 0.85, 0.84), parameter_tolerance=0.01)
    assert decision["parameter_match_gate"]
    assert decision["readout_value_gate"]
    assert decision["geometry_reconfirmation_gate"]
    assert decision["g3_2_verdict"] == "EARNED_READOUT_EVIDENCE"
    assert not decision["scale_authorized"]


def test_g3_readout_decision_rejects_null_readout_effect() -> None:
    decision = readout_decision(_report(0.85, 0.85, 0.86), parameter_tolerance=0.01)
    assert not decision["readout_value_gate"]
    assert decision["geometry_reconfirmation_gate"]
    assert decision["g3_2_verdict"] == "NO_READOUT_EVIDENCE"
