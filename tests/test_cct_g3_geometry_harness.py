from __future__ import annotations

from typing import Any

from benchmarks.cct_g3_1_geometry_ablation import G3_MODEL_NAMES, geometry_decision


def _report(full_loss: float, geometry_free_loss: float) -> dict[str, Any]:
    records = []
    for seed in (11, 29, 47):
        for model, loss in (
            ("dcss_cdi", full_loss),
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
        "summary": {name: {"parameter_count": 80_500} for name in G3_MODEL_NAMES},
        "decision": {"finite_values_gate": True},
    }


def test_g3_geometry_decision_requires_per_seed_full_improvement() -> None:
    decision = geometry_decision(_report(full_loss=0.8, geometry_free_loss=0.85), parameter_tolerance=0.01)
    assert decision["parameter_match_gate"]
    assert decision["geometry_value_gate"]
    assert decision["g3_verdict"] == "EARNED_GEOMETRY_EVIDENCE"
    assert not decision["scale_authorized"]


def test_g3_geometry_decision_rejects_null_or_negative_mechanism_effect() -> None:
    decision = geometry_decision(_report(full_loss=0.85, geometry_free_loss=0.85), parameter_tolerance=0.01)
    assert not decision["geometry_value_gate"]
    assert decision["g3_verdict"] == "NO_GEOMETRY_EVIDENCE"
