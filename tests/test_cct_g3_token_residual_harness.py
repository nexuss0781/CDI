from __future__ import annotations

from typing import Any

from benchmarks.cct_g3_4_token_residual import (
    G3_4_MODEL_NAMES,
    MATERIAL_GRU_TARGET_LOSS,
    token_residual_decision,
)


def _report(candidate_loss: float, control_loss: float, predecessor_loss: float, gru_loss: float) -> dict[str, Any]:
    records = []
    for seed in (11, 29, 47):
        for model, loss in (
            ("dcss_residual_cdi", candidate_loss),
            ("dcss_residual_control", control_loss),
            ("dcss_cdi", predecessor_loss),
            ("gru_baseline", gru_loss),
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
        "summary": {
            name: {
                "parameter_count": 80_550 if name.startswith("dcss_residual") else 80_510 if name == "dcss_cdi" else 80_120,
                "mean_validation_loss": candidate_loss if name == "dcss_residual_cdi" else control_loss if name == "dcss_residual_control" else predecessor_loss if name == "dcss_cdi" else gru_loss if name == "gru_baseline" else 0.9,
            }
            for name in G3_4_MODEL_NAMES
        },
        "decision": {"finite_values_gate": True},
    }


def test_g3_token_residual_material_quality_gate_requires_declared_margin() -> None:
    candidate = MATERIAL_GRU_TARGET_LOSS - 0.01
    decision = token_residual_decision(_report(candidate, candidate + 0.05, candidate + 0.04, 6.80), parameter_tolerance=0.01)
    assert decision["token_residual_value_gate"]
    assert decision["gru_per_seed_gate"]
    assert decision["material_gru_margin_gate"]
    assert decision["g3_4_mechanism_verdict"] == "EARNED_TOKEN_RESIDUAL_EVIDENCE"
    assert decision["g3_4_quality_verdict"] == "MATERIAL_QUALITY_ADVANTAGE_EARNED"
    assert not decision["scale_authorized"]


def test_g3_token_residual_quality_recovery_can_be_partial_without_margin() -> None:
    candidate = 6.79
    decision = token_residual_decision(_report(candidate, 6.85, 6.83, 6.80), parameter_tolerance=0.01)
    assert decision["token_residual_value_gate"]
    assert decision["gru_per_seed_gate"]
    assert not decision["material_gru_margin_gate"]
    assert decision["g3_4_quality_verdict"] == "QUALITY_RECOVERY_PARTIAL"


def test_g3_token_residual_rejects_candidate_that_does_not_beat_exact_control() -> None:
    decision = token_residual_decision(_report(6.85, 6.85, 6.86, 6.80), parameter_tolerance=0.01)
    assert not decision["token_residual_value_gate"]
    assert decision["g3_4_mechanism_verdict"] == "NO_TOKEN_RESIDUAL_EVIDENCE"
    assert decision["g3_4_quality_verdict"] == "REDESIGN_BEFORE_SCALE"
