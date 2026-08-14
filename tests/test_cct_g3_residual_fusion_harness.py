from __future__ import annotations

from typing import Any

from benchmarks.cct_g3_4_token_residual import MATERIAL_GRU_TARGET_LOSS
from benchmarks.cct_g3_5_residual_fusion import G3_5_MODEL_NAMES, residual_fusion_decision


def _report(candidate_loss: float, control_loss: float, predecessor_loss: float, gru_loss: float) -> dict[str, Any]:
    records = []
    for seed in (11, 29, 47):
        for model, loss in (
            ("dcss_fused_residual_cdi", candidate_loss),
            ("dcss_fusion_control", control_loss),
            ("dcss_residual_cdi", predecessor_loss),
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
                "parameter_count": 80_586 if name.startswith("dcss_fused") or name.startswith("dcss_fusion") else 80_550 if name == "dcss_residual_cdi" else 80_120,
                "mean_validation_loss": candidate_loss if name == "dcss_fused_residual_cdi" else control_loss if name == "dcss_fusion_control" else predecessor_loss if name == "dcss_residual_cdi" else gru_loss if name == "gru_baseline" else 0.9,
            }
            for name in G3_5_MODEL_NAMES
        },
        "decision": {"finite_values_gate": True},
    }


def test_g3_fusion_material_gate_requires_declared_margin() -> None:
    candidate = MATERIAL_GRU_TARGET_LOSS - 0.01
    decision = residual_fusion_decision(_report(candidate, candidate + 0.05, candidate + 0.04, 6.80), parameter_tolerance=0.01)
    assert decision["fusion_value_gate"]
    assert decision["gru_per_seed_gate"]
    assert decision["material_gru_margin_gate"]
    assert decision["g3_5_mechanism_verdict"] == "EARNED_FUSION_EVIDENCE"
    assert decision["g3_5_quality_verdict"] == "MATERIAL_QUALITY_ADVANTAGE_EARNED"
    assert not decision["scale_authorized"]


def test_g3_fusion_can_be_partial_without_material_margin() -> None:
    decision = residual_fusion_decision(_report(6.79, 6.85, 6.83, 6.80), parameter_tolerance=0.01)
    assert decision["fusion_value_gate"]
    assert decision["gru_per_seed_gate"]
    assert not decision["material_gru_margin_gate"]
    assert decision["g3_5_quality_verdict"] == "QUALITY_RECOVERY_PARTIAL"


def test_g3_fusion_rejects_tie_with_exact_control() -> None:
    decision = residual_fusion_decision(_report(6.85, 6.85, 6.86, 6.80), parameter_tolerance=0.01)
    assert not decision["fusion_value_gate"]
    assert decision["g3_5_mechanism_verdict"] == "NO_FUSION_EVIDENCE"
    assert decision["g3_5_quality_verdict"] == "REDESIGN_BEFORE_SCALE"
