from __future__ import annotations

from typing import Any

from benchmarks.cct_g3_6_bounded_quality import REFERENCE_VALIDATION_LOSS, continuation_decision


def _report(candidate: float, gru: float, *, learning: bool = True) -> dict[str, Any]:
    records = []
    for seed in (11, 29, 47):
        for model, loss in (
            ("dcss_residual_cdi", candidate),
            ("gru_baseline", gru),
            ("transformer", 7.0),
        ):
            records.append(
                {
                    "seed": seed,
                    "model": model,
                    "train_loss_decreased": learning,
                    "validation": {"loss": loss},
                }
            )
    return {
        "records": records,
        "summary": {
            "dcss_residual_cdi": {"mean_validation_loss": candidate},
            "gru_baseline": {"mean_validation_loss": gru},
            "transformer": {"mean_validation_loss": 7.0},
        },
        "decision": {
            "finite_values_gate": True,
            "learning_gate": learning,
            "gru_per_seed_gate": candidate <= gru,
        },
    }


def test_g36_continuation_requires_progress_and_gru_relation() -> None:
    decision = continuation_decision(_report(6.70, 6.80))
    assert decision["progress_gate"]
    assert decision["gru_per_seed_gate"]
    assert decision["verdict"] == "EARNED_BOUNDED_CONTINUATION"
    assert not decision["material_target_gate"]
    assert not decision["scale_authorized"]


def test_g36_continuation_reports_material_target_separately() -> None:
    decision = continuation_decision(_report(6.60, 6.80))
    assert decision["progress_gate"]
    assert decision["material_target_gate"]
    assert decision["verdict"] == "EARNED_BOUNDED_CONTINUATION"
    assert not decision["scale_authorized"]


def test_g36_continuation_rejects_no_progress_or_gru_loss() -> None:
    no_progress = continuation_decision(_report(6.80, 6.90))
    no_gru = continuation_decision(_report(6.70, 6.60))
    assert no_progress["verdict"] == "NO_CONTINUATION_EVIDENCE"
    assert no_gru["verdict"] == "NO_CONTINUATION_EVIDENCE"
    assert REFERENCE_VALIDATION_LOSS == 6.743546
