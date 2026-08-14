from __future__ import annotations

from typing import Any

from benchmarks.cct_g3_3_harmonic_ablation import G3_3_MODEL_NAMES, harmonic_decision


def _diagnostics(model: str) -> dict[str, Any] | None:
    if not model.startswith("dcss_"):
        return None
    harmonic = 0.0 if model == "dcss_harmonic_disabled" else 0.2
    harmonic_gradient = 0.0 if model == "dcss_harmonic_disabled" else 0.3
    return {
        "applicable": True,
        "state_norm_trace": [{"step": 0, "fast": 0.1, "middle": 0.1, "harmonic": harmonic, "total": 0.3}],
        "final_band_energy": {"fast": 0.1, "middle": 0.1, "harmonic": harmonic, "total": 0.3},
        "gradient_l2_by_group": {
            "fast": 0.1,
            "middle": 0.1,
            "harmonic": harmonic_gradient,
            "geometry": 0.1,
            "readout": 0.1,
            "embedding_output": 0.1,
        },
    }


def _report(full_loss: float, harmonic_disabled_loss: float, geometry_free_loss: float) -> dict[str, Any]:
    records = []
    for seed in (11, 29, 47):
        for model, loss in (
            ("dcss_cdi", full_loss),
            ("dcss_harmonic_disabled", harmonic_disabled_loss),
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
                    "post_training_diagnostics": _diagnostics(model),
                }
            )
    return {
        "config": {"seeds": [11, 29, 47]},
        "records": records,
        "summary": {name: {"parameter_count": 80_500} for name in G3_3_MODEL_NAMES},
        "decision": {"finite_values_gate": True},
    }


def test_g3_harmonic_decision_requires_per_seed_full_improvement() -> None:
    decision = harmonic_decision(_report(0.80, 0.85, 0.84), parameter_tolerance=0.01)
    assert decision["parameter_match_gate"]
    assert decision["state_gradient_diagnostics_gate"]
    assert decision["harmonic_value_gate"]
    assert decision["geometry_reconfirmation_gate"]
    assert decision["g3_3_verdict"] == "EARNED_HARMONIC_EVIDENCE"
    assert not decision["scale_authorized"]


def test_g3_harmonic_decision_identifies_consistent_negative_effect() -> None:
    decision = harmonic_decision(_report(0.85, 0.80, 0.86), parameter_tolerance=0.01)
    assert decision["harmonic_negative_gate"]
    assert decision["g3_3_verdict"] == "HARMONIC_NEGATIVE_EVIDENCE"


def test_g3_harmonic_decision_rejects_null_effect() -> None:
    decision = harmonic_decision(_report(0.85, 0.85, 0.86), parameter_tolerance=0.01)
    assert not decision["harmonic_value_gate"]
    assert not decision["harmonic_negative_gate"]
    assert decision["g3_3_verdict"] == "NO_HARMONIC_EVIDENCE"
