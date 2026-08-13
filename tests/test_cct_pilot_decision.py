from __future__ import annotations

from typing import Any

from benchmarks.ethiobbpe_synaxarium_pilot import PilotConfig, architecture_decision


def _record(model: str, seed: int, validation_loss: float, *, decreased: bool = True) -> dict[str, Any]:
    metric = {
        "loss": validation_loss,
        "perplexity": 10.0,
        "token_accuracy": 0.25,
        "token_count": 100.0,
    }
    return {
        "model": model,
        "seed": seed,
        "train_loss_first": 3.0,
        "train_loss_last": 2.0,
        "train_loss_decreased": decreased,
        "elapsed_seconds": 1.0,
        "tokens_processed": 100,
        "initial_validation": dict(metric),
        "validation": dict(metric),
        "test": dict(metric),
    }


def _summary(dcss: float, gru: float, transformer: float) -> dict[str, dict[str, float]]:
    return {
        "dcss_cdi": {"mean_validation_loss": dcss, "all_train_loss_decreased": 1.0},
        "gru_baseline": {"mean_validation_loss": gru, "all_train_loss_decreased": 1.0},
        "transformer": {"mean_validation_loss": transformer, "all_train_loss_decreased": 1.0},
    }


def _records(dcss: float, gru: float, transformer: float) -> list[dict[str, Any]]:
    return [
        _record(model, seed, loss)
        for seed in (11, 29, 47)
        for model, loss in (("dcss_cdi", dcss), ("gru_baseline", gru), ("transformer", transformer))
    ]


def test_cct_decision_passes_only_when_every_gate_passes() -> None:
    decision = architecture_decision(
        _summary(dcss=1.00, gru=1.01, transformer=0.98),
        PilotConfig(relative_loss_tolerance=0.05),
        _records(dcss=1.00, gru=1.01, transformer=0.98),
    )
    assert decision["verdict"] == "EARNED_NEXT_PILOT"
    assert decision["finite_values_gate"]
    assert decision["learning_gate"]
    assert decision["transformer_tolerance_gate"]
    assert decision["gru_per_seed_gate"]


def test_cct_decision_rejects_transformer_tolerant_but_gru_losing_run() -> None:
    decision = architecture_decision(
        _summary(dcss=1.00, gru=0.99, transformer=0.98),
        PilotConfig(relative_loss_tolerance=0.05),
        _records(dcss=1.00, gru=0.99, transformer=0.98),
    )
    assert decision["transformer_tolerance_gate"]
    assert not decision["gru_per_seed_gate"]
    assert decision["verdict"] == "REDESIGN_BEFORE_SCALE"


def test_cct_decision_rejects_nonfinite_or_incomplete_seed_evidence() -> None:
    records = _records(dcss=1.00, gru=1.01, transformer=0.98)
    records[0]["validation"]["loss"] = float("nan")
    decision = architecture_decision(
        _summary(dcss=1.00, gru=1.01, transformer=0.98),
        PilotConfig(relative_loss_tolerance=0.05),
        records,
    )
    assert not decision["finite_values_gate"]
    assert decision["verdict"] == "REDESIGN_BEFORE_SCALE"
