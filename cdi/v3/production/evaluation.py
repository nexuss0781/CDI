"""Offline evaluation and evidence structures for P1 training-system hardening."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Dict, Mapping, Sequence

import torch
from torch import nn

from ..training import evaluate
from .config import ReleaseBoundary
from .lineage import ArtifactLineage


@dataclass(frozen=True)
class EvaluationCard:
    name: str
    intended_use: str
    split_manifest_fingerprint: str
    metrics: tuple[str, ...]
    evaluator_version: str = "dcss-cdi-p1-evaluator-v1"
    private_holdout: bool = False

    def validate(self) -> None:
        if not self.name or not self.intended_use or not self.split_manifest_fingerprint:
            raise ValueError("Evaluation cards require name, intended use, and data-manifest fingerprint.")
        if not self.metrics:
            raise ValueError("Evaluation cards require at least one metric.")
        if self.private_holdout:
            raise ValueError("P1 does not claim access to private real-data holdouts.")

    def as_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return sha256(json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvaluationEvidence:
    card_fingerprint: str
    lineage_fingerprint: str
    metrics: Dict[str, float]
    status: str
    release_boundary: Dict[str, Any]
    evaluator_version: str = "dcss-cdi-p1-evaluator-v1"

    def validate(self) -> None:
        if self.status not in {"PASS", "FAIL", "OBSERVATION"}:
            raise ValueError("Evaluation status must be PASS, FAIL, or OBSERVATION.")
        if not self.card_fingerprint or not self.lineage_fingerprint:
            raise ValueError("Evaluation evidence must be linked to card and artifact lineage.")
        ReleaseBoundary(**self.release_boundary).validate()
        if any(not isinstance(value, float) or not torch.isfinite(torch.tensor(value)) for value in self.metrics.values()):
            raise ValueError("Evaluation metrics must be finite floats.")

    def as_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


def evaluate_causal_offline(model: nn.Module, batches: Sequence[Mapping[str, Any]], card: EvaluationCard, lineage: ArtifactLineage, boundary: ReleaseBoundary | None = None) -> EvaluationEvidence:
    """Run only the existing local causal evaluator and bind results to P1 lineage."""
    card.validate()
    lineage.validate()
    boundary = boundary or ReleaseBoundary()
    boundary.validate()
    raw_metrics = evaluate(model, batches)
    metrics = {key: float(value) for key, value in raw_metrics.items()}
    evidence = EvaluationEvidence(card.fingerprint, lineage.fingerprint, metrics, "OBSERVATION", boundary.as_dict())
    evidence.validate()
    return evidence


def matched_baseline_summary(records: Mapping[str, EvaluationEvidence]) -> Dict[str, Any]:
    """Return a comparison only if every record used the same evaluation card."""
    if len(records) < 2:
        raise ValueError("A baseline summary requires at least two model records.")
    cards = {record.card_fingerprint for record in records.values()}
    if len(cards) != 1:
        raise ValueError("Matched baseline results must share an evaluation card.")
    ordered = {name: record.metrics for name, record in sorted(records.items())}
    return {"format": "dcss-cdi-p1-matched-baseline-v1", "evaluation_card_fingerprint": next(iter(cards)), "records": ordered, "comparison_is_claim_free": True}


def max_tensor_error(reference: Sequence[torch.Tensor], candidate: Sequence[torch.Tensor]) -> float:
    if len(reference) != len(candidate):
        raise ValueError("Tensor sequences must have matching lengths.")
    if not reference:
        return 0.0
    return max(float((left.detach().cpu() - right.detach().cpu()).abs().max()) for left, right in zip(reference, candidate))


def assert_core_optionality(reference: Sequence[torch.Tensor], candidate: Sequence[torch.Tensor], atol: float = 1e-6) -> float:
    error = max_tensor_error(reference, candidate)
    if error > atol:
        raise AssertionError(f"Core optionality violated: max absolute difference {error} exceeds {atol}.")
    return error
