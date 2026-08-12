"""Typed immutable configuration for offline DCSS-CDI production-readiness runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ProductionRunConfig:
    """P1 configuration; it intentionally cannot enable training deployment or tools."""

    schema_version: str = "dcss-cdi-production-run-v1"
    phase: str = "P1"
    run_name: str = "offline-training-hardening"
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"
    model_family: str = "dcss_cdi"
    tokenizer_version: str = "stage-d-character-v1"
    training_mode: str = "offline_diagnostic"
    external_side_effects_enabled: bool = False
    capability_tools_enabled: bool = False
    allowed_data_classes: Tuple[str, ...] = ("synthetic", "rights_cleared_pilot")
    max_steps: int = 100
    checkpoint_interval: int = 25
    tags: Tuple[str, ...] = ("offline", "p1", "non-production")

    def validate(self) -> None:
        if self.phase != "P1":
            raise ValueError("This configuration is restricted to P1 training-system hardening.")
        if self.training_mode != "offline_diagnostic":
            raise ValueError("P1 permits offline diagnostic runs only.")
        if self.external_side_effects_enabled or self.capability_tools_enabled:
            raise ValueError("P1 never enables external side effects or capability tools.")
        if self.seed < 0 or self.max_steps <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("Seed, max_steps, and checkpoint_interval must be positive or zero as appropriate.")
        if self.checkpoint_interval > self.max_steps:
            raise ValueError("checkpoint_interval cannot exceed max_steps.")
        if self.device != "cpu" or self.dtype != "float32":
            raise ValueError("The P1 reference path is CPU float32 until a separately approved scale stage.")
        if self.model_family not in {"dcss_cdi", "transformer", "v2"}:
            raise ValueError("Unknown model family.")
        if not self.allowed_data_classes or "synthetic" not in self.allowed_data_classes:
            raise ValueError("P1 must retain synthetic-only compatibility coverage.")

    def as_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return sha256(json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReleaseBoundary:
    """Explicit non-production boundary attached to every P1 artifact."""

    status: str = "offline_research_only"
    real_corpus_training_authorized: bool = False
    fine_tuning_authorized: bool = False
    deployment_authorized: bool = False
    external_side_effects_enabled: bool = False

    def validate(self) -> None:
        if self.status != "offline_research_only" or any((self.real_corpus_training_authorized, self.fine_tuning_authorized, self.deployment_authorized, self.external_side_effects_enabled)):
            raise ValueError("P1 release boundary must remain offline research only.")

    def as_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)
