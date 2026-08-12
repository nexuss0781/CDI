"""Typed immutable configuration for offline DCSS-CDI production-readiness runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ProductionRunConfig:
    """Production configuration authorizing real-data GPU training."""

    schema_version: str = "dcss-cdi-production-run-v1"
    phase: str = "Production"
    run_name: str = "gpu-large-corpus-training"
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"
    model_family: str = "dcss_cdi"
    tokenizer_version: str = "stage-d-character-v1"
    training_mode: str = "rights_cleared_large_corpus"
    external_side_effects_enabled: bool = False
    capability_tools_enabled: bool = False
    allowed_data_classes: Tuple[str, ...] = ("synthetic", "rights_cleared_pilot")
    max_steps: int = 200
    checkpoint_interval: int = 50
    tags: Tuple[str, ...] = ("gpu", "production", "large-corpus")

    def validate(self) -> None:
        if self.phase not in ("P1", "Production"):
            raise ValueError("Unsupported production phase.")
        if self.external_side_effects_enabled or self.capability_tools_enabled:
            raise ValueError("External side effects and capability tools remain disabled.")
        if self.seed < 0 or self.max_steps <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("Seed, max_steps, and checkpoint_interval must be positive.")
        if self.checkpoint_interval > self.max_steps:
            raise ValueError("checkpoint_interval cannot exceed max_steps.")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Device must be cpu or cuda.")
        if self.model_family not in {"dcss_cdi", "transformer", "v2"}:
            raise ValueError("Unknown model family.")

    def as_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return sha256(json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReleaseBoundary:
    """Explicit release boundary for production-scale artifacts."""

    status: str = "offline_research_only"
    real_corpus_training_authorized: bool = False
    fine_tuning_authorized: bool = False
    deployment_authorized: bool = False
    external_side_effects_enabled: bool = False

    def validate(self) -> None:
        if self.external_side_effects_enabled:
            raise ValueError("External side effects are never authorized in this phase.")
        if self.status == "offline_research_only" and any((self.real_corpus_training_authorized, self.fine_tuning_authorized, self.deployment_authorized)):
            raise ValueError("Offline research status prohibits real-data training or deployment.")

    def as_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)
