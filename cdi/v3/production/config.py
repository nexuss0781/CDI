"""Strict offline configuration for verified CCT support artifacts.

External production training is blocked. This schema exists only for local
checkpoint lineage and offline-hardening tests; it cannot authorize ingestion,
real-corpus training, deployment, GPU execution, or capability tools.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ProductionRunConfig:
    """Versioned, fail-closed offline lineage configuration."""

    schema_version: str = "dcss-cdi-offline-run-v2"
    phase: str = "P1"
    run_name: str = "offline-training-hardening"
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"
    model_family: str = "dcss_cdi"
    tokenizer_version: str = "EthioBBPE==2.0.0"
    training_mode: str = "offline_governed_synthetic"
    external_side_effects_enabled: bool = False
    capability_tools_enabled: bool = False
    allowed_data_classes: Tuple[str, ...] = ("synthetic", "rights_cleared_pilot")
    max_steps: int = 200
    checkpoint_interval: int = 50
    tags: Tuple[str, ...] = ("offline", "cct", "governed")

    @staticmethod
    def from_json(path: str | Path) -> "ProductionRunConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Offline run configuration must be a JSON object.")
        allowed = {item.name for item in fields(ProductionRunConfig)}
        unknown = set(data).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown offline run configuration fields: {sorted(unknown)}")
        if "allowed_data_classes" in data:
            data["allowed_data_classes"] = tuple(data["allowed_data_classes"])
        if "tags" in data:
            data["tags"] = tuple(data["tags"])
        return ProductionRunConfig(**data)

    def validate(self) -> None:
        if self.schema_version != "dcss-cdi-offline-run-v2" or self.phase != "P1":
            raise ValueError("Only the P1 offline CCT compatibility schema is supported.")
        if self.external_side_effects_enabled or self.capability_tools_enabled:
            raise ValueError("External side effects and capability tools remain disabled.")
        if self.seed < 0 or self.max_steps <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("Seed, max_steps, and checkpoint_interval must be positive.")
        if self.checkpoint_interval > self.max_steps:
            raise ValueError("checkpoint_interval cannot exceed max_steps.")
        if self.device != "cpu" or self.dtype != "float32":
            raise ValueError("Offline hardening is CPU float32 only.")
        if self.model_family not in {"dcss_cdi", "transformer", "gru_baseline"}:
            raise ValueError("Unknown offline CCT model family.")
        if self.tokenizer_version != "EthioBBPE==2.0.0":
            raise ValueError("Offline lineage requires EthioBBPE==2.0.0.")
        if self.training_mode != "offline_governed_synthetic":
            raise ValueError("This schema cannot authorize external or real-corpus training.")
        if not self.allowed_data_classes or not set(self.allowed_data_classes).issubset({"synthetic", "rights_cleared_pilot"}):
            raise ValueError("Offline lineage permits only synthetic or rights-cleared pilot data classes.")

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
