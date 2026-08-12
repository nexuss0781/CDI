"""Offline P1 production-training hardening primitives for DCSS-CDI."""
from .checkpoints import CHECKPOINT_FORMAT, build_envelope, file_sha256, load_verified, save_atomic, validate_envelope
from .config import ProductionRunConfig, ReleaseBoundary
from .data import DataManifest, GovernedDocument, P1DataPolicy, P2DataPolicy
from .evaluation import EvaluationCard, EvaluationEvidence, assert_core_optionality, evaluate_causal_offline, matched_baseline_summary
from typing import TYPE_CHECKING

from .lineage import ArtifactLineage, EnvironmentLineage, assert_compatible, fingerprint

if TYPE_CHECKING:
    from .inference import DCSSInferenceEngine, GenerationConfig, InferenceMetadata, interactive_chat

__all__ = [
    "CHECKPOINT_FORMAT",
    "ArtifactLineage",
    "DataManifest",
    "EnvironmentLineage",
    "EvaluationCard",
    "EvaluationEvidence",
    "GovernedDocument",
    "P1DataPolicy",
    "P2DataPolicy",
    "ProductionRunConfig",
    "ReleaseBoundary",
    "assert_compatible",
    "assert_core_optionality",
    "build_envelope",
    "evaluate_causal_offline",
    "file_sha256",
    "fingerprint",
    "load_verified",
    "matched_baseline_summary",
    "save_atomic",
    "validate_envelope",
    "DCSSInferenceEngine",
    "GenerationConfig",
    "InferenceMetadata",
    "interactive_chat",
]


_INFERENCE_EXPORTS = frozenset({"DCSSInferenceEngine", "GenerationConfig", "InferenceMetadata", "interactive_chat"})


def __getattr__(name: str):
    """Load the optional inference interface only when a caller requests it."""
    if name in _INFERENCE_EXPORTS:
        from . import inference

        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
