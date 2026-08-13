"""Active CCT v3 DCSS-CDI implementation.

This namespace contains the EthioBBPE tokenizer adapter, sparse matrix-free
operator substrate, selective recurrent state-space model, causal language
model, matched baselines, training utilities, and verified checkpoint
interfaces. Optional legacy and experimental objects remain explicitly bounded
by their own modules and are not active CCT evidence paths.
"""
from .cochain import SparseCochainMap
from .config import DCSSConfig, load_config
from .diagnostics import OperatorDiagnostics
from .incidence import SparseIncidence
from .laplacian import MatrixFreeLaplacian
from .reference import DenseReferenceOperators
from .topology import SparseTopology
from .tokenizer import CDITokenizer, CharacterTokenizer, EncodedText, EthioBBPETokenizer, TokenizerConfig
from .language_model import DCSSLanguageModel, LegacyCDIV2Adapter, LossReport, TinyTransformerBaseline
from .ablations import ExplicitEulerIntegrator, UngatedSelectiveGate, apply_stage_e_ablation
from .stage_e import MATRIX, MatrixDefinition, build_matrix_model, matrix_manifest
from .capabilities import AuditTrail, CapabilityOrchestrator, EpisodicMemory, Executor, MemoryRecord, Plan, PlanAction, Planner, Retriever, ToolDefinition, ToolRegistry, Verifier
from .training import LocalSyntheticCorpus, StageDConfig, build_model, checkpoint_payload, evaluate, optimizer_for, parameter_fingerprint, restore_checkpoint, train_steps
from typing import TYPE_CHECKING

from .production import ArtifactLineage, DataManifest, EnvironmentLineage, EvaluationCard, EvaluationEvidence, GovernedDocument, P1DataPolicy, P2DataPolicy, ProductionRunConfig, ReleaseBoundary, assert_core_optionality, build_envelope, evaluate_causal_offline, load_verified, save_atomic

if TYPE_CHECKING:
    from .production.inference import DCSSInferenceEngine, GenerationConfig, InferenceMetadata, interactive_chat
from .ssm import (
    BAND_NAMES,
    CayleyIntegrator,
    CohomodynamicCell,
    CohomodynamicState,
    DynamicsDiagnostics,
    MemoryBand,
    SelectiveCohomodynamicSSM,
    SelectiveGate,
    StageCConfig,
    StableGenerator,
    StateCodec,
)

__all__ = [
    "DCSSConfig",
    "DenseReferenceOperators",
    "MatrixFreeLaplacian",
    "OperatorDiagnostics",
    "SparseCochainMap",
    "SparseIncidence",
    "SparseTopology",
    "load_config",
    "BAND_NAMES",
    "CayleyIntegrator",
    "CohomodynamicCell",
    "CohomodynamicState",
    "DynamicsDiagnostics",
    "MemoryBand",
    "SelectiveCohomodynamicSSM",
    "SelectiveGate",
    "StageCConfig",
    "StableGenerator",
    "StateCodec",
    "CDITokenizer",
    "EthioBBPETokenizer",
    "CharacterTokenizer",
    "EncodedText",
    "TokenizerConfig",
    "DCSSLanguageModel",
    "LegacyCDIV2Adapter",
    "LossReport",
    "TinyTransformerBaseline",
    "ExplicitEulerIntegrator",
    "UngatedSelectiveGate",
    "apply_stage_e_ablation",
    "MATRIX",
    "MatrixDefinition",
    "build_matrix_model",
    "matrix_manifest",
    "AuditTrail",
    "CapabilityOrchestrator",
    "EpisodicMemory",
    "Executor",
    "MemoryRecord",
    "Plan",
    "PlanAction",
    "Planner",
    "Retriever",
    "ToolDefinition",
    "ToolRegistry",
    "Verifier",
    "LocalSyntheticCorpus",
    "StageDConfig",
    "build_model",
    "checkpoint_payload",
    "evaluate",
    "optimizer_for",
    "parameter_fingerprint",
    "restore_checkpoint",
    "train_steps",
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
    "assert_core_optionality",
    "build_envelope",
    "evaluate_causal_offline",
    "load_verified",
    "save_atomic",
    "DCSSInferenceEngine",
    "GenerationConfig",
    "InferenceMetadata",
    "interactive_chat",
]


_INFERENCE_EXPORTS = frozenset({"DCSSInferenceEngine", "GenerationConfig", "InferenceMetadata", "interactive_chat"})


def __getattr__(name: str):
    """Resolve inference exports lazily to keep module execution side-effect free."""
    if name in _INFERENCE_EXPORTS:
        from .production import inference

        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
