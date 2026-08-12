"""DCSS-CDI v3 sparse, factorized operator substrate.

Stage B contains no selective recurrence or language-model training path. Its
public objects are CPU-safe matrix-free building blocks for later approved work.
"""
from .cochain import SparseCochainMap
from .config import DCSSConfig, load_config
from .diagnostics import OperatorDiagnostics
from .incidence import SparseIncidence
from .laplacian import MatrixFreeLaplacian
from .reference import DenseReferenceOperators
from .topology import SparseTopology
from .tokenizer import CDITokenizer, CharacterTokenizer, EncodedText, TokenizerConfig
from .language_model import DCSSLanguageModel, LegacyCDIV2Adapter, LossReport, TinyTransformerBaseline
from .ablations import ExplicitEulerIntegrator, UngatedSelectiveGate, apply_stage_e_ablation
from .stage_e import MATRIX, MatrixDefinition, build_matrix_model, matrix_manifest
from .capabilities import AuditTrail, CapabilityOrchestrator, EpisodicMemory, Executor, MemoryRecord, Plan, PlanAction, Planner, Retriever, ToolDefinition, ToolRegistry, Verifier
from .training import LocalSyntheticCorpus, StageDConfig, build_model, checkpoint_payload, evaluate, optimizer_for, restore_checkpoint, train_steps
from .production import ArtifactLineage, DataManifest, EvaluationCard, EvaluationEvidence, GovernedDocument, P1DataPolicy, P2DataPolicy, ProductionRunConfig, ReleaseBoundary, assert_core_optionality, build_envelope, load_verified, save_atomic
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
    "restore_checkpoint",
    "train_steps",
    "ArtifactLineage",
    "DataManifest",
    "EvaluationCard",
    "EvaluationEvidence",
    "GovernedDocument",
    "P1DataPolicy",
    "P2DataPolicy",
    "ProductionRunConfig",
    "ReleaseBoundary",
    "assert_core_optionality",
    "build_envelope",
    "load_verified",
    "save_atomic",
]
