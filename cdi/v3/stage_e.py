"""Controlled Stage E model matrix construction and audit metadata."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal

from torch import nn

from .ablations import apply_stage_e_ablation
from .language_model import DCSSLanguageModel
from .ssm import StageCConfig
from .tokenizer import CharacterTokenizer
from .training import StageDConfig, build_model


MatrixID = Literal["T", "V2", "U", "G", "H", "E", "C", "F"]


@dataclass(frozen=True)
class MatrixDefinition:
    identifier: MatrixID
    model_key: str
    description: str
    named_difference: str
    allocation_claim_scope: str


MATRIX: Dict[str, MatrixDefinition] = {
    "T": MatrixDefinition("T", "transformer", "Small causal Transformer baseline", "Conventional attention baseline", "not_applicable"),
    "V2": MatrixDefinition("V2", "v2", "Compact legacy-v2 style adapter", "Historical-style recurrent adapter", "not_applicable"),
    "U": MatrixDefinition("U", "dcss_cdi", "Ungated sparse CDI recurrence", "Fixed controls replace content-selective gates", "full_sparse"),
    "G": MatrixDefinition("G", "dcss_cdi", "Geometry-free selective recurrence", "Matrix-free geometric field disabled", "full_sparse"),
    "H": MatrixDefinition("H", "dcss_cdi", "No harmonic memory", "Harmonic/slow band disabled and frozen", "full_sparse"),
    "E": MatrixDefinition("E", "dcss_cdi", "Explicit Euler recurrence", "Exact pairwise exponential replaced by Euler", "full_sparse"),
    "C": MatrixDefinition("C", "dcss_cdi", "Unconstrained cochain diagnostic", "Learned unconstrained vertex mixing added", "named_dense_vertex_exception"),
    "F": MatrixDefinition("F", "dcss_cdi", "Full DCSS-CDI", "Frozen Stage D selective geometric exact three-band engine", "full_sparse"),
}


def build_matrix_model(identifier: MatrixID | str, tokenizer: CharacterTokenizer, training_config: StageDConfig) -> nn.Module:
    """Build a single Stage E matrix member from frozen Stage D hyperparameters."""
    identifier = str(identifier)
    if identifier not in MATRIX:
        raise ValueError(f"Unknown Stage E matrix identifier: {identifier}")
    definition = MATRIX[identifier]
    if identifier == "T":
        return build_model("transformer", tokenizer, training_config)
    if identifier == "V2":
        return build_model("v2", tokenizer, training_config)
    geometry_ablation = identifier == "G"
    ssm_config = StageCConfig.nano(seed=training_config.seed, geometry_ablation=geometry_ablation)
    model = DCSSLanguageModel(tokenizer, ssm_config)
    apply_stage_e_ablation(model.ssm, identifier)  # type: ignore[arg-type]
    if identifier == "G":
        for parameter in model.ssm.cell.geometry.parameters():
            parameter.requires_grad_(False)
    if identifier == "H":
        for parameter in model.ssm.cell.bands["harmonic"].parameters():
            parameter.requires_grad_(False)
    return model


def matrix_manifest(identifier: MatrixID | str, tokenizer: CharacterTokenizer, training_config: StageDConfig) -> Dict[str, Any]:
    definition = MATRIX[str(identifier)]
    return {
        "format": "dcss-cdi-stage-e-model-manifest-v1",
        "definition": asdict(definition),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "training_config": training_config.as_dict(),
        "matching_contract": {
            "tokenizer": "shared",
            "data_order": "shared deterministic document chunks",
            "token_budget": "shared fixed optimizer-step budget",
            "optimizer": "shared AdamW",
            "precision": "shared CPU float32",
            "chunk_length": "shared",
            "batch_size": "shared",
        },
    }
