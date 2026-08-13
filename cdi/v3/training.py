"""Deterministic CPU-first Stage D data, training, and checkpoint utilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .language_model import DCSSLanguageModel, LegacyCDIV2Adapter, TinyTransformerBaseline
from .ssm import StageCConfig
from .tokenizer import EncodedText, EthioBBPETokenizer, TokenizerConfig


@dataclass(frozen=True)
class StageDConfig:
    """Small reproducible CPU training configuration for all Stage D models."""

    name: str = "nano"
    seed: int = 42
    dtype_str: str = "float32"
    device: str = "cpu"
    chunk_length: int = 8
    batch_size: int = 4
    learning_rate: float = 0.05
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    gradient_accumulation: int = 1
    warmup_steps: int = 0
    overfit_steps: int = 100
    resume_steps: int = 50
    resume_interrupt_at: int = 25

    @property
    def dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype_str)

    def validate(self) -> None:
        if self.name != "nano":
            raise ValueError("Stage D currently exposes only the CPU-safe nano tier.")
        if self.dtype != torch.float32 or self.device != "cpu":
            raise ValueError("The reproducible Stage D reference path is CPU float32 only.")
        if self.chunk_length < 2 or self.batch_size <= 0:
            raise ValueError("chunk_length must be at least two and batch_size positive.")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("Training hyperparameters must be positive.")
        if self.resume_interrupt_at <= 0 or self.resume_interrupt_at >= self.resume_steps:
            raise ValueError("resume_interrupt_at must lie strictly inside resume_steps.")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def nano(cls, seed: int = 42) -> "StageDConfig":
        config = cls(seed=seed)
        config.validate()
        return config


@dataclass(frozen=True)
class CorpusDocument:
    identifier: str
    text: str

    @property
    def fingerprint(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PackedExample:
    ids: Tuple[int, ...]
    document_id: str


class LocalSyntheticCorpus:
    """Versioned local corpus with deterministic splits and auditable manifests."""

    source = "data/stage_d/synthetic_corpus.jsonl"
    revision = "stage_d_synthetic_pattern_corpus_v1"

    def __init__(self, documents: Sequence[CorpusDocument]) -> None:
        if len(documents) < 6:
            raise ValueError("The Stage D corpus needs at least six documents for deterministic splits.")
        if len({document.identifier for document in documents}) != len(documents):
            raise ValueError("Corpus document identifiers must be unique.")
        self.documents = tuple(documents)

    @classmethod
    def default(cls) -> "LocalSyntheticCorpus":
        # This committed JSONL file is deliberately local/synthetic. Its
        # patterns exercise causal learning without making a quality claim.
        source_path = Path(__file__).resolve().parents[2] / cls.source
        documents = []
        for line in source_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            documents.append(CorpusDocument(str(row["id"]), str(row["text"])))
        return cls(documents)

    def tokenizer(self, config: StageDConfig) -> EthioBBPETokenizer:
        tokenizer_config = TokenizerConfig(max_chunk_length=config.chunk_length, embedding_dim=4)
        return EthioBBPETokenizer.from_pretrained(tokenizer_config)

    def split(self, seed: int) -> Dict[str, Tuple[CorpusDocument, ...]]:
        indices = list(range(len(self.documents)))
        random.Random(seed).shuffle(indices)
        train_count = int(len(indices) * 2 / 3)
        validation_count = max(1, (len(indices) - train_count) // 2)
        partitions = {
            "train": tuple(self.documents[index] for index in indices[:train_count]),
            "validation": tuple(self.documents[index] for index in indices[train_count: train_count + validation_count]),
            "test": tuple(self.documents[index] for index in indices[train_count + validation_count:]),
        }
        if not partitions["test"]:
            raise RuntimeError("Deterministic corpus split produced an empty test partition.")
        return partitions

    def manifest(self, tokenizer: EthioBBPETokenizer, config: StageDConfig) -> Dict[str, Any]:
        splits = self.split(config.seed)
        split_payload = {}
        for name, documents in splits.items():
            split_payload[name] = {
                "document_ids": [document.identifier for document in documents],
                "document_hashes": {document.identifier: document.fingerprint for document in documents},
                "document_count": len(documents),
                "character_count": sum(len(document.text) for document in documents),
                "token_count": sum(len(tokenizer.encode(document.text).ids) for document in documents),
            }
        all_hashes = {document.identifier: document.fingerprint for document in self.documents}
        manifest = {
            "format": "dcss-cdi-stage-d-data-manifest-v1",
            "source": self.source,
            "revision": self.revision,
            "source_url_or_local_source": self.source,
            "preprocessing": {"normalization": tokenizer.config.normalization, "whitespace_policy": tokenizer.config.whitespace_policy, "boundary_policy": "bos_eos_per_document_no_cross_document_packing"},
            "corpus_content_hash": sha256("".join(document.fingerprint for document in self.documents).encode("utf-8")).hexdigest(),
            "all_document_hashes": all_hashes,
            "splits": split_payload,
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "config": config.as_dict(),
        }
        manifest["fingerprint"] = sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return manifest


def pack_documents(documents: Sequence[CorpusDocument], tokenizer: EthioBBPETokenizer, chunk_length: int) -> Tuple[List[PackedExample], int]:
    """Chunk each document independently; no sequence crosses a document boundary."""
    examples: List[PackedExample] = []
    truncation_count = 0
    for document in documents:
        encoded = tokenizer.encode(document.text, add_special_tokens=True)
        values = list(encoded.ids)
        for start in range(0, len(values), chunk_length):
            chunk = values[start: start + chunk_length]
            if len(chunk) >= 2:
                examples.append(PackedExample(tuple(chunk), document.identifier))
    if not examples:
        raise ValueError("No usable causal chunks were packed from the provided documents.")
    return examples, truncation_count


def collate_examples(examples: Sequence[PackedExample], tokenizer: EthioBBPETokenizer, chunk_length: int) -> Dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("Cannot collate an empty example list.")
    encoded = [EncodedText(example.ids, False, "") for example in examples]
    ids, mask, _ = tokenizer.pad(encoded, max_length=chunk_length)
    return {"input_ids": ids, "attention_mask": mask, "document_ids": [example.document_id for example in examples]}


def deterministic_batches(examples: Sequence[PackedExample], tokenizer: EthioBBPETokenizer, config: StageDConfig) -> List[Dict[str, torch.Tensor]]:
    batches = []
    for start in range(0, len(examples), config.batch_size):
        group = list(examples[start: start + config.batch_size])
        while len(group) < config.batch_size:
            group.append(examples[len(group) % len(examples)])
        batches.append(collate_examples(group, tokenizer, config.chunk_length))
    return batches


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def build_model(name: str, tokenizer: EthioBBPETokenizer, config: StageDConfig) -> nn.Module:
    if name == "dcss_cdi":
        stage_c = StageCConfig.nano(seed=config.seed)
        return DCSSLanguageModel(tokenizer, stage_c)
    if name == "v2":
        return LegacyCDIV2Adapter(tokenizer, width=tokenizer.config.embedding_dim, dtype=config.dtype)
    if name == "transformer":
        return TinyTransformerBaseline(tokenizer, width=tokenizer.config.embedding_dim, dtype=config.dtype)
    raise ValueError(f"Unknown Stage D model: {name}")


def optimizer_for(model: nn.Module, config: StageDConfig) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    identities = [id(parameter) for parameter in parameters]
    if len(identities) != len(set(identities)):
        raise RuntimeError("A trainable parameter appears more than once in the optimizer inventory.")
    return torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def model_loss(model: nn.Module, batch: Mapping[str, Any]):
    return model.causal_loss(batch["input_ids"], batch["attention_mask"])


def train_steps(
    model: nn.Module,
    batches: Sequence[Mapping[str, Any]],
    config: StageDConfig,
    steps: int,
    optimizer: torch.optim.Optimizer | None = None,
    start_cursor: int = 0,
    *,
    shuffle_each_epoch: bool = False,
) -> Tuple[List[float], torch.optim.Optimizer, int]:
    """Train a fixed number of steps with reproducible fixed or shuffled batch order.

    When ``shuffle_each_epoch`` is enabled, a local RNG derives one permutation
    from ``config.seed + epoch``.  This preserves deterministic resume behavior
    while avoiding repeated exposure to only the first batches of a larger
    corpus.
    """
    if not batches:
        raise ValueError("train_steps requires at least one batch.")
    model.train()
    optimizer = optimizer or optimizer_for(model, config)
    losses: List[float] = []
    cursor = start_cursor
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if shuffle_each_epoch:
            epoch = cursor // len(batches)
            order = list(range(len(batches)))
            random.Random(config.seed + epoch).shuffle(order)
            batch = batches[order[cursor % len(batches)]]
        else:
            batch = batches[cursor % len(batches)]
        report = model_loss(model, batch)
        if not torch.isfinite(report.loss):
            raise FloatingPointError("Encountered a non-finite Stage D causal loss.")
        report.loss.backward()
        gradients = [
            (name, parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.endswith("learned_initial_state")
        ]
        invalid_gradients = [name for name, gradient in gradients if gradient is None or not bool(torch.isfinite(gradient).all().item())]
        if invalid_gradients:
            raise FloatingPointError(f"Active trainable parameters have missing or non-finite gradients: {invalid_gradients}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        losses.append(float(report.loss.detach().cpu()))
        cursor += 1
    return losses, optimizer, cursor


def evaluate(model: nn.Module, batches: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    losses, tokens = [], 0
    with torch.no_grad():
        for batch in batches:
            report = model_loss(model, batch)
            losses.append(float(report.loss.cpu()))
            tokens += report.token_count
    if was_training:
        model.train()
    mean_loss = sum(losses) / max(len(losses), 1)
    return {"loss": mean_loss, "perplexity": float(torch.exp(torch.tensor(mean_loss))), "token_count": tokens}


def _random_state_payload() -> Dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def _restore_random_state(payload: Mapping[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])


def checkpoint_payload(model: nn.Module, optimizer: torch.optim.Optimizer, tokenizer: EthioBBPETokenizer, data_manifest: Mapping[str, Any], config: StageDConfig, step: int, cursor: int) -> Dict[str, Any]:
    topology_fingerprint = None
    if isinstance(model, DCSSLanguageModel):
        topology_fingerprint = model.ssm.cell.topology.fingerprint()
    return {
        "format": "dcss-cdi-stage-d-checkpoint-v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "tokenizer_artifact": tokenizer.artifact(),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "data_manifest": dict(data_manifest),
        "config": config.as_dict(),
        "global_step": step,
        "cursor": cursor,
        "random_state": _random_state_payload(),
        "topology_fingerprint": topology_fingerprint,
        "hardware": {"device": config.device, "dtype": config.dtype_str, "torch": torch.__version__},
    }


def restore_checkpoint(payload: Mapping[str, Any], model: nn.Module, optimizer: torch.optim.Optimizer, tokenizer: EthioBBPETokenizer, *, allow_tokenizer_conversion: bool = False) -> Tuple[int, int]:
    if payload.get("format") != "dcss-cdi-stage-d-checkpoint-v1":
        raise ValueError("Unsupported Stage D checkpoint format.")
    if payload.get("tokenizer_fingerprint") != tokenizer.fingerprint and not allow_tokenizer_conversion:
        raise ValueError("Tokenizer fingerprint mismatch; explicit conversion is required.")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_random_state(payload["random_state"])
    return int(payload["global_step"]), int(payload["cursor"])


def parameter_fingerprint(model: nn.Module) -> str:
    digest = sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
