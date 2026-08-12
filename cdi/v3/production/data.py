"""Data-lineage and split-isolation primitives for offline P1 training hardening."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class GovernedDocument:
    identifier: str
    text: str
    source_uri: str
    license_id: str
    retention_policy: str
    data_class: str = "synthetic"
    pii_review: str = "not_applicable_synthetic"

    @property
    def content_hash(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()

    def validate(self, allowed_data_classes: Sequence[str] = ("synthetic",)) -> None:
        if not self.identifier or not self.text.strip():
            raise ValueError("Governed documents require a non-empty identifier and text.")
        if not self.source_uri or not self.license_id or not self.retention_policy:
            raise ValueError("Governed documents require source, license, and retention metadata.")
        if self.data_class not in set(allowed_data_classes):
            raise ValueError(f"Data class {self.data_class!r} is not admitted in this phase.")
        if self.data_class == "synthetic" and self.pii_review != "not_applicable_synthetic":
            raise ValueError("Synthetic records must preserve the explicit synthetic PII marker.")


@dataclass(frozen=True)
class P1DataPolicy:
    """P1 remains synthetic-only until a later data-admission gate is approved."""

    phase: str = "P1"
    allowed_data_classes: Tuple[str, ...] = ("synthetic",)
    real_corpus_training_authorized: bool = False

    def validate(self) -> None:
        if self.phase != "P1" or self.allowed_data_classes != ("synthetic",) or self.real_corpus_training_authorized:
            raise ValueError("P1 data policy permits synthetic records only.")


@dataclass(frozen=True)
class P2DataPolicy:
    """P2 admits governed rights-cleared pilot data and real-corpus pilot training."""

    phase: str = "P2"
    allowed_data_classes: Tuple[str, ...] = ("synthetic", "rights_cleared_pilot")
    real_corpus_training_authorized: bool = True

    def validate(self) -> None:
        if self.phase != "P2" or "rights_cleared_pilot" not in self.allowed_data_classes or not self.real_corpus_training_authorized:
            raise ValueError("P2 data policy requires rights-cleared pilot admission.")


@dataclass(frozen=True)
class DataManifest:
    """Immutable corpus/split description containing no hidden ingestion operation."""

    format: str
    policy: Dict[str, Any]
    documents: Dict[str, Dict[str, Any]]
    splits: Dict[str, Tuple[str, ...]]
    fingerprint: str

    @classmethod
    def build(cls, documents: Sequence[GovernedDocument], splits: Mapping[str, Sequence[str]], policy: P1DataPolicy | P2DataPolicy | None = None) -> "DataManifest":
        policy = policy or P1DataPolicy()
        policy.validate()
        if not documents:
            raise ValueError("A data manifest needs at least one governed document.")
        by_id = {}
        hash_to_id = {}
        for document in documents:
            document.validate(policy.allowed_data_classes)
            if document.identifier in by_id:
                raise ValueError(f"Duplicate document identifier: {document.identifier}")
            if document.content_hash in hash_to_id:
                raise ValueError(f"Duplicate content detected: {document.identifier} and {hash_to_id[document.content_hash]}")
            by_id[document.identifier] = document
            hash_to_id[document.content_hash] = document.identifier
        expected_splits = {"train", "validation", "test"}
        if set(splits) != expected_splits:
            raise ValueError("P1 manifests require exactly train, validation, and test splits.")
        seen = set()
        normalized_splits: Dict[str, Tuple[str, ...]] = {}
        for name in sorted(expected_splits):
            identifiers = tuple(str(identifier) for identifier in splits[name])
            if not identifiers:
                raise ValueError(f"Split {name} cannot be empty.")
            unknown = set(identifiers).difference(by_id)
            if unknown:
                raise ValueError(f"Split {name} contains unknown identifiers: {sorted(unknown)}")
            overlap = seen.intersection(identifiers)
            if overlap:
                raise ValueError(f"Split leakage detected for identifiers: {sorted(overlap)}")
            seen.update(identifiers)
            normalized_splits[name] = identifiers
        if seen != set(by_id):
            raise ValueError("Every governed document must occur in exactly one split.")
        document_payload = {
            identifier: {
                "content_hash": document.content_hash,
                "source_uri": document.source_uri,
                "license_id": document.license_id,
                "retention_policy": document.retention_policy,
                "data_class": document.data_class,
                "pii_review": document.pii_review,
                "character_count": len(document.text),
            }
            for identifier, document in sorted(by_id.items())
        }
        raw = {"format": "dcss-cdi-p1-data-manifest-v1", "policy": asdict(policy), "documents": document_payload, "splits": normalized_splits}
        digest = sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return cls(fingerprint=digest, **raw)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def assert_no_split_leakage(self) -> None:
        ids = [identifier for split in self.splits.values() for identifier in split]
        if len(ids) != len(set(ids)):
            raise ValueError("Identifier leakage found in manifest splits.")
        hashes = [self.documents[identifier]["content_hash"] for identifier in ids]
        if len(hashes) != len(set(hashes)):
            raise ValueError("Content-hash leakage found in manifest splits.")
