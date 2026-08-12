"""Content-addressed lineage and compatibility contracts for P1 artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from platform import platform, python_version
from typing import Any, Dict, Mapping

import torch


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, default=str, separators=(",", ":"))


def fingerprint(value: Mapping[str, Any]) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ArtifactLineage:
    """Immutable references required to reproduce or reject a model artifact."""

    code_revision: str
    run_config_fingerprint: str
    corpus_manifest_fingerprint: str
    tokenizer_fingerprint: str
    model_fingerprint: str
    parent_checkpoint_hash: str | None = None
    stage: str = "P1"

    def validate(self) -> None:
        required = {
            "code_revision": self.code_revision,
            "run_config_fingerprint": self.run_config_fingerprint,
            "corpus_manifest_fingerprint": self.corpus_manifest_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "model_fingerprint": self.model_fingerprint,
        }
        if self.stage != "P1":
            raise ValueError("Artifact lineage must identify the current P1 stage.")
        if any(not value or not isinstance(value, str) for value in required.values()):
            raise ValueError("Every required lineage field must be a non-empty string.")

    def as_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.as_dict())


@dataclass(frozen=True)
class EnvironmentLineage:
    """Minimal software environment record used for deterministic-run diagnosis."""

    python: str
    torch: str
    platform: str
    device: str
    dtype: str

    @classmethod
    def current(cls, device: str = "cpu", dtype: str = "float32") -> "EnvironmentLineage":
        return cls(python=python_version(), torch=torch.__version__, platform=platform(), device=device, dtype=dtype)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.as_dict())


def assert_compatible(expected: ArtifactLineage, observed: Mapping[str, Any], *, allow_parent_mismatch: bool = False) -> None:
    """Reject artifact restoration when immutable lineage diverges."""
    required = expected.as_dict()
    for key, value in required.items():
        if key == "parent_checkpoint_hash" and allow_parent_mismatch:
            continue
        if observed.get(key) != value:
            raise ValueError(f"Lineage mismatch for {key}: expected {value!r}, found {observed.get(key)!r}.")
