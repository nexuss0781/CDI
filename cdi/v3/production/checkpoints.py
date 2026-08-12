"""Atomic, hash-verified local checkpoint storage for offline P1 runs."""
from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping

import torch

from .config import ReleaseBoundary
from .lineage import ArtifactLineage, EnvironmentLineage


CHECKPOINT_FORMAT = "dcss-cdi-production-checkpoint-v1"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def build_envelope(stage_d_payload: Mapping[str, Any], lineage: ArtifactLineage, boundary: ReleaseBoundary, environment: EnvironmentLineage | None = None) -> Dict[str, Any]:
    lineage.validate()
    boundary.validate()
    if stage_d_payload.get("format") != "dcss-cdi-stage-d-checkpoint-v1":
        raise ValueError("P1 envelopes require a validated Stage D checkpoint payload.")
    return {
        "format": CHECKPOINT_FORMAT,
        "stage_d_payload": dict(stage_d_payload),
        "lineage": lineage.as_dict(),
        "lineage_fingerprint": lineage.fingerprint,
        "release_boundary": boundary.as_dict(),
        "environment": (environment or EnvironmentLineage.current()).as_dict(),
    }


def validate_envelope(payload: Mapping[str, Any]) -> None:
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported production checkpoint format.")
    ArtifactLineage(**dict(payload.get("lineage", {}))).validate()
    ReleaseBoundary(**dict(payload.get("release_boundary", {}))).validate()
    stage_d = payload.get("stage_d_payload", {})
    if not isinstance(stage_d, Mapping) or stage_d.get("format") != "dcss-cdi-stage-d-checkpoint-v1":
        raise ValueError("Production checkpoint lacks a valid Stage D payload.")
    expected = ArtifactLineage(**dict(payload["lineage"])).fingerprint
    if payload.get("lineage_fingerprint") != expected:
        raise ValueError("Production checkpoint lineage fingerprint mismatch.")


def save_atomic(path: str | Path, envelope: Mapping[str, Any]) -> Dict[str, str]:
    """Save a local checkpoint atomically and write a neighbouring integrity sidecar."""
    validate_envelope(envelope)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(envelope), temporary)
        digest = file_sha256(temporary)
        os.replace(temporary, target)
        sidecar = checkpoint_sidecar(target)
        sidecar.write_text(f"format={CHECKPOINT_FORMAT}\nsha256={digest}\n", encoding="utf-8")
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(target), "sha256": digest, "sidecar": str(checkpoint_sidecar(target))}


def load_verified(path: str | Path) -> Dict[str, Any]:
    """Verify sidecar integrity before loading a trusted local P1 checkpoint."""
    target = Path(path)
    sidecar = checkpoint_sidecar(target)
    if not target.is_file() or not sidecar.is_file():
        raise FileNotFoundError("Checkpoint and integrity sidecar are both required.")
    fields = dict(line.split("=", 1) for line in sidecar.read_text(encoding="utf-8").splitlines() if "=" in line)
    if fields.get("format") != CHECKPOINT_FORMAT or fields.get("sha256") != file_sha256(target):
        raise ValueError("Checkpoint integrity verification failed.")
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a mapping.")
    validate_envelope(payload)
    return payload
