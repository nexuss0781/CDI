"""Replayable hash-linked audit events for bounded Stage F capability modules."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class AuditEvent:
    event_id: int
    kind: str
    payload: Dict[str, Any]
    previous_hash: str
    event_hash: str


class AuditTrail:
    """Append-only local audit trail with deterministic event hashes."""

    def __init__(self) -> None:
        self._events: List[AuditEvent] = []
        self._last_hash = "0" * 64

    @staticmethod
    def _canonical(value: Mapping[str, Any]) -> str:
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))

    def append(self, kind: str, payload: Mapping[str, Any]) -> AuditEvent:
        if not kind or not isinstance(kind, str):
            raise ValueError("Audit event kind must be a non-empty string.")
        event_id = len(self._events)
        normalized_payload = json.loads(self._canonical(dict(payload)))
        envelope = {"event_id": event_id, "kind": kind, "payload": normalized_payload, "previous_hash": self._last_hash}
        event_hash = sha256(self._canonical(envelope).encode("utf-8")).hexdigest()
        event = AuditEvent(event_id, kind, normalized_payload, self._last_hash, event_hash)
        self._events.append(event)
        self._last_hash = event_hash
        return event

    def events(self) -> List[Dict[str, Any]]:
        return [asdict(event) for event in self._events]

    def serialize(self) -> Dict[str, Any]:
        """Return a replayable, canonical representation of the complete trail."""
        return {"format": "dcss-cdi-stage-f-audit-v1", "events": self.events(), "fingerprint": self.fingerprint, "valid": self.verify()}

    @classmethod
    def replay(cls, payload: Mapping[str, Any]) -> "AuditTrail":
        """Reconstruct and validate a serialized trail without trusting supplied hashes."""
        if payload.get("format") != "dcss-cdi-stage-f-audit-v1":
            raise ValueError("Unsupported audit format.")
        trail = cls()
        for raw in payload.get("events", []):
            event = trail.append(str(raw["kind"]), dict(raw["payload"]))
            if event.event_id != raw.get("event_id") or event.previous_hash != raw.get("previous_hash") or event.event_hash != raw.get("event_hash"):
                raise ValueError("Audit replay hash mismatch.")
        if trail.fingerprint != payload.get("fingerprint") or not payload.get("valid", False):
            raise ValueError("Audit replay fingerprint mismatch.")
        return trail

    @property
    def fingerprint(self) -> str:
        return self._last_hash

    def verify(self) -> bool:
        previous = "0" * 64
        for index, event in enumerate(self._events):
            if event.event_id != index or event.previous_hash != previous:
                return False
            envelope = {"event_id": event.event_id, "kind": event.kind, "payload": event.payload, "previous_hash": event.previous_hash}
            expected = sha256(self._canonical(envelope).encode("utf-8")).hexdigest()
            if expected != event.event_hash:
                return False
            previous = event.event_hash
        return previous == self._last_hash
