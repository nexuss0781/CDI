"""Bounded explicit-write episodic memory and sparse local retrieval."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .audit import AuditTrail

_TOKEN_PATTERN = re.compile(r"[\w-]+", re.UNICODE)


def terms(text: str) -> Tuple[str, ...]:
    return tuple(sorted(set(token.casefold() for token in _TOKEN_PATTERN.findall(text))))


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    content: str
    source_id: str
    offset: int
    sequence_index: int
    namespace: str
    retention_policy: str
    provenance: Dict[str, Any]
    content_hash: str

    @classmethod
    def create(
        cls,
        record_id: str,
        content: str,
        source_id: str,
        offset: int = 0,
        sequence_index: int = 0,
        namespace: str = "default",
        retention_policy: str = "ephemeral",
        provenance: Mapping[str, Any] | None = None,
    ) -> "MemoryRecord":
        if not content.strip():
            raise ValueError("Memory records require non-empty content.")
        if not record_id or not source_id or offset < 0 or sequence_index < 0:
            raise ValueError("Memory record identifiers and indices must be valid.")
        return cls(record_id, content, source_id, offset, sequence_index, namespace, retention_policy, dict(provenance or {}), sha256(content.encode("utf-8")).hexdigest())


@dataclass(frozen=True)
class RetrievalState:
    query: str
    candidate_ids: Tuple[str, ...]
    selected_ids: Tuple[str, ...]


class EpisodicMemory:
    """Small LRU memory; similarity scoring occurs only over token-index candidates."""

    def __init__(self, capacity: int = 8, write_threshold: float = 0.5, audit: AuditTrail | None = None) -> None:
        if capacity <= 0 or not 0.0 <= write_threshold <= 1.0:
            raise ValueError("Memory capacity must be positive and write_threshold must be in [0, 1].")
        self.capacity = capacity
        self.write_threshold = write_threshold
        self.audit = audit or AuditTrail()
        self._records: Dict[str, MemoryRecord] = {}
        self._lru: List[str] = []
        self._term_index: Dict[str, set[str]] = {}

    def _index(self, record: MemoryRecord) -> None:
        for token in terms(record.content):
            self._term_index.setdefault(token, set()).add(record.record_id)

    def _deindex(self, record: MemoryRecord) -> None:
        for token in terms(record.content):
            values = self._term_index.get(token)
            if values is not None:
                values.discard(record.record_id)
                if not values:
                    del self._term_index[token]

    def _touch(self, record_id: str) -> None:
        if record_id in self._lru:
            self._lru.remove(record_id)
        self._lru.append(record_id)

    def write(self, record: MemoryRecord, importance: float, explicit: bool = True) -> "EpisodicMemory":
        if not explicit:
            self.audit.append("memory_write_rejected", {"record_id": record.record_id, "reason": "explicit_write_required"})
            return self
        if importance < self.write_threshold:
            self.audit.append("memory_write_rejected", {"record_id": record.record_id, "reason": "below_threshold", "importance": importance})
            return self
        existing = next((item for item in self._records.values() if item.content_hash == record.content_hash and item.namespace == record.namespace), None)
        if existing is not None:
            self._touch(existing.record_id)
            self.audit.append("memory_write_deduplicated", {"record_id": record.record_id, "existing_record_id": existing.record_id, "content_hash": record.content_hash})
            return self
        if record.record_id in self._records:
            self._deindex(self._records[record.record_id])
        self._records[record.record_id] = record
        self._index(record)
        self._touch(record.record_id)
        evicted = self.evict()
        self.audit.append("memory_write", {"record_id": record.record_id, "source_id": record.source_id, "namespace": record.namespace, "importance": importance, "evicted": evicted})
        return self

    def evict(self) -> List[str]:
        evicted = []
        while len(self._records) > self.capacity:
            oldest = self._lru.pop(0)
            record = self._records.pop(oldest)
            self._deindex(record)
            evicted.append(oldest)
            self.audit.append("memory_evict", {"record_id": oldest, "reason": "capacity"})
        return evicted

    def retrieve(self, query: str, k: int = 3, namespace: str | None = None) -> Tuple[List[Tuple[MemoryRecord, float]], RetrievalState]:
        if k <= 0:
            raise ValueError("k must be positive.")
        query_terms = terms(query)
        candidate_ids = set()
        for token in query_terms:
            candidate_ids.update(self._term_index.get(token, set()))
        if namespace is not None:
            candidate_ids = {record_id for record_id in candidate_ids if self._records[record_id].namespace == namespace}
        scored = []
        query_set = set(query_terms)
        for record_id in candidate_ids:
            record = self._records[record_id]
            record_terms = set(terms(record.content))
            score = len(query_set.intersection(record_terms)) / max(len(query_set.union(record_terms)), 1)
            scored.append((record, score))
        scored.sort(key=lambda item: (-item[1], item[0].record_id))
        selected = scored[:k]
        for record, _ in selected:
            self._touch(record.record_id)
        state = RetrievalState(query, tuple(sorted(candidate_ids)), tuple(record.record_id for record, _ in selected))
        self.audit.append("memory_retrieve", {"query": query, "candidate_count": len(candidate_ids), "candidate_ids": list(state.candidate_ids), "selected_ids": list(state.selected_ids), "k": k})
        return selected, state

    def update(self, retrieval_state: RetrievalState) -> "EpisodicMemory":
        for record_id in retrieval_state.selected_ids:
            self._touch(record_id)
        self.audit.append("memory_update", {"selected_ids": list(retrieval_state.selected_ids)})
        return self

    def records(self) -> List[MemoryRecord]:
        return [self._records[record_id] for record_id in self._lru]

    def serialize(self) -> Dict[str, Any]:
        payload = {"format": "dcss-cdi-stage-f-memory-v1", "capacity": self.capacity, "write_threshold": self.write_threshold, "records": [asdict(record) for record in self.records()], "lru": list(self._lru)}
        payload["fingerprint"] = sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()
        return payload
