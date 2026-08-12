"""Local sparse retrieval with provenance and contradiction reporting."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence

from .audit import AuditTrail
from .memory import EpisodicMemory, MemoryRecord


@dataclass(frozen=True)
class RetrievedRecord:
    record_id: str
    content: str
    score: float
    source_id: str
    offset: int
    namespace: str
    provenance: Dict[str, Any]
    content_hash: str
    untrusted: bool


class Retriever:
    """Indexes local records and returns source-tagged sparse retrieval results."""

    def __init__(self, memory: EpisodicMemory, audit: AuditTrail | None = None) -> None:
        self.memory = memory
        self.audit = audit or memory.audit

    def index(self, records: Sequence[MemoryRecord]) -> None:
        for record in records:
            self.memory.write(record, importance=1.0, explicit=True)
        self.audit.append("retriever_index", {"record_count": len(records), "memory_size": len(self.memory.records())})

    def query(self, query: str, k: int = 3, namespace: str | None = None) -> List[RetrievedRecord]:
        records, state = self.memory.retrieve(query, k=k, namespace=namespace)
        result = [
            RetrievedRecord(
                record_id=record.record_id,
                content=record.content,
                score=score,
                source_id=record.source_id,
                offset=record.offset,
                namespace=record.namespace,
                provenance=dict(record.provenance),
                content_hash=record.content_hash,
                untrusted=record.namespace == "untrusted",
            )
            for record, score in records
        ]
        self.audit.append("retriever_query", {"query": query, "result_ids": [item.record_id for item in result], "candidate_ids": list(state.candidate_ids), "namespace": namespace})
        return result

    def explain(self, records: Sequence[RetrievedRecord]) -> Dict[str, Any]:
        return {"records": [asdict(record) for record in records], "all_have_provenance": all(record.source_id and record.offset >= 0 and record.content_hash for record in records), "untrusted_record_ids": [record.record_id for record in records if record.untrusted]}

    def contradictions(self, records: Sequence[RetrievedRecord]) -> Dict[str, Any]:
        normalized = {record.content.casefold().replace("open", "<status>").replace("closed", "<status>"): [] for record in records}
        for record in records:
            key = record.content.casefold().replace("open", "<status>").replace("closed", "<status>")
            normalized[key].append(record)
        conflicts = []
        for group in normalized.values():
            contents = {record.content.casefold() for record in group}
            if len(contents) > 1:
                conflicts.append([record.record_id for record in group])
        status = "CONFLICT" if conflicts else "NO_CONFLICT"
        self.audit.append("retriever_contradiction_check", {"status": status, "conflicts": conflicts})
        return {"status": status, "conflicts": conflicts}
