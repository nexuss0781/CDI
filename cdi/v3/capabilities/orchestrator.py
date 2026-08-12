"""Bounded planning, execution, verification, and composition for Stage F."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import ast
from hashlib import sha256
from typing import Any, Dict, List, Mapping, Sequence

import torch

from .audit import AuditTrail
from .memory import EpisodicMemory, MemoryRecord
from .retrieval import Retriever, RetrievedRecord
from .tools import ToolRegistry, ToolResult, _safe_arithmetic, register_default_tools


@dataclass(frozen=True)
class PlanAction:
    action_id: str
    tool_name: str
    payload: Dict[str, Any]
    preconditions: tuple[str, ...] = ()
    expected_status: str = "OK"


@dataclass(frozen=True)
class Plan:
    actions: tuple[PlanAction, ...]
    max_steps: int
    stop_on_error: bool = True


class Planner:
    """Validates only explicitly supplied typed plans; it is not an autonomous planner."""

    def __init__(self, registry: ToolRegistry, audit: AuditTrail | None = None, max_steps: int = 4) -> None:
        self.registry = registry
        self.audit = audit or registry.audit
        self.max_steps = max_steps

    def plan(self, actions: Sequence[PlanAction]) -> Plan:
        plan = Plan(tuple(actions), max_steps=self.max_steps)
        self.audit.append("plan_created", {"action_ids": [action.action_id for action in plan.actions], "max_steps": self.max_steps})
        return plan

    def validate_plan(self, plan: Plan) -> Dict[str, Any]:
        reasons = []
        ids = [action.action_id for action in plan.actions]
        if len(plan.actions) == 0:
            reasons.append("empty_plan")
        if len(plan.actions) > min(plan.max_steps, self.max_steps):
            reasons.append("step_budget_exceeded")
        if len(ids) != len(set(ids)):
            reasons.append("duplicate_action_id")
        known = set()
        for action in plan.actions:
            if action.action_id in action.preconditions:
                reasons.append(f"self_cycle:{action.action_id}")
            if any(precondition not in known for precondition in action.preconditions):
                reasons.append(f"missing_or_cyclic_precondition:{action.action_id}")
            validation = self.registry.validate(action.tool_name, action.payload)
            if validation.status != "VALID":
                reasons.append(f"invalid_tool_action:{action.action_id}:{validation.reason or validation.status}")
            known.add(action.action_id)
        status = "VALID" if not reasons else "REJECTED"
        result = {"status": status, "reasons": reasons, "action_count": len(plan.actions)}
        self.audit.append("plan_validate", result)
        return result

    def revise(self, plan: Plan, failed_action_id: str) -> Plan:
        revised = Plan(tuple(action for action in plan.actions if action.action_id != failed_action_id), plan.max_steps, plan.stop_on_error)
        self.audit.append("plan_revise", {"failed_action_id": failed_action_id, "remaining_action_ids": [action.action_id for action in revised.actions]})
        return revised


class Executor:
    """Dry-run sequential executor with explicit cancellation and recoverable audit state."""

    def __init__(self, registry: ToolRegistry, planner: Planner, audit: AuditTrail | None = None) -> None:
        self.registry = registry
        self.planner = planner
        self.audit = audit or registry.audit
        self.cancelled = False

    def cancel(self, reason: str = "user_or_test_cancellation") -> None:
        self.cancelled = True
        self.audit.append("executor_cancel", {"reason": reason})

    def dry_run(self, plan: Plan) -> Dict[str, Any]:
        return self.run(plan, dry_run=True)

    def run(self, plan: Plan, dry_run: bool = True) -> Dict[str, Any]:
        if not dry_run or not self.registry.dry_run:
            result = {"status": "REJECTED", "reason": "external_execution_not_enabled", "events": []}
            self.audit.append("executor_reject", result)
            return result
        validation = self.planner.validate_plan(plan)
        if validation["status"] != "VALID":
            result = {"status": "REJECTED", "reason": "invalid_plan", "validation": validation, "events": []}
            self.audit.append("executor_reject", result)
            return result
        events = []
        completed = set()
        for action in plan.actions:
            if self.cancelled:
                result = {"status": "CANCELLED", "reason": "cancelled", "events": events}
                self.audit.append("executor_complete", result)
                return result
            if not set(action.preconditions).issubset(completed):
                result = {"status": "REJECTED", "reason": "precondition_not_satisfied", "events": events}
                self.audit.append("executor_complete", result)
                return result
            tool_result = self.registry.invoke(action.tool_name, action.payload)
            event = {"action_id": action.action_id, "tool_name": action.tool_name, "payload": action.payload, "status": tool_result.status, "output": tool_result.output, "reason": tool_result.reason}
            events.append(event)
            self.audit.append("executor_action", event)
            if tool_result.status != action.expected_status:
                result = {"status": "STOPPED", "reason": "tool_failure_or_unexpected_status", "events": events}
                self.audit.append("executor_complete", result)
                return result
            completed.add(action.action_id)
        result = {"status": "OK", "events": events}
        self.audit.append("executor_complete", result)
        return result


class Verifier:
    """Independent local verifier with pass, fail, and explicit unknown outcomes."""

    def __init__(self, audit: AuditTrail | None = None) -> None:
        self.audit = audit or AuditTrail()
        self.version = "stage-f-verifier-v1"

    def check(self, claim: Mapping[str, Any], evidence: Sequence[RetrievedRecord] = ()) -> Dict[str, Any]:
        kind = claim.get("kind")
        if kind == "arithmetic":
            try:
                actual = _safe_arithmetic(str(claim["expression"]))
                claimed = float(claim["claimed"])
            except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
                result = {"status": "UNKNOWN", "reason": f"invalid_or_unsupported_arithmetic:{type(exc).__name__}", "claim": dict(claim), "checker_version": self.version}
            else:
                status = "VERIFIED" if abs(actual - claimed) <= 1e-9 else "FAILED"
                result = {"status": status, "claim": dict(claim), "actual": actual, "checker_version": self.version}
        elif kind == "citation":
            source_id = claim.get("source_id")
            needle = str(claim.get("contains", ""))
            supported = [record for record in evidence if record.source_id == source_id and needle.casefold() in record.content.casefold()]
            if supported:
                result = {"status": "VERIFIED", "claim": dict(claim), "evidence": [{"record_id": record.record_id, "source_id": record.source_id, "offset": record.offset, "content_hash": record.content_hash} for record in supported], "checker_version": self.version}
            elif evidence:
                result = {"status": "FAILED", "reason": "citation_not_supported_by_retrieved_evidence", "claim": dict(claim), "checker_version": self.version}
            else:
                result = {"status": "UNKNOWN", "reason": "no_retrieved_evidence", "claim": dict(claim), "checker_version": self.version}
        else:
            result = {"status": "UNKNOWN", "reason": "unsupported_claim_kind", "claim": dict(claim), "checker_version": self.version}
        self.audit.append("verify", result)
        return result

    def explain(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(result)

    def abstain(self, claim: Mapping[str, Any], reason: str = "insufficient_evidence") -> Dict[str, Any]:
        result = {"status": "UNKNOWN", "reason": reason, "claim": dict(claim), "checker_version": self.version}
        self.audit.append("verify_abstain", result)
        return result


class CapabilityOrchestrator:
    """Visible composition shell; modules are optional and never alter core weights."""

    def __init__(self, core: Any | None = None, memory: EpisodicMemory | None = None, retriever: Retriever | None = None, registry: ToolRegistry | None = None, planner: Planner | None = None, executor: Executor | None = None, verifier: Verifier | None = None, audit: AuditTrail | None = None) -> None:
        self.audit = audit or AuditTrail()
        self.core = core
        self.memory = memory or EpisodicMemory(audit=self.audit)
        self.retriever = retriever or Retriever(self.memory, audit=self.audit)
        self.registry = registry or ToolRegistry(audit=self.audit, dry_run=True)
        if not self.registry.definitions():
            register_default_tools(self.registry, self._lookup_tool)
        self.planner = planner or Planner(self.registry, audit=self.audit)
        self.executor = executor or Executor(self.registry, self.planner, audit=self.audit)
        self.verifier = verifier or Verifier(audit=self.audit)

    def _lookup_tool(self, query: str) -> Dict[str, Any]:
        records = self.retriever.query(query, k=1)
        return {"result": records[0].content if records else "", "provenance": self.retriever.explain(records)}

    def ingest(self, records: Sequence[MemoryRecord]) -> None:
        self.retriever.index(records)
        self.audit.append("orchestrator_ingest", {"record_ids": [record.record_id for record in records]})

    def answer(self, query: str, k: int = 3) -> Dict[str, Any]:
        retrieved = self.retriever.query(query, k=k)
        explanation = self.retriever.explain(retrieved)
        result = {"generated": {"status": "NOT_GENERATED", "reason": "Stage F diagnostic orchestrator does not claim free-form model answers."}, "retrieved": explanation, "verified": {"status": "UNKNOWN", "reason": "no claim submitted for verification"}, "query": query}
        self.audit.append("orchestrator_answer", result)
        return result

    def execute(self, plan: Plan) -> Dict[str, Any]:
        result = self.executor.dry_run(plan)
        self.audit.append("orchestrator_execute", {"plan_action_ids": [action.action_id for action in plan.actions], "status": result["status"]})
        return result

    def audit_log(self) -> Dict[str, Any]:
        return {"events": self.audit.events(), "fingerprint": self.audit.fingerprint, "valid": self.audit.verify(), "external_side_effects_enabled": False}
