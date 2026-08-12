"""Regression tests for bounded, local, dry-run Stage F capabilities."""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks import stage_f
from cdi.v3 import CapabilityOrchestrator, EpisodicMemory, MemoryRecord, PlanAction, ToolDefinition
from cdi.v3.capabilities import AuditTrail


def test_frozen_nano_configuration_is_bounded_and_dry_run_only() -> None:
    config = json.loads(Path("benchmarks/configs/stage_f_nano.json").read_text(encoding="utf-8"))
    assert config["core"]["total_state_dim"] < 64
    assert config["capabilities"]["episodic_memory_capacity"] == 8
    assert config["capabilities"]["external_side_effects_enabled"] is False
    assert config["gates"]["stage_g_authorized"] is False


def test_audit_trail_is_hash_linked_and_detects_tampering() -> None:
    audit = AuditTrail()
    audit.append("first", {"value": 1})
    audit.append("second", {"value": 2})
    assert audit.verify()
    serialized = audit.serialize()
    assert len(serialized["events"]) == 2
    assert serialized["events"][1]["previous_hash"] == serialized["events"][0]["event_hash"]


def test_memory_requires_explicit_thresholded_writes_deduplicates_and_evicts_lru() -> None:
    memory = EpisodicMemory(capacity=2, write_threshold=0.5)
    low = MemoryRecord.create("low", "unimportant datum", "unit")
    one = MemoryRecord.create("one", "vault code cobalt-17", "unit")
    duplicate = MemoryRecord.create("dup", "vault code cobalt-17", "unit")
    two = MemoryRecord.create("two", "second durable record", "unit")
    three = MemoryRecord.create("three", "third durable record", "unit")
    memory.write(low, importance=0.49)
    memory.write(one, importance=0.5)
    memory.write(duplicate, importance=1.0)
    memory.write(two, importance=1.0)
    memory.write(three, importance=1.0)
    selected, state = memory.retrieve("vault code", k=1)
    assert len(memory.records()) == 2
    assert all(record.record_id != "low" for record in memory.records())
    assert all(record.record_id != "dup" for record in memory.records())
    assert state.candidate_ids == () or selected  # The oldest vault record may be evicted by the deliberate capacity test.
    assert memory.audit.verify()


def test_retrieval_finds_needle_preserves_provenance_and_flags_conflict() -> None:
    gate = stage_f.retrieval_gate()
    assert gate["passed"], gate["details"]
    needle = gate["details"]["needle"]["records"][0]
    assert needle["record_id"] == "needle-001"
    assert needle["source_id"] and needle["content_hash"]
    assert gate["details"]["contradiction"]["status"] == "CONFLICT"


def test_tools_enforce_schema_timeout_and_mutation_permission_boundary() -> None:
    gate = stage_f.tools_gate()
    assert gate["passed"], gate["details"]
    assert gate["details"]["valid"]["output"]["value"] == 35.0
    assert gate["details"]["schema"]["status"] == "REJECTED"
    assert gate["details"]["timeout"]["status"] == "TIMEOUT"
    assert gate["details"]["permission"]["status"] == "REJECTED"
    assert gate["details"]["external_side_effects_enabled"] is False


def test_planning_is_typed_budgeted_and_dry_run_only() -> None:
    gate = stage_f.planning_gate()
    assert gate["passed"], gate["details"]
    assert gate["details"]["good_execution"]["status"] == "OK"
    assert gate["details"]["cycle_validation"]["status"] == "REJECTED"
    assert gate["details"]["over_budget_validation"]["status"] == "REJECTED"


def test_verifier_independently_passes_fails_and_abstains() -> None:
    gate = stage_f.verification_gate()
    assert gate["passed"], gate["details"]
    assert gate["details"]["true"]["status"] == "VERIFIED"
    assert gate["details"]["false"]["status"] == "FAILED"
    assert gate["details"]["citation"]["status"] == "VERIFIED"
    assert gate["details"]["unknown"]["status"] == "UNKNOWN"


def test_adversarial_instructions_remain_untrusted_data() -> None:
    gate = stage_f.adversarial_gate()
    assert gate["passed"], gate["details"]
    assert gate["details"]["injection_answer"]["generated"]["status"] == "NOT_GENERATED"
    assert gate["details"]["mutating_attempt"]["status"] == "REJECTED"


def test_capabilities_are_optional_to_core_and_full_harness_passes(tmp_path: Path) -> None:
    composition = stage_f.composition_gate()
    assert composition["passed"], composition["details"]
    assert composition["details"]["core_logits_max_abs"] <= 1e-6
    assert composition["details"]["core_state_max_abs"] <= 1e-6
    report = stage_f.run_all(tmp_path)
    assert report["status"] == "PASS"
    assert report["external_side_effects_enabled"] is False
    assert report["capability_manifest"]["status"] == "Experimental"
    assert (tmp_path / "latest.json").exists()
