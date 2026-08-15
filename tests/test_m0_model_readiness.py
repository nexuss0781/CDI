from __future__ import annotations

from benchmarks.m0_model_readiness import run_all


def test_m0_model_readiness_all_gates_pass(tmp_path):
    report = run_all(tmp_path / "m0")
    assert report["status"] == "PASS"
    assert all(gate["passed"] for gate in report["gates"])
    assert report["gates"][-1]["details"]["peak_rss_gib"] <= 11.0
