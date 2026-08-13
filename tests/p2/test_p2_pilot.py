"""Regression tests for P2 real-data pilot and scale-ladder comparison."""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.p2_pilot import matched_baseline_gate, negative_result_gate, pilot_data_gate, run_all


def test_p2_pilot_gates() -> None:
    data_res = pilot_data_gate()
    assert data_res["passed"], data_res["details"]
    baseline_res = matched_baseline_gate()
    assert baseline_res["passed"], baseline_res["details"]
    neg_res = negative_result_gate()
    assert neg_res["passed"], neg_res["details"]


def test_p2_run_all_generates_report(tmp_path: Path) -> None:
    report = run_all(tmp_path)
    assert report["status"] == "PASS"
    assert (tmp_path / "latest.json").is_file()
    assert (tmp_path / "REPORT.md").is_file()
    assert not Path("Stages/P2_REAL_DATA_PILOT_REPORT.md").resolve().samefile(tmp_path / "REPORT.md")
