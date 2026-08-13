from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from cdi.v3.production.hf_ingest import ProductionRouteBlockedError, ingest_wikitext_and_sciq
from cdi.v3.production.train_production import run_production_pipeline


ROOT = Path(__file__).resolve().parents[2]


def test_external_ingestion_is_fail_closed_without_side_effects(tmp_path: Path) -> None:
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ProductionRouteBlockedError, match="blocked"):
        ingest_wikitext_and_sciq(destination)
    assert not destination.exists()


def test_external_production_training_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ProductionRouteBlockedError, match="split isolation"):
        run_production_pipeline(output_dir=tmp_path / "must-not-exist")


def test_safe_shell_status_has_no_training_side_effects() -> None:
    result = subprocess.run(
        ["bash", "run.sh", "status"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "CCT safe entry point" in result.stdout
    assert "CCT-G3.1" in result.stdout
    assert "Launching GPU training" not in result.stdout


def test_safe_shell_rejects_retired_commands() -> None:
    result = subprocess.run(
        ["bash", "run.sh", "production"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "not an approved CCT command" in result.stderr
