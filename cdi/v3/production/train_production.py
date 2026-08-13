"""Blocked legacy external-production training route.

The active CCT language-engine evidence is produced only by the governed,
matched pilot harness.  This module previously downloaded mutable external
corpora and trained across manifest train, validation, and test rows.  It is
now fail-closed until a separately approved production data contract exists.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .hf_ingest import BLOCKED_REASON, ProductionRouteBlockedError


def run_production_pipeline(
    config_path: str | Path = "benchmarks/configs/production_large.json",
    output_dir: str | Path = "results/production",
) -> Dict[str, Any]:
    """Reject the retired pipeline before any network, data, or training action."""
    del config_path, output_dir
    raise ProductionRouteBlockedError(
        "External production training is blocked because its prior implementation "
        "did not preserve manifest split isolation. " + BLOCKED_REASON
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="benchmarks/configs/production_large.json")
    parser.add_argument("--out", default="results/production")
    parser.parse_args()
    raise SystemExit(
        "ERROR: External production training is intentionally blocked. "
        "Consult Todo.md and ISSUES_TODO.md; the active work is CCT-G3.1."
    )


if __name__ == "__main__":
    _main()
