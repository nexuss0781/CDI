"""Repository-root entry point for the bounded Stage F dry-run diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.stage_f import run_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="nano")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--output-dir", default="results/stage_f")
    args = parser.parse_args()
    if args.config != "nano":
        raise ValueError("Stage F supports only the frozen CPU-safe nano configuration.")
    if args.mode != "dry_run":
        raise ValueError("Stage F capability evaluation is dry-run only.")
    report = run_all(Path(args.output_dir))
    print(f"Stage F {report['status']}; external_side_effects_enabled={report['external_side_effects_enabled']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
