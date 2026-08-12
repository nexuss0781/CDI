"""Repository-root entry point for P2 real-data pilot and scale-ladder execution."""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.p2_pilot import run_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pilot")
    parser.add_argument("--output-dir", default="results/p2")
    args = parser.parse_args()
    if args.config != "pilot":
        raise ValueError("P2 supports only the frozen pilot configuration.")
    report = run_all(Path(args.output_dir))
    print(f"P2 {report['status']}; gates_passed={len(report['gates'])}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
