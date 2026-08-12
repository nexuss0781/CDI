"""Repository-root entry point for the complete Stage D gate."""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.stage_d import run_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="nano")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/stage_d")
    args = parser.parse_args()
    if args.config != "nano":
        raise ValueError("Stage D currently supports only the CPU-safe nano configuration.")
    report = run_all(seed=args.seed, output_dir=Path(args.output_dir))
    print(f"Stage D {report['status']}; elapsed_seconds={report['elapsed_seconds']:.4f}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
