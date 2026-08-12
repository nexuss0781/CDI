"""Repository-root entry point for the complete Stage E controlled study."""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.stage_e import run_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="nano")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-dir", default="results/stage_e")
    args = parser.parse_args()
    if args.config != "nano":
        raise ValueError("Stage E currently supports only the CPU-safe nano configuration.")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    report = run_all(seeds=seeds, output_dir=Path(args.output_dir), steps=args.steps)
    print(f"Stage E {report['status']}; elapsed_seconds={report['elapsed_seconds']:.4f}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
