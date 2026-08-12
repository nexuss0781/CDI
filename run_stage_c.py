"""Repository-root entry point for the complete Stage C gate."""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.stage_c import run_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="nano")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/stage_c")
    parser.add_argument("--stability-steps", type=int, default=10_000)
    args = parser.parse_args()
    if args.config != "nano":
        raise ValueError("Stage C currently supports only the CPU-safe nano configuration.")
    report = run_all(seed=args.seed, output_dir=Path(args.output_dir), stability_steps=args.stability_steps)
    print(f"Stage C {report['status']}; elapsed_seconds={report['elapsed_seconds']:.4f}")
    return 0 if report["status"] == "PASS" and report["nano_under_30_seconds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
