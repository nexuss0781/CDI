"""One-command DCSS-CDI Stage B sparse-substrate gate runner."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from benchmarks.stage_b import run_all


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="nano")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/stage_b")
    parser.add_argument("--sizes", default="4,8,16,32,64")
    arguments = parser.parse_args(argv)
    sizes = [int(value) for value in arguments.sizes.split(",") if value.strip()]
    report = run_all(arguments.config, arguments.seed, sizes, Path(arguments.output_dir))
    print(f"Stage B {report['status']}; elapsed_seconds={report['elapsed_seconds']:.4f}")
    return 0 if report["status"] == "PASS" and report["nano_under_30_seconds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
