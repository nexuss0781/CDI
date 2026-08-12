"""Repository-root entry point for offline P1 training-system hardening verification."""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.p1_readiness import run_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="offline")
    parser.add_argument("--output-dir", default="results/p1")
    args = parser.parse_args()
    if args.config != "offline":
        raise ValueError("P1 supports only the frozen offline configuration.")
    report = run_all(Path(args.output_dir))
    print(f"P1 {report['status']}; offline_only={report['offline_only']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
