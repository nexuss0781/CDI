#!/usr/bin/env python3
"""Record the runtime contract for a CCT-G0 readiness run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys

import ethiobbpe
import torch


def git_value(*arguments: str) -> str:
    try:
        return subprocess.check_output(["git", *arguments], text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    payload = {
        "format": "cct-g0-environment-v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "working_tree": git_value("status", "--short"),
        },
        "runtime": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "operating_system": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "default_dtype": str(torch.get_default_dtype()),
        },
        "packages": {
            "EthioBBPE": package_version("EthioBBPE"),
            "datasets": package_version("datasets"),
            "numpy": package_version("numpy"),
            "pytest": package_version("pytest"),
            "scipy": package_version("scipy"),
            "torch": package_version("torch"),
            "tokenizers": package_version("tokenizers"),
        },
        "imports": {
            "ethiobbpe_module": str(Path(ethiobbpe.__file__).resolve()),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
