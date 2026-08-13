"""Blocked legacy external-ingestion route.

The active CCT pipeline uses an explicit governed manifest in the matched pilot
harness.  This former helper fetched mutable external datasets, optionally
logged into Hugging Face, used ``trust_remote_code=True``, and assigned
provenance metadata in code.  It is retained only as a compatibility import;
it must not fetch or execute remote content until a separately reviewed source
registry and data-admission contract are approved.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict


class ProductionRouteBlockedError(RuntimeError):
    """Raised when a retired external-production route is invoked."""


BLOCKED_REASON = (
    "External production ingestion is blocked. It has no approved immutable "
    "source registry, governed split contract, or CCT authorization. Use the "
    "active CCT pilot harness and its governed manifest instead."
)


def ingest_wikitext_and_sciq(output_dir: str | Path = "data/production") -> Dict[str, Path]:
    """Reject the retired mutable external-ingestion route without side effects."""
    del output_dir
    raise ProductionRouteBlockedError(BLOCKED_REASON)
