"""
CDI Topology Modules
====================

- §2.2  CechCohomology: Čech cochain complex and cohomology
- §12   SpectralSequence: Algorithm 12.3.1 — O(n log n) hypercohomology
- §11   SystemInvariants: Intelligence index, learning time, Chern character
"""

from .cech import CechCohomology
from .spectral_sequence import SpectralSequence
from .invariants import SystemInvariants

__all__ = ["CechCohomology", "SpectralSequence", "SystemInvariants"]
