"""
CDI Field Modules
=================

- §7.1  Superconnection: Quillen superconnection 𝔸 = D + δ + A
- §7.2  FieldEquations: 𝔸Ψ = 𝒥
- §10.2 GaugeTransformation: Gauge invariance and Noether currents
"""

from .superconnection import Superconnection
from .field_equations import FieldEquations
from .gauge import GaugeTransformation

__all__ = ["Superconnection", "FieldEquations", "GaugeTransformation"]
