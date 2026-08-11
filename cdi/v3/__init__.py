"""DCSS-CDI v3 sparse, factorized operator substrate.

Stage B contains no selective recurrence or language-model training path. Its
public objects are CPU-safe matrix-free building blocks for later approved work.
"""
from .cochain import SparseCochainMap
from .config import DCSSConfig, load_config
from .diagnostics import OperatorDiagnostics
from .incidence import SparseIncidence
from .laplacian import MatrixFreeLaplacian
from .reference import DenseReferenceOperators
from .topology import SparseTopology

__all__ = [
    "DCSSConfig",
    "DenseReferenceOperators",
    "MatrixFreeLaplacian",
    "OperatorDiagnostics",
    "SparseCochainMap",
    "SparseIncidence",
    "SparseTopology",
    "load_config",
]
