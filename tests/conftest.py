"""
Shared test fixtures for CDI v2.0 test suite.

v2.0: All fixtures use v2.0-compliant configs (dim B_0 >= embed_dim).
"""

import pytest
import torch

from cdi.config import CDIConfig
from cdi.core.manifold import CognitiveManifold
from cdi.core.cover import GoodCover
from cdi.core.sheaf import ObservationSheaf
from cdi.core.belief import BeliefComplex
from cdi.geometry.clifford import CliffordAlgebra
from cdi.geometry.connection import BeliefConnection
from cdi.geometry.dirac import DiracOperator
from cdi.operators.laplacian import BeliefLaplacian
from cdi.operators.hodge import HodgeDecomposition
from cdi.operators.green import GreenOperator
from cdi.operators.inference import InferenceOperator
from cdi.dynamics.heat_equation import HeatEquation
from cdi.engine import CDIEngine


@pytest.fixture
def tiny_config():
    """v2.0 tiny config — spec-compliant (B_0=32 >= embed=32)."""
    return CDIConfig.tiny()


@pytest.fixture
def small_config():
    """v2.0 small config."""
    return CDIConfig.small()


@pytest.fixture
def manifold(tiny_config):
    return CognitiveManifold(tiny_config)


@pytest.fixture
def cover(manifold, tiny_config):
    return GoodCover(manifold, tiny_config)


@pytest.fixture
def sheaf(tiny_config):
    return ObservationSheaf(tiny_config)


@pytest.fixture
def belief(tiny_config):
    return BeliefComplex(tiny_config)


@pytest.fixture
def clifford(tiny_config):
    return CliffordAlgebra(tiny_config)


@pytest.fixture
def connection(tiny_config, cover):
    return BeliefConnection(tiny_config, cover.edges)


@pytest.fixture
def dirac(manifold, clifford, connection, belief, cover, tiny_config):
    """v2.0: built without .detach() — live parameters."""
    d = DiracOperator(manifold, clifford, connection, belief, cover, tiny_config)
    d.build()
    return d


@pytest.fixture
def laplacian(dirac, belief, connection, tiny_config):
    """v2.0: built from live Dirac — differentiable."""
    lap = BeliefLaplacian(dirac, belief, connection, tiny_config)
    lap.build()
    return lap


@pytest.fixture
def hodge(laplacian):
    return HodgeDecomposition(laplacian)


@pytest.fixture
def green(laplacian):
    return GreenOperator(laplacian)


@pytest.fixture
def built_engine(tiny_config):
    """Fully built v2.0 CDI engine."""
    engine = CDIEngine(tiny_config)
    engine.build()
    return engine
