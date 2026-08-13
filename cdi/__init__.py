"""Legacy CDI v2 compatibility API.

The active CCT language-engine implementation is ``cdi.v3``.  This top-level
namespace remains available only for the earlier dense v2 mathematical
prototype and must not be used as CCT training or benchmark evidence.
"""

from cdi.config import CDIConfig
from cdi.engine import CDIEngine
from cdi.tokenizer import CDITokenizer

CCT_ACTIVE_NAMESPACE = "cdi.v3"

__version__ = "2.0.0-legacy"
__all__ = ["CDIConfig", "CDIEngine", "CDITokenizer", "CCT_ACTIVE_NAMESPACE"]
