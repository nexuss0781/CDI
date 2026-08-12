"""Public CDI tokenizer API backed by EthioBBPE.

`CDITokenizer` retains the historical embedding and tied-logit interface for
legacy CDI v2 code.  New v3 code should use `EthioBBPETokenizer` directly.
`CharacterTokenizer` remains an import alias only for source compatibility and
must not be used as a semantic description of the active tokenizer.
"""
from cdi.v3.tokenizer import CDITokenizer, CharacterTokenizer, EncodedText, EthioBBPETokenizer, TokenizerConfig

__all__ = ["CDITokenizer", "EthioBBPETokenizer", "CharacterTokenizer", "EncodedText", "TokenizerConfig"]
