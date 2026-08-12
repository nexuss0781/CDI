"""Zero-dependency CDI tokenizer compatibility module.

Stage D replaces the former external download dependency with the versioned,
pure-Python Unicode character tokenizer implemented in :mod:`cdi.v3.tokenizer`.
The legacy ``CDITokenizer`` name remains available for existing CDI v2 callers.
"""
from cdi.v3.tokenizer import CDITokenizer, CharacterTokenizer, EncodedText, TokenizerConfig

__all__ = ["CDITokenizer", "CharacterTokenizer", "EncodedText", "TokenizerConfig"]
