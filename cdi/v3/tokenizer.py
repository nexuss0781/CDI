"""Pure-Python tokenizer and versioned vocabulary artifact for Stage D.

The tokenizer deliberately avoids external model downloads and dependencies. It
uses Unicode NFC normalization, preserves whitespace exactly, and represents
unknown characters with a deterministic fallback token.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import unicodedata

import torch


@dataclass(frozen=True)
class TokenizerConfig:
    """Versioned behavior contract for :class:`CharacterTokenizer`."""

    format: str = "dcss-cdi-character-tokenizer-v1"
    normalization: str = "NFC"
    whitespace_policy: str = "preserve"
    byte_policy: str = "unicode_character_with_unk_fallback"
    max_chunk_length: int = 8
    embedding_dim: int = 4
    special_tokens: Tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>", "<doc>")

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def bos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    @property
    def doc_id(self) -> int:
        return 4

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EncodedText:
    """Explicit encoding result, including any requested truncation decision."""

    ids: Tuple[int, ...]
    truncated: bool
    normalized_text: str


class CharacterTokenizer:
    """Deterministic pure-Python character tokenizer with artifact fingerprinting."""

    def __init__(self, vocabulary: Sequence[str], config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        if tuple(vocabulary[: len(self.config.special_tokens)]) != self.config.special_tokens:
            raise ValueError("Vocabulary must begin with the configured ordered special tokens.")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Tokenizer vocabulary contains duplicate entries.")
        self.vocabulary = tuple(vocabulary)
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        self.id_to_token = {index: token for index, token in enumerate(self.vocabulary)}

    @classmethod
    def from_texts(cls, texts: Iterable[str], config: TokenizerConfig | None = None) -> "CharacterTokenizer":
        config = config or TokenizerConfig()
        observed = set()
        for text in texts:
            normalized = cls._normalize_static(text, config)
            observed.update(normalized)
        specials = set(config.special_tokens)
        vocabulary = list(config.special_tokens) + sorted(character for character in observed if character not in specials)
        return cls(vocabulary, config)

    @classmethod
    def default(cls, config: TokenizerConfig | None = None) -> "CharacterTokenizer":
        """Compatibility tokenizer with a stable printable-Unicode seed vocabulary."""
        config = config or TokenizerConfig()
        printable = [chr(value) for value in range(32, 127)] + ["\n", "\t"]
        return cls(list(config.special_tokens) + printable, config)

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @property
    def pad_id(self) -> int:
        return self.config.pad_id

    @property
    def unk_id(self) -> int:
        return self.config.unk_id

    @property
    def bos_id(self) -> int:
        return self.config.bos_id

    @property
    def eos_id(self) -> int:
        return self.config.eos_id

    @property
    def doc_id(self) -> int:
        return self.config.doc_id

    @staticmethod
    def _normalize_static(text: str, config: TokenizerConfig) -> str:
        if not isinstance(text, str):
            raise TypeError("CharacterTokenizer accepts str input only.")
        normalized = unicodedata.normalize(config.normalization, text)
        if config.whitespace_policy != "preserve":
            raise ValueError(f"Unsupported whitespace policy: {config.whitespace_policy}")
        return normalized

    def normalize(self, text: str) -> str:
        return self._normalize_static(text, self.config)

    def encode(self, text: str, *, add_special_tokens: bool = True, max_length: int | None = None, truncate: bool = False) -> EncodedText:
        normalized = self.normalize(text)
        ids: List[int] = [self.token_to_id.get(character, self.unk_id) for character in normalized]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        truncated = False
        if max_length is not None and len(ids) > max_length:
            if not truncate:
                raise ValueError("Encoding exceeds max_length; pass truncate=True to permit explicit truncation.")
            ids = ids[:max_length]
            if add_special_tokens and ids:
                ids[-1] = self.eos_id
            truncated = True
        return EncodedText(tuple(ids), truncated, normalized)

    def encode_ids(self, text: str, **kwargs: Any) -> List[int]:
        return list(self.encode(text, **kwargs).ids)

    def encode_batch(self, texts: Sequence[str], **kwargs: Any) -> List[EncodedText]:
        return [self.encode(text, **kwargs) for text in texts]

    def pad(self, encoded: Sequence[EncodedText | Sequence[int]], max_length: int | None = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Pad explicit encodings and return IDs, boolean attention mask, truncation count."""
        rows = [list(value.ids) if isinstance(value, EncodedText) else list(value) for value in encoded]
        if not rows:
            raise ValueError("pad requires at least one encoded sequence.")
        maximum = max_length or max(len(row) for row in rows)
        if maximum <= 0:
            raise ValueError("max_length must be positive.")
        ids, masks, truncations = [], [], 0
        for row in rows:
            if len(row) > maximum:
                raise ValueError("pad received an overlength row; truncation must be explicit in encode().")
            padding = maximum - len(row)
            ids.append(row + [self.pad_id] * padding)
            masks.append([True] * len(row) + [False] * padding)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.bool), truncations

    def decode(self, token_ids: Sequence[int] | torch.Tensor, *, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().reshape(-1).tolist()
        skipped_special_ids = {self.pad_id, self.bos_id, self.eos_id, self.doc_id}
        output: List[str] = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), "<unk>")
            # Unknown is a visible fallback character, not a silently removed
            # control token; only padding and boundary controls are skipped.
            if skip_special_tokens and int(token_id) in skipped_special_ids:
                continue
            output.append("�" if token == "<unk>" else token)
        return "".join(output)

    def artifact(self) -> Dict[str, Any]:
        payload = {
            "format": self.config.format,
            "config": self.config.as_dict(),
            "vocabulary": list(self.vocabulary),
        }
        payload["fingerprint"] = self._fingerprint_payload(payload)
        return payload

    @staticmethod
    def _fingerprint_payload(payload: Mapping[str, Any]) -> str:
        canonical = {key: value for key, value in payload.items() if key != "fingerprint"}
        return sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self.artifact()["fingerprint"]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.artifact(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != TokenizerConfig().format:
            raise ValueError("Unsupported tokenizer artifact format.")
        if payload.get("fingerprint") != cls._fingerprint_payload(payload):
            raise ValueError("Tokenizer artifact fingerprint does not match its contents.")
        config_data = dict(payload["config"])
        config_data["special_tokens"] = tuple(config_data["special_tokens"])
        return cls(payload["vocabulary"], TokenizerConfig(**config_data))

    def assert_fingerprint(self, expected: str) -> None:
        if self.fingerprint != expected:
            raise ValueError(f"Tokenizer fingerprint mismatch: expected {expected}, got {self.fingerprint}.")


class CDITokenizer:
    """Legacy-compatible pure-Python tokenizer wrapper with tied tensor embeddings.

    New Stage D code uses :class:`CharacterTokenizer` directly. This wrapper
    keeps the historical `encode`, `embed`, `to_logits`, and `decode_ids`
    methods without importing EthioBBPE or downloading external artifacts.
    """

    def __init__(self, embed_dim: int = 64, max_len: int = 32, dtype: torch.dtype = torch.float32, tokenizer: CharacterTokenizer | None = None) -> None:
        config = TokenizerConfig(max_chunk_length=max_len, embedding_dim=embed_dim)
        self.tokenizer = tokenizer or CharacterTokenizer.default(config)
        self.vocab_size = self.tokenizer.vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.dtype = dtype
        self.pad_id, self.unk_id, self.eos_id = self.tokenizer.pad_id, self.tokenizer.unk_id, self.tokenizer.eos_id
        scale = (2.0 / (self.vocab_size + embed_dim)) ** 0.5
        self.embedding = (torch.randn(self.vocab_size, embed_dim, dtype=dtype) * scale).requires_grad_(True)

    def encode(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer.encode(text, max_length=self.max_len, truncate=True)
        ids, _, _ = self.tokenizer.pad([encoded], max_length=self.max_len)
        return ids[0]

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        encoded = [self.tokenizer.encode(text, max_length=self.max_len, truncate=True) for text in texts]
        ids, _, _ = self.tokenizer.pad(encoded, max_length=self.max_len)
        return ids

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids.clamp(0, self.vocab_size - 1)]

    def encode_and_embed(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.encode(text)
        return ids, self.embed(ids)

    def to_logits(self, output: torch.Tensor) -> torch.Tensor:
        return output @ self.embedding.T

    def decode_ids(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)

    def decode_logits(self, logits: torch.Tensor) -> str:
        return self.decode_ids(logits.argmax(dim=-1))

    def get_parameters(self) -> list[torch.Tensor]:
        return [self.embedding]

    def __repr__(self) -> str:
        return f"CDITokenizer(vocab={self.vocab_size}, embed={self.embed_dim}, max_len={self.max_len}, backend=pure_python_character)"
