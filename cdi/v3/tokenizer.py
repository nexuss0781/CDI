"""EthioBBPE-backed tokenizer contracts for CDI language-model paths.

The active CDI tokenizer is a snapshot of the published EthioBBPE model.  Every
training run, checkpoint, and inference engine carries the exact tokenizer JSON
used to produce token IDs.  This prevents the historical failure mode where data
were encoded by EthioBBPE but embedded through a different character vocabulary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

import torch
from ethiobbpe import EthioBBPETokenizer as _BackendEthioBBPETokenizer
from tokenizers import Tokenizer as _HFTokenizer


@dataclass(frozen=True)
class TokenizerConfig:
    """Versioned CDI contract for a locally snapshotted EthioBBPE tokenizer."""

    format: str = "dcss-cdi-ethiobbpe-tokenizer-v1"
    model_id: str = "Nexuss0781/Ethio-BBPE"
    max_chunk_length: int = 8
    embedding_dim: int = 4
    normalization: str = "artifact_defined"
    whitespace_policy: str = "artifact_defined"
    special_tokens: Tuple[str, ...] = ("<pad>", "<unk>", "<s>", "</s>", "<mask>")

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

    def as_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EncodedText:
    """Explicit tokenization result, including an intentional truncation choice."""

    ids: Tuple[int, ...]
    truncated: bool
    normalized_text: str


def _canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "fingerprint"}
    encoded = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _snapshot_backend(tokenizer: _BackendEthioBBPETokenizer) -> str:
    """Return the exact Hugging Face tokenizer JSON used by the backend."""

    handle = NamedTemporaryFile(prefix="cdi-ethiobbpe-", suffix=".json", delete=False)
    path = Path(handle.name)
    handle.close()
    try:
        tokenizer.save(path)
        return path.read_text(encoding="utf-8")
    finally:
        path.unlink(missing_ok=True)


class EthioBBPETokenizer:
    """CDI adapter around the published EthioBBPE Byte-Level BPE tokenizer.

    The adapter owns a trainable-tokenizer contract rather than a trainable
    embedding table.  Language models create the embedding table from
    :attr:`vocab_size`; every ID is range-checked before lookup, never clamped.
    """

    _SPECIAL_ROLES = {
        "pad": "<pad>",
        "unk": "<unk>",
        "bos": "<s>",
        "eos": "</s>",
        "doc": "<mask>",
    }

    def __init__(
        self,
        backend: _BackendEthioBBPETokenizer,
        config: TokenizerConfig | None = None,
        *,
        tokenizer_json: str | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or TokenizerConfig()
        if self.config.max_chunk_length < 2:
            raise ValueError("max_chunk_length must be at least two.")
        if self.config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.token_to_id = dict(backend.get_vocab())
        self.vocabulary = tuple(token for token, _ in sorted(self.token_to_id.items(), key=lambda item: item[1]))
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self._vocab_size = int(backend.get_vocab_size())
        if self._vocab_size <= 0 or len(self.token_to_id) != self._vocab_size:
            raise ValueError("EthioBBPE vocabulary is incomplete or has an invalid size.")
        if set(self.id_to_token) != set(range(self._vocab_size)):
            raise ValueError("EthioBBPE vocabulary IDs must be contiguous from zero.")
        self._special_ids = {
            role: self._required_token_id(token)
            for role, token in self._SPECIAL_ROLES.items()
        }
        self._tokenizer_json = tokenizer_json or _snapshot_backend(backend)

    @classmethod
    def from_pretrained(
        cls,
        config: TokenizerConfig | None = None,
        *,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
    ) -> "EthioBBPETokenizer":
        resolved = config or TokenizerConfig()
        backend = _BackendEthioBBPETokenizer.from_pretrained(
            resolved.model_id,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            force_download=force_download,
        )
        return cls(backend, resolved)

    @classmethod
    def from_texts(
        cls,
        _texts: Iterable[str],
        config: TokenizerConfig | None = None,
    ) -> "EthioBBPETokenizer":
        """Compatibility constructor retaining the fixed published tokenizer.

        CDI no longer creates a corpus-derived character vocabulary.  The texts
        argument is intentionally unused because model token semantics must stay
        stable across train, validation, and inference.
        """

        return cls.from_pretrained(config=config)

    @classmethod
    def from_artifact(cls, payload: Mapping[str, Any]) -> "EthioBBPETokenizer":
        if payload.get("format") != TokenizerConfig().format:
            raise ValueError("Unsupported EthioBBPE tokenizer artifact format.")
        if payload.get("fingerprint") != _canonical_fingerprint(payload):
            raise ValueError("EthioBBPE tokenizer artifact fingerprint does not match its contents.")
        if not isinstance(payload.get("tokenizer_json"), str):
            raise ValueError("EthioBBPE tokenizer artifact is missing its tokenizer JSON snapshot.")
        config_data = dict(payload.get("config", {}))
        config_data["special_tokens"] = tuple(config_data.get("special_tokens", ()))
        config = TokenizerConfig(**config_data)
        tokenizer_json = str(payload["tokenizer_json"])
        try:
            raw = _HFTokenizer.from_str(tokenizer_json)
        except Exception as exc:
            raise ValueError(f"EthioBBPE tokenizer snapshot is invalid: {exc}") from exc
        backend = _BackendEthioBBPETokenizer(raw, {"vocab_size": len(raw.get_vocab()), "model_name": config.model_id})
        return cls(backend, config, tokenizer_json=tokenizer_json)

    @classmethod
    def load(cls, path: str | Path) -> "EthioBBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_artifact(payload)

    def _required_token_id(self, token: str) -> int:
        try:
            return int(self.token_to_id[token])
        except KeyError as exc:
            raise ValueError(f"EthioBBPE artifact does not provide required special token {token!r}.") from exc

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_id(self) -> int:
        return self._special_ids["pad"]

    @property
    def unk_id(self) -> int:
        return self._special_ids["unk"]

    @property
    def bos_id(self) -> int:
        return self._special_ids["bos"]

    @property
    def eos_id(self) -> int:
        return self._special_ids["eos"]

    @property
    def doc_id(self) -> int:
        return self._special_ids["doc"]

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("EthioBBPETokenizer accepts str input only.")
        return text

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncate: bool = False,
    ) -> EncodedText:
        normalized = self.normalize(text)
        base_ids = list(self.backend.encode(normalized, add_special_tokens=False).ids)
        ids = [self.bos_id, *base_ids, self.eos_id] if add_special_tokens else base_ids
        truncated = False
        if max_length is not None and len(ids) > max_length:
            if not truncate:
                raise ValueError("Encoding exceeds max_length; pass truncate=True to permit explicit truncation.")
            ids = ids[:max_length]
            if add_special_tokens and ids:
                ids[-1] = self.eos_id
            truncated = True
        self.assert_ids_in_range(ids)
        return EncodedText(tuple(ids), truncated, normalized)

    def encode_ids(self, text: str, **kwargs: Any) -> List[int]:
        return list(self.encode(text, **kwargs).ids)

    def encode_batch(self, texts: Sequence[str], **kwargs: Any) -> List[EncodedText]:
        return [self.encode(text, **kwargs) for text in texts]

    def pad(
        self,
        encoded: Sequence[EncodedText | Sequence[int]],
        max_length: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        rows = [list(value.ids) if isinstance(value, EncodedText) else list(value) for value in encoded]
        if not rows:
            raise ValueError("pad requires at least one encoded sequence.")
        maximum = max_length or max(len(row) for row in rows)
        if maximum <= 0:
            raise ValueError("max_length must be positive.")
        ids, masks = [], []
        for row in rows:
            if len(row) > maximum:
                raise ValueError("pad received an overlength row; truncation must be explicit in encode().")
            self.assert_ids_in_range(row)
            padding = maximum - len(row)
            ids.append(row + [self.pad_id] * padding)
            masks.append([True] * len(row) + [False] * padding)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.bool), 0

    def decode(self, token_ids: Sequence[int] | torch.Tensor, *, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().reshape(-1).tolist()
        values = [int(token_id) for token_id in token_ids]
        self.assert_ids_in_range(values)
        if skip_special_tokens:
            hidden = {self.pad_id, self.bos_id, self.eos_id, self.doc_id}
            values = [token_id for token_id in values if token_id not in hidden]
        return self.backend.decode(values, skip_special_tokens=skip_special_tokens)

    def assert_ids_in_range(self, token_ids: Sequence[int] | torch.Tensor) -> None:
        values = token_ids.detach().cpu().reshape(-1).tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        if not values:
            return
        lowest, highest = min(int(value) for value in values), max(int(value) for value in values)
        if lowest < 0 or highest >= self.vocab_size:
            raise ValueError(
                f"Token IDs must lie in [0, {self.vocab_size - 1}] for the saved EthioBBPE artifact; "
                f"received range [{lowest}, {highest}]."
            )

    def artifact(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "format": self.config.format,
            "config": dict(self.config.as_dict()),
            "model_id": self.config.model_id,
            "vocab_size": self.vocab_size,
            "special_token_ids": dict(self._special_ids),
            "vocabulary_fingerprint": sha256(
                json.dumps(self.token_to_id, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "tokenizer_json": self._tokenizer_json,
        }
        payload["fingerprint"] = _canonical_fingerprint(payload)
        return payload

    @property
    def fingerprint(self) -> str:
        return str(self.artifact()["fingerprint"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.artifact(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def assert_fingerprint(self, expected: str) -> None:
        if self.fingerprint != expected:
            raise ValueError(f"Tokenizer fingerprint mismatch: expected {expected}, got {self.fingerprint}.")


class CDITokenizer:
    """Legacy CDI v2 adapter backed by the same EthioBBPE artifact as v3.

    The adapter exposes the historical embedding and tied-logit API, while
    enforcing range checks to prevent token-ID clamping and data corruption.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        max_len: int = 32,
        dtype: torch.dtype = torch.float32,
        tokenizer: EthioBBPETokenizer | None = None,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        config = TokenizerConfig(max_chunk_length=max_len, embedding_dim=embed_dim)
        self.tokenizer = tokenizer or EthioBBPETokenizer.from_pretrained(config, cache_dir=cache_dir)
        if self.tokenizer.config.embedding_dim != embed_dim or self.tokenizer.config.max_chunk_length != max_len:
            raise ValueError("Provided EthioBBPE tokenizer configuration does not match CDITokenizer dimensions.")
        self.vocab_size = self.tokenizer.vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.dtype = dtype
        self.pad_id = self.tokenizer.pad_id
        self.unk_id = self.tokenizer.unk_id
        self.eos_id = self.tokenizer.eos_id
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
        token_ids = token_ids.to(dtype=torch.long)
        self.tokenizer.assert_ids_in_range(token_ids)
        return self.embedding[token_ids]

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
        return f"CDITokenizer(vocab={self.vocab_size}, embed={self.embed_dim}, max_len={self.max_len}, backend=EthioBBPE)"


# Backwards-compatible import only.  New code must use EthioBBPETokenizer.
CharacterTokenizer = EthioBBPETokenizer
