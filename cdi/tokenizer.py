"""
CDI Tokenizer — Text ↔ Tensor Bridge
======================================

Wraps a HuggingFace tokenizer with learnable embeddings.
NO nn.Module.  Just raw tensors + the tokenizer.

The CDI engine processes numerical tensors on a manifold.
This module converts text → token IDs → embeddings (manifold observations)
and projects CDI outputs back → logits → text.

Architecture:
    text → tokenizer → token_ids → embedding_matrix[ids] → (n_points, embed_dim)
         → CDI Engine → (n_points, output_dim)
         → output @ embedding_matrix.T → (n_points, vocab_size) → argmax → text

Complexity: O(n) for encode/decode where n = sequence length.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class CDITokenizer:
    """HuggingFace tokenizer + learnable embeddings for CDI.

    No nn.Module.  All parameters are plain tensors with requires_grad.

    Parameters
    ----------
    tokenizer_name : str
        HuggingFace tokenizer identifier (e.g. "gpt2").
    embed_dim : int
        Embedding dimension (= CDI observation_dim = output_dim).
    max_len : int
        Maximum sequence length (= CDI n_points).
    dtype : torch.dtype
        Tensor dtype (float64 for CDI).
    """

    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        embed_dim: int = 64,
        max_len: int = 32,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        from transformers import AutoTokenizer

        self.hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Ensure pad token exists
        if self.hf_tokenizer.pad_token is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

        self.vocab_size = self.hf_tokenizer.vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.dtype = dtype
        self.pad_id = self.hf_tokenizer.pad_token_id

        # ── Learnable token embeddings: vocab_size × embed_dim ───
        # Xavier initialisation — no nn.Embedding
        scale = (2.0 / (self.vocab_size + embed_dim)) ** 0.5
        self.embedding = torch.randn(
            self.vocab_size, embed_dim, dtype=dtype
        ) * scale
        self.embedding.requires_grad_(True)

    # ──────────────────────────────────────────────────────────────
    # Encode: text → tensors
    # ──────────────────────────────────────────────────────────────

    def encode(self, text: str) -> torch.Tensor:
        """Tokenise text → token IDs tensor.

        Returns (max_len,) int64 tensor, padded/truncated.
        Complexity: O(n).
        """
        ids = self.hf_tokenizer.encode(
            text,
            max_length=self.max_len,
            truncation=True,
            padding=False,
        )
        # Pad to max_len
        if len(ids) < self.max_len:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long)

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Tokenise batch of texts → (batch, max_len) int64.

        Complexity: O(batch × n).
        """
        return torch.stack([self.encode(t) for t in texts])

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Token IDs → embedding vectors.

        token_ids : (...) int64
        Returns   : (..., embed_dim) float64

        Complexity: O(n).
        """
        return self.embedding[token_ids]

    def encode_and_embed(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """text → (token_ids, embeddings).

        Returns:
            token_ids  : (max_len,) int64
            embeddings : (max_len, embed_dim) float64
        """
        ids = self.encode(text)
        return ids, self.embed(ids)

    # ──────────────────────────────────────────────────────────────
    # Decode: tensors → text
    # ──────────────────────────────────────────────────────────────

    def to_logits(self, output: torch.Tensor) -> torch.Tensor:
        """Project CDI output to vocabulary logits.

        output : (..., embed_dim) — CDI output at each position.
        Returns: (..., vocab_size) — logits via weight tying.

        Uses embedding transpose (weight tying): logits = output @ E^T
        Complexity: O(n × vocab_size).
        """
        return output @ self.embedding.T

    def decode_ids(self, token_ids) -> str:
        """Token IDs → text string.

        Complexity: O(n).
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        # Filter pad tokens
        filtered = [t for t in token_ids if t != self.pad_id]
        return self.hf_tokenizer.decode(filtered, skip_special_tokens=True)

    def decode_logits(self, logits: torch.Tensor) -> str:
        """Logits → greedy decode → text.

        logits : (seq_len, vocab_size)
        Returns: decoded string.
        """
        ids = logits.argmax(dim=-1)
        return self.decode_ids(ids)

    # ──────────────────────────────────────────────────────────────
    # Parameters
    # ──────────────────────────────────────────────────────────────

    def get_parameters(self) -> list:
        """Learnable: embedding matrix."""
        return [self.embedding]
