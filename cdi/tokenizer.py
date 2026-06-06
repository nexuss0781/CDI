"""
CDI Tokenizer — Powered by EthioBBPE
======================================

Uses the EthioBBPE tokenizer (Nexuss0781/Ethio-BBPE) — a production-ready
Byte Pair Encoding tokenizer with 16,000 tokens.

Advantages over GPT-2 tokenizer for CDI:
    - 16K vocab (vs 50K) → 3× smaller embedding matrix
    - Supports English + Amharic + Ge'ez
    - 100% reconstruction accuracy
    - Auto-downloads from HuggingFace Hub
    - Built by the CDI author — full control

Architecture:
    text → EthioBBPE → token_ids → embedding[ids] → (n_points, embed_dim)
         → CDI Engine (manifold, belief, Dirac, heat equation, Hodge)
         → (n_points, embed_dim)
         → output @ embedding.T → logits → next token prediction

Complexity: O(n) for encode/decode where n = sequence length.
No nn.Module.  All parameters are plain tensors.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class CDITokenizer:
    """EthioBBPE tokenizer + learnable embeddings for CDI.

    No nn.Module.  All parameters are plain tensors with requires_grad.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension (= CDI observation_dim = output_dim).
    max_len : int
        Maximum sequence length (= CDI n_points).
    dtype : torch.dtype
        Tensor dtype (float64 for CDI).
    """

    def __init__(
        self,
        embed_dim: int = 64,
        max_len: int = 32,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        from ethiobbpe import EthioBBPETokenizer

        self.hf_tokenizer = EthioBBPETokenizer.from_pretrained()
        self.vocab_size = self.hf_tokenizer.get_vocab_size()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.dtype = dtype

        # Find pad token ID
        vocab = self.hf_tokenizer.get_vocab()
        self.pad_id = vocab.get("[PAD]", 0)
        self.unk_id = vocab.get("[UNK]", 1)
        self.eos_id = vocab.get("[SEP]", vocab.get("[EOS]", self.pad_id))

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
        encoded = self.hf_tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_len,
        )
        ids = list(encoded.ids)

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
        # Clamp IDs to valid range
        safe_ids = token_ids.clamp(0, self.vocab_size - 1)
        return self.embedding[safe_ids]

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

    def __repr__(self) -> str:
        return (f"CDITokenizer(vocab={self.vocab_size}, "
                f"embed={self.embed_dim}, max_len={self.max_len}, "
                f"backend=EthioBBPE)")
