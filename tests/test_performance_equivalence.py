from __future__ import annotations

import torch
import torch.nn.functional as F

from cdi.v3.language_model import DCSSLanguageModel
from cdi.v3.ssm import CohomodynamicState, StageCConfig
from cdi.v3.tokenizer import EthioBBPETokenizer, TokenizerConfig


def _model() -> tuple[DCSSLanguageModel, EthioBBPETokenizer]:
    tokenizer = EthioBBPETokenizer.from_pretrained(TokenizerConfig(max_chunk_length=16, embedding_dim=4))
    model = DCSSLanguageModel(
        tokenizer,
        StageCConfig.nano(seed=11, token_residual_enabled=True),
    )
    model.eval()
    return model, tokenizer


def _reference_forward(model: DCSSLanguageModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    embeddings = model.embedding(input_ids)
    current = model.ssm.initial_state(batch_shape=(input_ids.shape[0],), mode="zero")
    outputs = []
    for index in range(input_ids.shape[1]):
        source_embedding = embeddings[:, index]
        hidden, candidate = model.ssm.step(source_embedding, current)
        if model.token_residual is not None:
            residual = model.token_residual(source_embedding, ablated=model.config.token_residual_ablation)
            if model.residual_fusion is not None:
                hidden = model.residual_fusion(hidden, residual, ablated=model.config.residual_fusion_ablation)
            else:
                hidden = hidden + residual
        active = attention_mask[:, index]
        current = model._select_state(current, candidate, active)
        outputs.append(hidden * active.unsqueeze(-1).to(dtype=hidden.dtype))
    hidden_chunk = torch.stack(outputs, dim=1)
    return F.linear(hidden_chunk, model.embedding.weight, model.output_bias)


def test_dense_mask_fast_path_is_logit_equivalent() -> None:
    model, tokenizer = _model()
    ids = torch.tensor([[tokenizer.bos_id, 5, 6, 7, tokenizer.eos_id]], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    optimized = model.forward_chunk(ids, attention_mask=mask, return_state=False)
    reference = _reference_forward(model, ids, mask)
    torch.testing.assert_close(optimized, reference, rtol=1e-6, atol=1e-7)


def test_padded_mask_path_is_logit_equivalent() -> None:
    model, tokenizer = _model()
    ids = torch.tensor([[tokenizer.bos_id, 5, 6, tokenizer.pad_id, tokenizer.pad_id]], dtype=torch.long)
    mask = ids.ne(tokenizer.pad_id)
    optimized = model.forward_chunk(ids, attention_mask=mask, return_state=False)
    reference = _reference_forward(model, ids, mask)
    torch.testing.assert_close(optimized, reference, rtol=1e-6, atol=1e-7)


def test_matmul_contrast_projection_preserves_fixed_basis_features() -> None:
    model, _ = _model()
    state = CohomodynamicState(
        torch.randn(2, 4, 4),
        torch.randn(2, 4, 4),
        torch.randn(2, 4, 4),
    )
    reference_parts = []
    for name in ("fast", "middle", "harmonic"):
        band = state.by_name(name)
        mean = band.mean(dim=-2)
        contrast = torch.einsum("vi,...vw->...iw", model.ssm.cell.vertex_contrast_basis, band)
        reference_parts.extend((mean, contrast.reshape(*band.shape[:-2], -1)))
    reference = torch.cat(reference_parts, dim=-1)
    optimized = model.ssm.cell._readout_features(state)
    torch.testing.assert_close(optimized, reference, rtol=1e-6, atol=1e-7)
