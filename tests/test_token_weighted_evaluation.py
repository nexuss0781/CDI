from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from cdi.v3.training import evaluate


class _UnequalTokenLossModel(nn.Module):
    def causal_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        marker = int(input_ids[0, 0])
        if marker == 1:
            return SimpleNamespace(loss=torch.tensor(1.0), token_count=1)
        return SimpleNamespace(loss=torch.tensor(3.0), token_count=3)


def test_evaluate_weights_causal_loss_by_active_token_count() -> None:
    model = _UnequalTokenLossModel()
    model.train()
    batches = [
        {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[True]])},
        {"input_ids": torch.tensor([[2]]), "attention_mask": torch.tensor([[True]])},
    ]
    result = evaluate(model, batches)
    assert result["token_count"] == 4
    assert result["loss"] == 2.5
    assert model.training
