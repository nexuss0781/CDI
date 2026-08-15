from __future__ import annotations

import torch
import torch.nn.functional as F

from benchmarks.ethiobbpe_synaxarium_pilot import build_model
from cdi.v3.tokenizer import EthioBBPETokenizer, TokenizerConfig


class CompiledDenseCDI(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor):
        return self.model.forward_chunk_active(
            input_ids,
            return_state=False,
            runtime_guard_mode="deferred",
        )


def check_metrics(metrics: tuple[torch.Tensor, torch.Tensor, torch.Tensor], model: torch.nn.Module) -> None:
    spectral, energy, state_norm = metrics
    cell = model.ssm.cell
    assert not bool(spectral.detach().item())
    assert not bool((energy > cell.stage_b_config.energy_limit).detach().item())
    assert not bool((state_norm > cell.config.state_norm_bound).detach().item())


torch.set_num_threads(1)
torch.manual_seed(23)
tokenizer = EthioBBPETokenizer.from_pretrained(TokenizerConfig(max_chunk_length=16, embedding_dim=4))
eager = build_model("dcss_residual_cdi", tokenizer, seed=23).train()
compiled_model = build_model("dcss_residual_cdi", tokenizer, seed=23).train()
compiled_model.load_state_dict(eager.state_dict())
compiled = torch.compile(CompiledDenseCDI(compiled_model), mode="reduce-overhead", dynamic=False, fullgraph=True).train()
ids = torch.randint(1, tokenizer.vocab_size, (2, 16), dtype=torch.long)
targets = ids[:, 1:]

# Compile and execute one complete training step.
compiled_logits, _, compiled_metrics = compiled(ids[:, :-1])
check_metrics(compiled_metrics, compiled_model)
compiled_loss = F.cross_entropy(compiled_logits.reshape(-1, tokenizer.vocab_size), targets.reshape(-1))
compiled_loss.backward()
compiled_grads = [parameter.grad.detach().clone() if parameter.grad is not None else None for parameter in compiled_model.parameters()]
compiled_optimizer = torch.optim.AdamW(compiled.parameters(), lr=0.001)
compiled_optimizer.step()

# Compare against the exact eager deferred path from identical initial weights.
eager_logits, _, eager_metrics = eager.forward_chunk_active(ids[:, :-1], return_state=False, runtime_guard_mode="deferred")
check_metrics(eager_metrics, eager)
eager_loss = F.cross_entropy(eager_logits.reshape(-1, tokenizer.vocab_size), targets.reshape(-1))
eager_loss.backward()
eager_grads = [parameter.grad for parameter in eager.parameters()]
logit_error = float((compiled_logits.detach() - eager_logits.detach()).abs().max())
loss_error = float((compiled_loss.detach() - eager_loss.detach()).abs())
gradient_error = max(
    float((left - right).abs().max())
    for left, right in zip(compiled_grads, eager_grads)
    if left is not None and right is not None
)
print({"logit_error": logit_error, "loss_error": loss_error, "gradient_error": gradient_error, "compiled_loss": float(compiled_loss.detach()), "eager_loss": float(eager_loss.detach())})
assert logit_error < 1e-5
assert loss_error < 1e-5
assert gradient_error < 1e-4
