from __future__ import annotations

import torch
import torch.nn.functional as F

from benchmarks.ethiobbpe_synaxarium_pilot import build_model
from cdi.v3.tokenizer import EthioBBPETokenizer, TokenizerConfig


def original(cell, x):
    gates = tuple(cell.bands[name].gate for name in ("fast", "middle", "harmonic"))
    forcing = torch.tanh(F.linear(x, torch.cat(tuple(g.forcing_projection.weight for g in gates), 0), torch.cat(tuple(g.forcing_projection.bias for g in gates), 0))).reshape(*x.shape[:-1], 3, cell.config.band_width)
    input_gate = torch.sigmoid(F.linear(x, torch.cat(tuple(g.input_gate_projection.weight for g in gates), 0), torch.cat(tuple(g.input_gate_projection.bias for g in gates), 0))).reshape(*x.shape[:-1], 3, cell.config.band_width // 2)
    transport_gate = torch.sigmoid(F.linear(x, torch.cat(tuple(g.transport_projection.weight for g in gates), 0), torch.cat(tuple(g.transport_projection.bias for g in gates), 0))).reshape(*x.shape[:-1], 3, cell.config.band_width // 2)
    offsets = (torch.tanh(F.linear(x, torch.cat(tuple(g.timescale_projection.weight for g in gates), 0), torch.cat(tuple(g.timescale_projection.bias for g in gates), 0))) * cell.config.max_log_timescale_offset).reshape(*x.shape[:-1], 3, cell.config.band_width)
    geometry = torch.sigmoid(F.linear(x, torch.cat(tuple(g.geometry_projection.weight for g in gates), 0), torch.cat(tuple(g.geometry_projection.bias for g in gates), 0))).reshape(*x.shape[:-1], 3)
    return forcing * input_gate.repeat_interleave(2, -1), input_gate, transport_gate, offsets, geometry


torch.manual_seed(11)
tokenizer = EthioBBPETokenizer.from_pretrained(TokenizerConfig(max_chunk_length=16, embedding_dim=4))
model = build_model("dcss_residual_cdi", tokenizer, seed=11)
x = torch.randn(2, 7, 4)
old = original(model.ssm.cell, x)
new = model.ssm.cell.fused_gate_tensors(x)
errors = [float((a - b).abs().max()) for a, b in zip(old, new)]
loss_old = sum(value.square().sum() for value in old)
loss_new = sum(value.square().sum() for value in new)
old_grads = torch.autograd.grad(loss_old, tuple(model.ssm.cell.parameters()), allow_unused=True, retain_graph=True)
new_grads = torch.autograd.grad(loss_new, tuple(model.ssm.cell.parameters()), allow_unused=True)
grad_error = max(float((a - b).abs().max()) for a, b in zip(old_grads, new_grads) if a is not None and b is not None)
print({"max_output_errors": errors, "max_gradient_error": grad_error})
assert max(errors) < 1e-6
assert grad_error < 1e-6
