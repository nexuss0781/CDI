from __future__ import annotations

import torch
from torch._higher_order_ops import scan

from benchmarks.ethiobbpe_synaxarium_pilot import build_model
from cdi.v3.tokenizer import EthioBBPETokenizer, TokenizerConfig


def scan_step(carry, xs):
    forcing, input_gate, transport_gate, offsets, geometry_gate, log_tau_base, rotation_bias, input_injection, geometry_operator = xs
    band_width = 4
    dt = 0.1
    rotation_limit = 1.0
    geometry_step_cap = 0.02
    max_geometry_edge_weight = 2.0
    n_vertices = 4
    band_lower = torch.tensor([-1.3862944, 0.6931472, 2.7725887], dtype=carry.dtype, device=carry.device).view(1, -1, 1)
    band_upper = torch.tensor([0.0, 2.0794415, 4.1588831], dtype=carry.dtype, device=carry.device).view(1, -1, 1)
    log_tau = (log_tau_base + offsets).clamp(min=band_lower, max=band_upper)
    tau = torch.exp(log_tau)
    dissipation = (0.05 + input_gate) / tau[..., ::2]
    omega = (2.0 * transport_gate - 1.0 + rotation_bias) * rotation_limit
    pairs = carry.reshape(*carry.shape[:-1], band_width // 2, 2)
    angle = (omega * dt).unsqueeze(-2)
    decay = torch.exp((-dissipation * dt).unsqueeze(-2))
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotated_first = cosine * pairs[..., 0] - sine * pairs[..., 1]
    rotated_second = sine * pairs[..., 0] + cosine * pairs[..., 1]
    homogeneous = torch.stack((rotated_first, rotated_second), dim=-1) * decay.unsqueeze(-1)
    band_state = homogeneous.reshape_as(carry) + dt * forcing.unsqueeze(-2) * input_injection
    correction = torch.matmul(geometry_operator, band_state)
    alpha = (geometry_step_cap * geometry_gate).unsqueeze(-1).unsqueeze(-1)
    value = band_state - alpha * correction
    return value, value.clone()


torch.manual_seed(11)
tokenizer = EthioBBPETokenizer.from_pretrained(TokenizerConfig(max_chunk_length=16, embedding_dim=4))
model = build_model("dcss_residual_cdi", tokenizer, seed=11)
cell = model.ssm.cell
print({"dt": cell.config.dt, "rotation_limit": cell.config.rotation_limit, "geometry_step_cap": cell.config.geometry_step_cap, "max_geometry_edge_weight": cell.config.max_geometry_edge_weight, "tau_lower": cell.band_log_tau_lower, "tau_upper": cell.band_log_tau_upper})
x = torch.randn(2, 16, 4)
forcing, input_gate, transport_gate, offsets, geometry = cell.fused_gate_tensors(x)
kernels = cell.fused_kernel_tensors()
operator = cell.geometry.operator(dtype=x.dtype, device=x.device)
state = torch.stack(cell.initial_state(batch_shape=(2,), mode="zero").tensors(), dim=-3)
xs = (
    forcing.transpose(0, 1), input_gate.transpose(0, 1), transport_gate.transpose(0, 1), offsets.transpose(0, 1), geometry.transpose(0, 1),
    *(kernel.unsqueeze(0).expand(x.shape[1], *kernel.shape) for kernel in kernels),
    operator.unsqueeze(0).expand(x.shape[1], *operator.shape),
)
final, trajectory = scan(scan_step, state, xs)
current = state
reference = []
for index in range(x.shape[1]):
    _, current = cell.step_fused_stacked(forcing[:, index], input_gate[:, index], transport_gate[:, index], offsets[:, index], geometry[:, index], current, runtime_guard_mode="disabled", kernel_tensors=kernels, geometry_operator=operator, return_output=False)
    reference.append(current)
reference = torch.stack(reference, dim=0)
scan_loss = trajectory.square().sum()
reference_loss = reference.square().sum()
scan_grads = torch.autograd.grad(scan_loss, tuple(cell.parameters()), retain_graph=True, allow_unused=True)
reference_grads = torch.autograd.grad(reference_loss, tuple(cell.parameters()), allow_unused=True)
gradient_error = max(float((left - right).abs().max()) for left, right in zip(scan_grads, reference_grads) if left is not None and right is not None)
print({"final_error": float((final - current).abs().max()), "trajectory_error": float((trajectory - reference).abs().max()), "gradient_error": gradient_error, "trajectory_shape": list(trajectory.shape)})
assert torch.allclose(final, current, atol=1e-6, rtol=1e-6)
assert torch.allclose(trajectory, reference, atol=1e-6, rtol=1e-6)
assert gradient_error < 1e-5
with torch.no_grad():
    for _ in range(3):
        scan(scan_step, state, xs)
        current_bench = state
        for index in range(x.shape[1]):
            _, current_bench = cell.step_fused_stacked(forcing[:, index], input_gate[:, index], transport_gate[:, index], offsets[:, index], geometry[:, index], current_bench, runtime_guard_mode="disabled", kernel_tensors=kernels, geometry_operator=operator, return_output=False)
    scan_start = __import__("time").perf_counter()
    for _ in range(10):
        scan(scan_step, state, xs)
    scan_seconds = (__import__("time").perf_counter() - scan_start) / 10.0
    loop_start = __import__("time").perf_counter()
    for _ in range(10):
        current_bench = state
        for index in range(x.shape[1]):
            _, current_bench = cell.step_fused_stacked(forcing[:, index], input_gate[:, index], transport_gate[:, index], offsets[:, index], geometry[:, index], current_bench, runtime_guard_mode="disabled", kernel_tensors=kernels, geometry_operator=operator, return_output=False)
    loop_seconds = (__import__("time").perf_counter() - loop_start) / 10.0
print({"scan_ms": scan_seconds * 1000.0, "loop_ms": loop_seconds * 1000.0, "speedup": loop_seconds / scan_seconds})
