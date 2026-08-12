import inspect

import pytest
import torch

from cdi.v3 import (
    CohomodynamicState,
    SelectiveCohomodynamicSSM,
    StageCConfig,
    StateCodec,
)


def make_model(seed: int = 42, geometry_ablation: bool = False):
    torch.manual_seed(seed)
    config = StageCConfig.nano(seed=seed, geometry_ablation=geometry_ablation)
    return SelectiveCohomodynamicSSM(config), config


def assert_states_close(left: CohomodynamicState, right: CohomodynamicState, atol: float = 1e-6):
    for left_tensor, right_tensor in zip(left.tensors(), right.tensors()):
        assert torch.allclose(left_tensor, right_tensor, atol=atol, rtol=1e-5)


def fold(model, sequence, state):
    outputs = []
    current = state
    intermediate = []
    for index in range(sequence.shape[-2]):
        output, current = model.step(sequence[..., index, :], current)
        outputs.append(output)
        intermediate.append(current)
    return torch.stack(outputs, dim=-2), current, intermediate


def test_nano_layout_and_frequency_cascade():
    model, config = make_model()
    assert config.total_state_dim == 48
    assert config.total_state_dim < 64
    assert config.band_ranges[0][1] < config.band_ranges[1][0] < config.band_ranges[1][1] < config.band_ranges[2][0]
    state = model.initial_state(batch_shape=(2,))
    assert [tuple(tensor.shape) for tensor in state.tensors()] == [(2, 4, 4)] * 3
    inventory = model.parameter_inventory()
    assert inventory["total_parameters"] > 0
    assert inventory["geometry"]["forbidden_operations"] == ["torch.kron", "dense_full_state_operator"]


@pytest.mark.parametrize("batch,length", [(1, 1), (2, 7), (2, 32)])
def test_step_chunk_and_intermediate_equivalence(batch, length):
    model, config = make_model()
    x = torch.randn(batch, length, config.input_width)
    initial = model.initial_state(batch_shape=(batch,))
    chunk_output, chunk_state, chunk_intermediate = model.forward_chunk(x, initial, return_intermediates=True)
    fold_output, fold_state, fold_intermediate = fold(model, x, initial)
    assert torch.allclose(chunk_output, fold_output, atol=1e-6, rtol=1e-5)
    assert_states_close(chunk_state, fold_state)
    for left, right in zip(chunk_intermediate, fold_intermediate):
        assert_states_close(left, right)


def test_causality():
    model, config = make_model()
    x = torch.randn(2, 12, config.input_width)
    perturbed = x.clone()
    perturbed[:, 7, :] += 10.0
    before, _ = model.forward_chunk(x)
    after, _ = model.forward_chunk(perturbed)
    assert (before[:, :7] - after[:, :7]).abs().max().item() <= 1e-6


def test_step_chunk_gradient_equivalence():
    first, config = make_model()
    second, _ = make_model()
    second.load_state_dict(first.state_dict())
    x_first = torch.randn(2, 8, config.input_width, requires_grad=True)
    x_second = x_first.detach().clone().requires_grad_(True)
    output_first, _ = first.forward_chunk(x_first)
    output_second, _, = second.forward_chunk(x_second)
    output_first.square().mean().backward()
    output_second.square().mean().backward()
    assert torch.allclose(x_first.grad, x_second.grad, atol=1e-6, rtol=1e-5)
    for (name_first, parameter_first), (name_second, parameter_second) in zip(first.named_parameters(), second.named_parameters()):
        assert name_first == name_second
        # Zero-state execution intentionally does not consume the learned-state
        # parameter. All production-path parameters must remain connected.
        if name_first == "cell.learned_initial_state":
            assert parameter_first.grad is None and parameter_second.grad is None
        else:
            assert parameter_first.grad is not None
            assert parameter_second.grad is not None
            assert torch.allclose(parameter_first.grad, parameter_second.grad, atol=1e-6, rtol=1e-5)


def test_state_codec_continuation_and_fingerprint():
    model, config = make_model()
    x = torch.randn(2, 10, config.input_width)
    _, mid_state = model.forward_chunk(x[:, :5])
    payload = StateCodec.pack(mid_state)
    restored = StateCodec.unpack(payload)
    assert StateCodec.fingerprint(mid_state) == StateCodec.fingerprint(restored)
    original_output, original_state = model.forward_chunk(x[:, 5:], mid_state)
    restored_output, restored_state = model.forward_chunk(x[:, 5:], restored)
    assert torch.allclose(original_output, restored_output, atol=1e-6, rtol=1e-5)
    assert_states_close(original_state, restored_state)


def test_zero_input_dissipative_stability():
    model, config = make_model(geometry_ablation=True)
    state = CohomodynamicState(*(torch.randn(2, 4, 4) for _ in range(3)))
    zero = torch.zeros(2, config.input_width)
    initial_energy = sum(t.square().sum().item() for t in state.tensors())
    for _ in range(200):
        _, state = model.step(zero, state)
    final_energy = sum(t.square().sum().item() for t in state.tensors())
    assert all(torch.isfinite(tensor).all() for tensor in state.tensors())
    assert final_energy <= initial_energy + 1e-6


def test_conservative_pairwise_integrator_preserves_energy():
    model, config = make_model(geometry_ablation=True)
    band = model.cell.bands["middle"]
    z = torch.randn(3, config.n_vertices, config.band_width)
    omega = torch.full((3, config.band_width // 2), 0.7)
    result = band.integrator.conservative_step(z, omega, config.dt)
    assert torch.allclose(z.square().sum(), result.square().sum(), atol=1e-5, rtol=1e-5)


def test_geometry_ablation_is_exact_and_enabled_path_is_active():
    active, config = make_model(geometry_ablation=False)
    ablated, _ = make_model(geometry_ablation=True)
    state = torch.randn(2, config.n_vertices, config.band_width)
    assert torch.equal(ablated.cell.geometry.apply(state), torch.zeros_like(state))
    assert active.cell.geometry.apply(state).abs().sum().item() > 0.0


def test_production_source_guard_has_no_dense_state_lift():
    source = inspect.getsource(SelectiveCohomodynamicSSM) + inspect.getsource(type(make_model()[0].cell))
    assert "torch.kron(" not in source
    assert "to_dense" not in source
    metadata = make_model()[0].production_metadata()
    assert metadata["state_elements"] == 48
    assert "dense_per_token_state_matrix" in metadata["forbidden_operations"]


def test_dynamics_diagnostics_are_available_after_step():
    model, config = make_model()
    output, _ = model.step(torch.ones(2, config.input_width), model.initial_state(batch_shape=(2,)))
    report = model.cell.last_diagnostics()
    assert output.shape == (2, config.output_width)
    assert report["available"]
    assert set(report["spectral_estimates"]) == {"fast", "middle", "harmonic"}
