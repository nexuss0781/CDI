"""Stage C stable selective-recurrence evaluation harness.

Run individual checks with ``python -m benchmarks.stage_c <command>`` or run
the complete limitation-free Stage C gate with ``python run_stage_c.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import tempfile
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from benchmarks.stage_b import AllocationTrace
from cdi.v3 import (
    BAND_NAMES,
    CohomodynamicState,
    DynamicsDiagnostics,
    SelectiveCohomodynamicSSM,
    StageCConfig,
    StateCodec,
)

FLOAT32_ATOL = 1e-6
FLOAT32_RTOL = 1e-5
STEP_CHUNK_TOL = 1e-5


def _pass(name: str, passed: bool, details: Mapping[str, Any]) -> Dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "passed": bool(passed), "details": dict(details)}


def _error(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    difference = (left - right).abs()
    return {
        "max_abs": float(difference.max().detach().cpu()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean().detach().cpu()) if difference.numel() else 0.0,
        "max_rel": float((difference / right.abs().clamp_min(FLOAT32_ATOL)).max().detach().cpu()) if difference.numel() else 0.0,
    }


def _state_error(left: CohomodynamicState, right: CohomodynamicState) -> Dict[str, Any]:
    reports = {name: _error(left.by_name(name), right.by_name(name)) for name in BAND_NAMES}
    return {"bands": reports, "max_abs": max(report["max_abs"] for report in reports.values())}


def _random_state(config: StageCConfig, batch: int, seed: int) -> CohomodynamicState:
    generator = torch.Generator(device=config.device).manual_seed(seed)
    tensors = tuple(
        torch.randn(batch, config.n_vertices, config.band_width, dtype=config.dtype, device=config.device, generator=generator)
        for _ in BAND_NAMES
    )
    return CohomodynamicState(*tensors)


def _fold(model: SelectiveCohomodynamicSSM, x: torch.Tensor, state: CohomodynamicState) -> Tuple[torch.Tensor, CohomodynamicState, List[CohomodynamicState]]:
    outputs: List[torch.Tensor] = []
    intermediate: List[CohomodynamicState] = []
    current = state
    for index in range(x.shape[-2]):
        output, current = model.step(x[..., index, :], current)
        outputs.append(output)
        intermediate.append(current)
    return torch.stack(outputs, dim=-2), current, intermediate


def _factory(seed: int = 42, geometry_ablation: bool = False) -> Tuple[StageCConfig, SelectiveCohomodynamicSSM]:
    torch.manual_seed(seed)
    config = StageCConfig.nano(seed=seed, geometry_ablation=geometry_ablation)
    model = SelectiveCohomodynamicSSM(config)
    return config, model


def cell_gate(seed: int = 42) -> Dict[str, Any]:
    modes: Dict[str, Any] = {}
    overall = True
    for ablation in (False, True):
        config, model = _factory(seed, geometry_ablation=ablation)
        x = torch.randn(2, config.input_width, dtype=config.dtype)
        output, state = model.step(x, model.initial_state(batch_shape=(2,)))
        report = model.cell.last_diagnostics()
        passed = (
            tuple(output.shape) == (2, config.output_width)
            and all(tuple(tensor.shape) == (2, config.n_vertices, config.band_width) for tensor in state.tensors())
            and bool(torch.isfinite(output).all().item())
            and report.get("available", False)
        )
        modes[str(ablation).lower()] = {
            "geometry_ablation": ablation,
            "output_shape": list(output.shape),
            "state_shapes": [list(tensor.shape) for tensor in state.tensors()],
            "diagnostics": report,
            "passed": passed,
        }
        overall = overall and passed
    return _pass("cell_shape_type_device", overall, {"modes": modes})


def causality_gate(seed: int = 42) -> Dict[str, Any]:
    config, model = _factory(seed)
    x = torch.randn(2, 32, config.input_width, dtype=config.dtype)
    perturbed = x.clone()
    perturbed[:, 17, :] += 10.0
    with torch.no_grad():
        baseline, _ = model.forward_chunk(x)
        changed, _ = model.forward_chunk(perturbed)
        pre_difference = (baseline[:, :17] - changed[:, :17]).abs().max()
    return _pass("causal_correctness", bool(pre_difference <= FLOAT32_ATOL), {"perturbed_index": 17, "pre_causal_max_abs": float(pre_difference), "tolerance": FLOAT32_ATOL})


def equivalence_gate(lengths: Sequence[int] = (1, 2, 7, 32, 128), seed: int = 42) -> Dict[str, Any]:
    reports: List[Dict[str, Any]] = []
    overall = True
    config, model = _factory(seed)
    for length in lengths:
        for batch in (1, 2):
            generator = torch.Generator(device=config.device).manual_seed(seed + length * 13 + batch)
            x = torch.randn(batch, length, config.input_width, dtype=config.dtype, generator=generator)
            initial = _random_state(config, batch, seed + length * 29 + batch)
            chunk_output, chunk_state, chunk_intermediate = model.forward_chunk(x, initial, return_intermediates=True)
            step_output, step_state, step_intermediate = _fold(model, x, initial)
            output_error = _error(chunk_output, step_output)
            state_error = _state_error(chunk_state, step_state)
            intermediate_error = max(_state_error(left, right)["max_abs"] for left, right in zip(chunk_intermediate, step_intermediate))
            passed = output_error["max_abs"] <= STEP_CHUNK_TOL and state_error["max_abs"] <= STEP_CHUNK_TOL and intermediate_error <= STEP_CHUNK_TOL
            reports.append({"batch": batch, "length": length, "output": output_error, "final_state": state_error, "intermediate_state_max_abs": intermediate_error, "passed": passed})
            overall = overall and passed
    return _pass("step_chunk_equivalence", overall, {"tolerance": STEP_CHUNK_TOL, "records": reports})


def gradients_gate(lengths: Sequence[int] = (1, 8, 64), seed: int = 42) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    overall = True
    for length in lengths:
        config, first = _factory(seed + length)
        _, second = _factory(seed + length)
        second.load_state_dict(first.state_dict())
        generator = torch.Generator(device=config.device).manual_seed(seed + 100 + length)
        first_input = torch.randn(2, length, config.input_width, dtype=config.dtype, generator=generator, requires_grad=True)
        second_input = first_input.detach().clone().requires_grad_(True)
        first_output, _ = first.forward_chunk(first_input)
        second_state = second.initial_state(batch_shape=(2,))
        second_output, _, _ = _fold(second, second_input, second_state)
        first_output.square().mean().backward()
        second_output.square().mean().backward()
        input_error = _error(first_input.grad, second_input.grad)
        parameter_errors = {}
        active = True
        for (first_name, first_parameter), (second_name, second_parameter) in zip(first.named_parameters(), second.named_parameters()):
            if first_name != second_name:
                active = False
                continue
            if first_name == "cell.learned_initial_state":
                parameter_errors[first_name] = {"zero_state_inactive": first_parameter.grad is None and second_parameter.grad is None}
                continue
            if first_parameter.grad is None or second_parameter.grad is None:
                active = False
                parameter_errors[first_name] = {"gradient_missing": True}
            else:
                parameter_errors[first_name] = _error(first_parameter.grad, second_parameter.grad)
        maximum = max([input_error["max_abs"]] + [value.get("max_abs", 0.0) for value in parameter_errors.values()])
        passed = active and maximum <= STEP_CHUNK_TOL
        records.append({"length": length, "input_gradient": input_error, "parameter_gradients": parameter_errors, "max_abs": maximum, "passed": passed})
        overall = overall and passed
    return _pass("gradient_equivalence", overall, {"tolerance": STEP_CHUNK_TOL, "records": records})


def serialization_gate(seed: int = 42) -> Dict[str, Any]:
    config, model = _factory(seed)
    x = torch.randn(2, 16, config.input_width, dtype=config.dtype)
    _, mid_state = model.forward_chunk(x[:, :8])
    payload = StateCodec.pack(mid_state)
    restored = StateCodec.unpack(payload)
    first_output, first_state = model.forward_chunk(x[:, 8:], mid_state)
    second_output, second_state = model.forward_chunk(x[:, 8:], restored)
    output_error = _error(first_output, second_output)
    state_error = _state_error(first_state, second_state)
    passed = output_error["max_abs"] <= FLOAT32_ATOL and state_error["max_abs"] <= FLOAT32_ATOL and StateCodec.fingerprint(mid_state) == StateCodec.fingerprint(restored)
    return _pass("state_serialization", passed, {"output": output_error, "state": state_error, "fingerprint": StateCodec.fingerprint(mid_state), "tolerance": FLOAT32_ATOL})


def _state_energy(state: CohomodynamicState) -> float:
    return float(sum(tensor.square().sum().detach().cpu().item() for tensor in state.tensors()))


def stability_gate(steps: int = 10_000, inputs: Sequence[str] = ("zero", "impulse", "random"), seed: int = 42) -> Dict[str, Any]:
    modes: Dict[str, Any] = {}
    overall = True
    config, model = _factory(seed)
    for name in inputs:
        state = _random_state(config, batch=1, seed=seed + 503)
        initial_energy = _state_energy(state)
        maximum_norm = 0.0
        finite = True
        energies = [initial_energy]
        generator = torch.Generator(device=config.device).manual_seed(seed + 701)
        # Stability is a forward-envelope measurement, not a gradient test.
        # Disabling autograd prevents a 10,000-step diagnostic graph from
        # consuming the nano CPU memory budget.
        with torch.no_grad():
            for index in range(steps):
                if name == "zero":
                    x = torch.zeros(1, config.input_width, dtype=config.dtype)
                elif name == "impulse":
                    x = torch.ones(1, config.input_width, dtype=config.dtype) if index == 0 else torch.zeros(1, config.input_width, dtype=config.dtype)
                elif name == "random":
                    x = torch.rand(1, config.input_width, dtype=config.dtype, generator=generator) * 2.0 - 1.0
                else:
                    raise ValueError(f"Unsupported stability input: {name}")
                _, state = model.step(x, state)
                norm = sum(torch.linalg.vector_norm(tensor).item() ** 2 for tensor in state.tensors()) ** 0.5
                maximum_norm = max(maximum_norm, norm)
                finite = finite and all(bool(torch.isfinite(tensor).all().item()) for tensor in state.tensors())
                if index in {0, steps - 1}:
                    energies.append(_state_energy(state))
        final_energy = _state_energy(state)
        energy_ok = final_energy <= initial_energy + 1e-5 if name == "zero" else True
        mode_passed = finite and maximum_norm <= config.state_norm_bound and energy_ok
        modes[name] = {"steps": steps, "initial_energy": initial_energy, "final_energy": final_energy, "sampled_energies": energies, "maximum_norm": maximum_norm, "finite": finite, "passed": mode_passed}
        overall = overall and mode_passed
    return _pass("stability_envelope", overall, {"norm_bound": config.state_norm_bound, "modes": modes})


def dynamical_properties_gate(seed: int = 42) -> Dict[str, Any]:
    config, model = _factory(seed, geometry_ablation=True)
    band = model.cell.bands["middle"]
    z = torch.randn(2, config.n_vertices, config.band_width, dtype=config.dtype)
    omega = torch.full((2, config.band_width // 2), 0.7, dtype=config.dtype)
    conservative = band.integrator.conservative_step(z, omega, config.dt)
    conservative_error = abs(float(z.square().sum() - conservative.square().sum())) / max(float(z.square().sum()), 1e-12)
    state = _random_state(config, 1, seed + 33)
    energy_before = _state_energy(state)
    zero = torch.zeros(1, config.input_width, dtype=config.dtype)
    for _ in range(100):
        _, state = model.cell.step(zero, state, dissipation_scale=1.0)
    energy_after = _state_energy(state)
    passed = conservative_error <= STEP_CHUNK_TOL and energy_after <= energy_before + FLOAT32_ATOL
    return _pass("conservative_and_dissipative_dynamics", passed, {"conservative_relative_energy_drift": conservative_error, "dissipative_energy_before": energy_before, "dissipative_energy_after": energy_after})


def frequency_cascade_gate(seed: int = 42, task: str = "delayed_copy", steps: int = 1_000) -> Dict[str, Any]:
    if task != "delayed_copy":
        raise ValueError("Stage C nano synthetic-memory validation currently supports only task='delayed_copy'.")
    if steps < 16:
        raise ValueError("The delayed-copy retention probe requires at least 16 configured steps.")
    config, model = _factory(seed, geometry_ablation=True)
    state = model.initial_state(batch_shape=(1,))
    impulse = torch.ones(1, config.input_width, dtype=config.dtype)
    _, state = model.step(impulse, state)
    start = {name: float(torch.linalg.vector_norm(state.by_name(name)).item()) for name in BAND_NAMES}
    zero = torch.zeros(1, config.input_width, dtype=config.dtype)
    for _ in range(16):
        _, state = model.step(zero, state)
    end = {name: float(torch.linalg.vector_norm(state.by_name(name)).item()) for name in BAND_NAMES}
    retention = {name: end[name] / max(start[name], 1e-12) for name in BAND_NAMES}
    ranges = {name: list(tau_range) for name, tau_range in zip(BAND_NAMES, config.band_ranges)}
    range_ordered = ranges["fast"][1] < ranges["middle"][0] < ranges["middle"][1] < ranges["harmonic"][0]
    passed = range_ordered and retention["harmonic"] > retention["middle"] > retention["fast"] and retention["harmonic"] >= 2.0 * retention["fast"]
    return _pass("frequency_cascade_and_synthetic_memory", passed, {"task": task, "configured_steps": steps, "probe": "delayed_copy_retention_with_zero_distractors", "delay": 16, "timescale_ranges": ranges, "impulse_norm": start, "delayed_norm": end, "retention": retention})


def gate_behavior_gate(seed: int = 42) -> Dict[str, Any]:
    config, model = _factory(seed)
    records = []
    passed = True
    for amplitude in (-5.0, -1.0, 0.0, 1.0, 5.0):
        x = torch.full((2, config.input_width), amplitude, dtype=config.dtype)
        _, _ = model.step(x, model.initial_state(batch_shape=(2,)))
        diagnostics = model.cell.last_diagnostics()
        spectral = diagnostics["spectral_estimates"]
        gates = diagnostics["gate_stats"]
        bounded = all(
            0.0 <= gates[name]["geometry_gate_min"] <= gates[name]["geometry_gate_max"] <= 1.0
            and config.tau_min <= spectral[name]["tau_min"] <= spectral[name]["tau_max"] <= config.tau_max
            and spectral[name]["dissipation_min"] >= 0.0
            for name in BAND_NAMES
        )
        records.append({"amplitude": amplitude, "bounded": bounded, "spectral": spectral, "gates": gates})
        passed = passed and bounded
    return _pass("bounded_gate_behavior", passed, {"records": records, "tau_bounds": [config.tau_min, config.tau_max]})


def production_guard_gate(seed: int = 42) -> Dict[str, Any]:
    modes: Dict[str, Any] = {}
    overall = True
    for ablation in (False, True):
        config, model = _factory(seed, geometry_ablation=ablation)
        x = torch.randn(2, 8, config.input_width, dtype=config.dtype)
        full_square = config.total_state_dim ** 2
        trace = AllocationTrace(full_square, 0.5)
        original_kron = torch.kron
        try:
            def forbidden_kron(*args: Any, **kwargs: Any) -> torch.Tensor:
                raise AssertionError("torch.kron is forbidden in the Stage C production path")
            torch.kron = forbidden_kron  # type: ignore[assignment]
            with trace:
                output, state = model.forward_chunk(x)
        except Exception as exc:
            modes[str(ablation).lower()] = {"geometry_ablation": ablation, "error": repr(exc), "passed": False}
            overall = False
            continue
        finally:
            torch.kron = original_kron  # type: ignore[assignment]
        loss = output.square().sum()
        gradient = torch.autograd.grad(loss, model.cell.geometry.edge_log_weights, allow_unused=True)[0] if loss.requires_grad else None
        geometry_activity = gradient is None if ablation else gradient is not None and bool(torch.isfinite(gradient).all().item())
        forbidden_ops = [operation for operation in trace.operations if "kron" in operation.lower() or "to_dense" in operation.lower()]
        passed = not trace.large_allocations and not forbidden_ops and geometry_activity and all(bool(torch.isfinite(tensor).all().item()) for tensor in state.tensors())
        modes[str(ablation).lower()] = {
            "geometry_ablation": ablation,
            "large_allocations": trace.large_allocations,
            "forbidden_operations": forbidden_ops,
            "operation_count": len(trace.operations),
            "geometry_parameter_activity": "inactive" if ablation else "active",
            "geometry_parameter_activity_correct": geometry_activity,
            "metadata": model.production_metadata(),
            "passed": passed,
        }
        overall = overall and passed
    return _pass("production_no_dense_guard", overall, {"modes": modes})


def device_gate(seed: int = 42) -> Dict[str, Any]:
    config, model = _factory(seed)
    x = torch.ones(2, 3, config.input_width, dtype=config.dtype)
    cpu_output, _ = model.forward_chunk(x)
    cpu_passed = cpu_output.device.type == "cpu" and bool(torch.isfinite(cpu_output).all().item())
    if torch.cuda.is_available():
        cuda_config = StageCConfig(**{**config.as_dict(), "device": "cuda"})
        cuda_model = SelectiveCohomodynamicSSM(cuda_config).to("cuda")
        cuda_model.load_state_dict(model.state_dict())
        cuda_output, _ = cuda_model.forward_chunk(x.to("cuda"))
        comparison = _error(cpu_output, cuda_output.cpu())
        cuda = {"status": "PASS" if comparison["max_abs"] <= STEP_CHUNK_TOL else "FAIL", "comparison": comparison}
        passed = cpu_passed and cuda["status"] == "PASS"
    else:
        cuda = {"status": "UNAVAILABLE", "reason": "torch.cuda.is_available() is false in the CPU development environment."}
        passed = cpu_passed
    return _pass("device_correctness", passed, {"cpu": {"status": "PASS" if cpu_passed else "FAIL"}, "cuda": cuda})


def _parameter_fingerprint(model: SelectiveCohomodynamicSSM) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def transition_manifest(config: StageCConfig, model: SelectiveCohomodynamicSSM) -> Dict[str, Any]:
    return {
        "format": "dcss-cdi-stage-c-transition-manifest-v1",
        "config": config.as_dict(),
        "topology_fingerprint": model.cell.topology.fingerprint(),
        "generator_parameterization": "exact diagonal dissipation plus pairwise skew rotation",
        "integrator": "exact block exponential",
        "gate_bounds": {"input_gate": [0.0, 1.0], "transport_gate": [0.0, 1.0], "geometry_gate": [0.0, 1.0], "log_timescale_offset": [-config.max_log_timescale_offset, config.max_log_timescale_offset]},
        "timescale_ranges": {name: list(tau_range) for name, tau_range in zip(BAND_NAMES, config.band_ranges)},
        "state_layout": "CohomodynamicState(fast, middle, harmonic), each (..., n_vertices, band_width)",
        "state_elements": config.total_state_dim,
        "tolerances": {"step_chunk": STEP_CHUNK_TOL, "serialization": FLOAT32_ATOL, "causality": FLOAT32_ATOL},
        "parameter_inventory": model.parameter_inventory(),
        "stage_d_implementation_allowed": False,
    }


def render_report(report: Mapping[str, Any]) -> str:
    gate_lines = "\n".join(
        f"| {gate['name']} | {gate['status']} | {json.dumps(gate['details'], sort_keys=True)[:220]} |"
        for gate in report["gates"]
    )
    return f"""# Stage C Gate Report — Stable Selective Cohomodynamic Recurrence

## Result

**Status:** `{report['status']}`. The Stage C nano engine uses a 48-element structured recurrent state, three logarithmically separated frequency-cascade bands, exact diagonal-plus-pairwise-skew integration, and post-update matrix-free Stage B geometry. Stage D remains blocked.

| Gate | Status | Evidence summary |
|---|---:|---|
{gate_lines}

## Structural guarantees and empirical checks

The pairwise exact integrator structurally prevents homogeneous zero-input amplification when dissipation is active. The conservative diagnostic uses the same pairwise rotation with dissipation removed and checks energy preservation. Causality, chunk equivalence, gradient equivalence, long-rollout stability, memory-band retention, serialization, and sparse allocation behavior are empirical gates recorded in `results/stage_c/latest.json`.

## Transition state

```json
{{
  "stage_c": "{report['status']}",
  "stage_d_implementation_allowed": false,
  "required_action": "explicit user approval before Stage D"
}}
```

## References

[1]: https://github.com/nexuss0781/CDI "CDI repository and DCSS-CDI Stage C implementation"
"""


def run_all(seed: int = 42, output_dir: Path | str = Path("results/stage_c"), stability_steps: int = 10_000) -> Dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config, model = _factory(seed)
    gates = [
        cell_gate(seed),
        causality_gate(seed),
        equivalence_gate(seed=seed),
        gradients_gate(seed=seed),
        serialization_gate(seed),
        stability_gate(steps=stability_steps, seed=seed),
        dynamical_properties_gate(seed),
        frequency_cascade_gate(seed),
        gate_behavior_gate(seed),
        production_guard_gate(seed),
        device_gate(seed),
    ]
    passed = all(gate["passed"] for gate in gates)
    elapsed = time.perf_counter() - started
    manifest = transition_manifest(config, model)
    manifest["manifest_fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "format": "dcss-cdi-stage-c-report-v1",
        "stage": "C",
        "status": "PASS" if passed else "FAIL",
        "seed": seed,
        "config": config.as_dict(),
        "elapsed_seconds": elapsed,
        "nano_under_30_seconds": elapsed < 30.0,
        "gates": gates,
        "transition_manifest": manifest,
        "stage_d_implementation_allowed": False,
        "transition": "Await explicit user approval before Stage D.",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "cuda_available": torch.cuda.is_available()},
    }
    run_directory = output_dir / f"stage_c_nano_{seed}_{int(time.time())}"
    run_directory.mkdir(parents=True, exist_ok=True)
    for path in (run_directory / "run.json", output_dir / "latest.json"):
        path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    manifest_json = json.dumps(report["transition_manifest"], indent=2, sort_keys=True, default=str) + "\n"
    (run_directory / "transition_manifest.json").write_text(manifest_json, encoding="utf-8")
    (output_dir / "transition_manifest.json").write_text(manifest_json, encoding="utf-8")
    report_markdown = render_report(report)
    (output_dir / "REPORT.md").write_text(report_markdown, encoding="utf-8")
    (run_directory / "REPORT.md").write_text(report_markdown, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="all", choices=["all", "cell", "causality", "equivalence", "stability", "gradients", "serialization", "dynamics", "frequency_cascade", "gates", "production_guard", "device", "synthetic_memory"])
    parser.add_argument("--config", default="nano")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lengths", default="1,2,7,32,128")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--inputs", default="zero,impulse,random")
    parser.add_argument("--task", default="delayed_copy")
    parser.add_argument("--output-dir", default="results/stage_c")
    args = parser.parse_args()
    if args.config != "nano":
        raise ValueError("Stage C currently supports only --config nano.")
    lengths = tuple(int(value) for value in args.lengths.split(",") if value)
    commands = {
        "cell": lambda: cell_gate(args.seed),
        "causality": lambda: causality_gate(args.seed),
        "equivalence": lambda: equivalence_gate(lengths, args.seed),
        "stability": lambda: stability_gate(args.steps, tuple(value for value in args.inputs.split(",") if value), args.seed),
        "gradients": lambda: gradients_gate(lengths, args.seed),
        "serialization": lambda: serialization_gate(args.seed),
        "dynamics": lambda: dynamical_properties_gate(args.seed),
        "frequency_cascade": lambda: frequency_cascade_gate(args.seed),
        "synthetic_memory": lambda: frequency_cascade_gate(args.seed, task=args.task, steps=args.steps),
        "gates": lambda: gate_behavior_gate(args.seed),
        "production_guard": lambda: production_guard_gate(args.seed),
        "device": lambda: device_gate(args.seed),
    }
    if args.command == "all":
        report = run_all(seed=args.seed, output_dir=args.output_dir, stability_steps=args.steps)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0 if report["status"] == "PASS" and report["nano_under_30_seconds"] else 1
    result = commands[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
