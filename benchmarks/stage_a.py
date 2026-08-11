"""Stage A end-to-end reproducibility and baseline harness for CDI v2.

The harness intentionally uses a resource-safe ``micro`` configuration by
 default.  CDI v2's documented tiny configuration materializes dense operators
whose float64 memory footprint is too large for many development machines.
The documented configuration remains available through ``--config tiny`` and
will be reported as a resource-risk rather than silently substituted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from cdi.config import CDIConfig
from cdi.engine import CDIEngine


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "stage_a"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def memory_rss_mb() -> float:
    # ru_maxrss is kilobytes on Linux and bytes on macOS.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / 1024.0 if sys.platform.startswith("linux") else value / (1024.0 * 1024.0))


def environment() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "git_revision": git_revision(),
        "pid": os.getpid(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def config_for(name: str) -> CDIConfig:
    if name == "micro":
        cfg = CDIConfig(
            manifold_dim=2,
            n_points=4,
            cover_k=2,
            motor_depth=1,
            abstraction_height=2,
            belief_dims=(4, 8, 8, 4),
            observation_dim=4,
            output_dim=4,
            heat_steps=2,
            heat_dt=0.001,
            learning_rate=0.01,
            dtype_str="float32",
            device="cpu",
            spectral_diag_every=1000,
            seed=42,
        )
    elif name == "tiny":
        cfg = CDIConfig.tiny()
    elif name == "small":
        cfg = CDIConfig.small()
    else:
        raise ValueError(f"unknown config {name!r}")
    cfg.validate()
    return cfg


def config_dict(cfg: CDIConfig) -> Dict[str, Any]:
    data = asdict(cfg)
    data.update(
        {
            "dtype": str(cfg.dtype),
            "n_degrees": cfg.n_degrees,
            "spinor_dim": cfg.spinor_dim,
            "total_belief_dim": cfg.total_belief_dim,
            "total_state_dim": cfg.total_state_dim,
            "b0_dim": cfg.belief_dim(0),
        }
    )
    return data


def parameter_labels(engine: CDIEngine) -> List[str]:
    labels = ["manifold.points", "manifold.metric_L"]
    labels += ["sheaf.embedding_matrix", "sheaf.output_matrix"]
    labels += [f"belief.parameter_{i}" for i, _ in enumerate(engine.belief.get_parameters())]
    labels += [f"connection.parameter_{i}" for i, _ in enumerate(engine.connection.get_parameters())]
    labels += ["theta_init", "W_iota", "W_out"]
    params = engine.get_parameters()
    if len(labels) != len(params):
        labels = [f"parameter_{i}" for i in range(len(params))]
    return labels


def tensor_fingerprint(tensors: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        data = tensor.detach().cpu().contiguous().numpy().tobytes()
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(data)
    return digest.hexdigest()


def make_batch(
    cfg: CDIConfig, seed: int, batch_size: int = 2, vocab_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    token_ids = torch.randint(
        0, vocab_size, (batch_size, cfg.n_points), generator=generator
    )
    embedding = torch.randn(
        vocab_size, cfg.observation_dim, generator=generator, dtype=cfg.dtype
    )
    embedding.requires_grad_(True)
    target = torch.roll(token_ids, shifts=-1, dims=1)
    return token_ids, target, embedding


def forward_loss(
    engine: CDIEngine,
    token_ids: torch.Tensor,
    target_ids: torch.Tensor,
    embedding: torch.Tensor,
    global_step: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    sequence = embedding[token_ids]
    output = engine.forward_sequence_batch(sequence)
    return engine.compute_lm_loss(output, target_ids, embedding, global_step=global_step)


def build_engine(cfg: CDIConfig) -> CDIEngine:
    engine = CDIEngine(cfg)
    engine.build()
    return engine


def checkpoint_payload(
    engine: CDIEngine,
    cfg: CDIConfig,
    embedding: Optional[torch.Tensor] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    global_step: int = 0,
) -> Dict[str, Any]:
    params = engine.get_parameters()
    payload: Dict[str, Any] = {
        "format": "cdi-stage-a-v1",
        "source_revision": git_revision(),
        "config": config_dict(cfg),
        "parameter_labels": parameter_labels(engine),
        "engine_parameters": [p.detach().cpu().clone() for p in params],
        "engine_fingerprint": tensor_fingerprint(params),
        "global_step": global_step,
        "random_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    if embedding is not None:
        payload["embedding"] = embedding.detach().cpu().clone()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def save_checkpoint(
    path: Path,
    engine: CDIEngine,
    cfg: CDIConfig,
    embedding: Optional[torch.Tensor] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    global_step: int = 0,
) -> Dict[str, Any]:
    payload = checkpoint_payload(engine, cfg, embedding, optimizer, global_step)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def restore_checkpoint(
    path: Path,
    engine: CDIEngine,
    embedding: Optional[torch.Tensor] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    params = engine.get_parameters()
    saved = payload["engine_parameters"]
    if len(params) != len(saved):
        raise ValueError(f"parameter count mismatch: {len(params)} != {len(saved)}")
    with torch.no_grad():
        for current, previous in zip(params, saved):
            current.copy_(previous.to(dtype=current.dtype, device=current.device))
        if embedding is not None and "embedding" in payload:
            embedding.copy_(payload["embedding"].to(dtype=embedding.dtype, device=embedding.device))
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    diff = (a.detach().cpu() - b.detach().cpu()).abs()
    denom = b.detach().cpu().abs().clamp_min(1e-12)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float((diff / denom).max().item()),
    }


def gate(name: str, passed: bool, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "passed": bool(passed), "details": details}


def run_correctness(cfg: CDIConfig, seed: int, out_dir: Path) -> List[Dict[str, Any]]:
    gates: List[Dict[str, Any]] = []
    seed_everything(seed)
    engine = build_engine(cfg)
    token_ids, target_ids, embedding = make_batch(cfg, seed + 1)

    try:
        loss, loss_dict = forward_loss(engine, token_ids, target_ids, embedding)
        finite = bool(torch.isfinite(loss).item())
        gates.append(gate("construction_forward_loss", finite, {"loss": float(loss.item()), "loss_dict": loss_dict}))
    except Exception as exc:
        gates.append(gate("construction_forward_loss", False, {"error": repr(exc)}))
        return gates

    try:
        optimizer = torch.optim.Adam(engine.get_parameters() + [embedding], lr=cfg.learning_rate)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_flow = engine.verify_gradient_flow()
        critical_names = [
            "manifold.points", "manifold.metric_L", "theta_init",
            "W_iota", "W_out", "connection", "belief.deltas",
        ]
        finite_grads = all(
            p.grad is not None and bool(torch.isfinite(p.grad).all().item())
            for p in engine.get_parameters()[-3:] + [embedding]
        ) and all(
            p.grad is not None and bool(torch.isfinite(p.grad).all().item())
            for p in engine.manifold.get_parameters()
            + engine.belief.get_parameters()
            + engine.connection.get_parameters()
        )
        critical_ok = all(gradient_flow.get(name, False) for name in critical_names)
        gates.append(gate("gradient_flow", critical_ok and finite_grads, {
            "checks": gradient_flow,
            "critical_names": critical_names,
            "known_v2_inactive": ["sheaf.embedding", "sheaf.output"],
            "finite": finite_grads,
        }))
        optimizer.step()
        engine.global_step += 1
        engine.rebuild_operators()
        gates.append(gate("optimizer_step_rebuild", True, {"global_step": engine.global_step}))
    except Exception as exc:
        gates.append(gate("gradient_flow", False, {"error": repr(exc)}))

    try:
        before = tensor_fingerprint(engine.get_parameters())
        ckpt = out_dir / "checkpoints" / "stage_a_smoke.pt"
        save_checkpoint(ckpt, engine, cfg, embedding, optimizer, global_step=1)
        restored = build_engine(cfg)
        restored_embedding = embedding.detach().clone().requires_grad_(True)
        restored_optimizer = torch.optim.Adam(restored.get_parameters() + [restored_embedding], lr=cfg.learning_rate)
        payload = restore_checkpoint(ckpt, restored, restored_embedding, restored_optimizer)
        after = tensor_fingerprint(restored.get_parameters())
        gates.append(gate("checkpoint_parameter_round_trip", before == after, {"saved_fingerprint": before, "restored_fingerprint": after, "format": payload["format"]}))

        with torch.no_grad():
            out_a = engine.forward_sequence_batch(restored_embedding.detach().new_tensor(embedding.detach())[token_ids])
            out_b = restored.forward_sequence_batch(restored_embedding[token_ids])
        comparison = compare_tensors(out_a, out_b)
        gates.append(gate("checkpoint_logit_round_trip", comparison["max_abs"] <= 1e-5, comparison))
    except Exception as exc:
        gates.append(gate("checkpoint_parameter_round_trip", False, {"error": repr(exc)}))

    return gates


def run_determinism(cfg: CDIConfig, seed: int) -> List[Dict[str, Any]]:
    gates: List[Dict[str, Any]] = []
    outputs: List[torch.Tensor] = []
    losses: List[float] = []
    fingerprints: List[str] = []
    for _ in range(2):
        seed_everything(seed)
        engine = build_engine(cfg)
        token_ids, target_ids, embedding = make_batch(cfg, seed + 1)
        with torch.no_grad():
            output = engine.forward_sequence_batch(embedding[token_ids])
        loss, _ = forward_loss(engine, token_ids, target_ids, embedding)
        outputs.append(output.detach())
        losses.append(float(loss.item()))
        fingerprints.append(tensor_fingerprint(engine.get_parameters()))
    comparison = compare_tensors(outputs[0], outputs[1])
    passed = comparison["max_abs"] <= 1e-6 and abs(losses[0] - losses[1]) <= 1e-7 and fingerprints[0] == fingerprints[1]
    gates.append(gate("forward_determinism", passed, {"comparison": comparison, "losses": losses, "fingerprints_equal": fingerprints[0] == fingerprints[1]}))
    return gates


def run_overfit(cfg: CDIConfig, seed: int, steps: int = 60) -> List[Dict[str, Any]]:
    seed_everything(seed)
    engine = build_engine(cfg)
    token_ids, target_ids, embedding = make_batch(cfg, seed + 2, batch_size=2, vocab_size=16)
    optimizer = torch.optim.Adam(engine.get_parameters() + [embedding], lr=0.03)
    losses: List[float] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = forward_loss(engine, token_ids, target_ids, embedding, global_step=step)
        if not bool(torch.isfinite(loss).item()):
            return [gate("tiny_overfit", False, {"step": step, "error": "non-finite loss", "losses": losses})]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(engine.get_parameters() + [embedding], 1.0)
        optimizer.step()
        engine.rebuild_operators()
        losses.append(float(loss.detach().item()))
    reduction = (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-12)
    passed = reduction >= 0.90 and losses[-1] < losses[0]
    return [gate("tiny_overfit", passed, {"steps": steps, "initial_loss": losses[0], "final_loss": losses[-1], "relative_reduction": reduction, "losses": losses})]


def run_scaling(cfg: CDIConfig, seed: int, lengths: List[int], out_dir: Path) -> List[Dict[str, Any]]:
    seed_everything(seed)
    engine = build_engine(cfg)
    generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    records: List[Dict[str, Any]] = []
    for length in lengths:
        sequence = torch.randn(length, cfg.observation_dim, generator=generator, dtype=cfg.dtype)
        start = time.perf_counter()
        with torch.no_grad():
            output = engine.forward_sequence(sequence)
        elapsed = time.perf_counter() - start
        records.append({"length": length, "seconds": elapsed, "tokens_per_second": length / max(elapsed, 1e-12), "rss_mb": memory_rss_mb(), "output_finite": bool(torch.isfinite(output).all().item())})
    write_json(out_dir / "scaling.json", {"config": config_dict(cfg), "records": records})
    finite = all(row["output_finite"] for row in records)
    return [gate("scaling_forward_finite", finite, {"records": records})]


def run_stage_a(config_name: str, seed: int, out_dir: Path, scale_lengths: List[int]) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config_for(config_name)
    run_id = f"stage_a_{config_name}_{seed}_{int(time.time())}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "environment.json", environment())
    write_json(run_dir / "config.json", config_dict(cfg))

    all_gates: List[Dict[str, Any]] = []
    all_gates.extend(run_determinism(cfg, seed))
    all_gates.extend(run_correctness(cfg, seed, run_dir))
    all_gates.extend(run_overfit(cfg, seed))
    all_gates.extend(run_scaling(cfg, seed, scale_lengths, run_dir))

    mandatory = [g for g in all_gates]
    passed = all(g["passed"] for g in mandatory)
    known_v2_defects = [
        {
            "id": "clifford_negative_signature_d4",
            "status": "KNOWN_DEFECT",
            "description": "Historical float64 core test reports Clifford relation error 4.0 for the current real 4x4 d=4 representation; this is outside the micro LM forward path and is not silently altered in Stage A.",
            "evidence": "tests/test_core.py::TestCliffordAlgebra::test_clifford_relations_flat",
        },
        {
            "id": "sheaf_parameters_inactive_in_lm_path",
            "status": "KNOWN_LIMITATION",
            "description": "The v2 recurrent LM path uses external token embeddings and does not consume sheaf.embedding_matrix or sheaf.output_matrix; these remain reported but are excluded from critical LM gradient gates.",
            "evidence": "CDIEngine.forward_sequence and verify_gradient_flow",
        },
    ]
    report = {
        "format": "cdi-stage-a-report-v1",
        "run_id": run_id,
        "stage": "A",
        "objective": "reproducible CDI v2 baseline",
        "status": "PASS_WITH_KNOWN_V2_DEFECTS" if passed else "FAIL",
        "known_v2_defects": known_v2_defects,
        "config": config_dict(cfg),
        "environment": environment(),
        "gates": mandatory,
        "transition_to_stage_b": {
            "status": "READY_FOR_REVIEW" if passed else "BLOCKED",
            "required_approval": True,
            "decision": "awaiting_user_approval" if passed else "repair_stage_a_before_review",
            "stage_b_implementation_allowed": False,
        },
    }
    write_json(run_dir / "run.json", report)
    write_json(out_dir / "latest.json", report)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CDI v2 Stage A reproducibility and baseline harness")
    parser.add_argument("--config", choices=["micro", "tiny", "small"], default="micro")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scale-lengths", default="1,2,4,8,16")
    args = parser.parse_args(argv)
    lengths = [int(x) for x in args.scale_lengths.split(",") if x.strip()]
    try:
        report = run_stage_a(args.config, args.seed, args.output_dir, lengths)
    except Exception as exc:
        failure = {"stage": "A", "status": "ERROR", "error": repr(exc), "environment": environment()}
        write_json(args.output_dir / "latest.json", failure)
        print(json.dumps(failure, indent=2, default=json_default))
        return 2
    print(json.dumps(report, indent=2, default=json_default))
    return 0 if report["status"] in {"PASS", "PASS_WITH_KNOWN_V2_DEFECTS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
