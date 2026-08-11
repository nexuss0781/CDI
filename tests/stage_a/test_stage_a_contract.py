"""Low-memory tests for Stage A harness contracts."""
from pathlib import Path

from benchmarks.stage_a import config_for, config_dict, make_batch, seed_everything


def test_micro_config_is_valid_and_resource_safe():
    cfg = config_for("micro")
    # The verified real Cl(0,2) module has spinor dimension four. The
    # corrected micro profile remains CPU-safe while allowing this proper
    # negative-signature representation.
    assert cfg.total_state_dim <= 512
    assert cfg.dtype_str == "float32"
    assert cfg.belief_dim(0) >= cfg.observation_dim
    assert cfg.total_belief_dim >= 4 * cfg.observation_dim


def test_tiny_profile_is_real_clifford_memory_safe():
    from cdi.config import CDIConfig

    cfg = CDIConfig.tiny()
    assert cfg.spinor_dim == 8
    assert cfg.total_state_dim <= 2048
    cfg.validate()


def test_micro_config_serializes_all_derived_dimensions():
    data = config_dict(config_for("micro"))
    for key in ("n_degrees", "spinor_dim", "total_belief_dim", "total_state_dim", "b0_dim"):
        assert key in data


def test_synthetic_batch_is_reproducible():
    seed_everything(42)
    first = make_batch(config_for("micro"), 43)
    seed_everything(42)
    second = make_batch(config_for("micro"), 43)
    for a, b in zip(first, second):
        assert a.equal(b)
