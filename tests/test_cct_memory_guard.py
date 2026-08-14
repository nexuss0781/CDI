from __future__ import annotations

import pytest

import benchmarks.ethiobbpe_synaxarium_pilot as pilot


def test_host_memory_monitor_records_usage_and_fails_at_limit(monkeypatch) -> None:
    samples = iter([5 * (1024 ** 3), 11 * (1024 ** 3)])
    monkeypatch.setattr(pilot, "host_resident_memory_bytes", lambda: next(samples))
    monitor = pilot.HostMemoryMonitor(11.0)
    assert monitor.check("before") == 5 * (1024 ** 3)
    assert monitor.as_dict()["peak_gb"] == 5.0
    with pytest.raises(pilot.HostMemoryLimitExceeded, match="configured CCT limit is 11.000 GiB"):
        monitor.check("at_limit")
    assert monitor.as_dict()["peak_gb"] == 11.0


def test_pilot_config_rejects_nonpositive_host_memory_limit() -> None:
    with pytest.raises(ValueError, match="max_host_memory_gb"):
        pilot.PilotConfig(max_host_memory_gb=0.0).validate()
