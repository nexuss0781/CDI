from __future__ import annotations

import inspect

from benchmarks import p1_readiness, stage_b, stage_c, stage_d, stage_e, stage_f


def test_benchmark_runners_do_not_write_generated_reports_to_tracked_stages() -> None:
    modules = (p1_readiness, stage_b, stage_c, stage_d, stage_e, stage_f)
    for module in modules:
        source = inspect.getsource(module)
        assert 'Path("Stages/' not in source
        assert "Path('Stages/" not in source


def test_benchmark_runners_emit_result_directory_reports() -> None:
    modules = (p1_readiness, stage_b, stage_c, stage_d, stage_e, stage_f)
    for module in modules:
        source = inspect.getsource(module)
        assert '"REPORT.md"' in source
