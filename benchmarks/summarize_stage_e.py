"""Print selected high-level metrics from a Stage E report JSON."""
from __future__ import annotations

import json
from pathlib import Path
import sys


report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("status", report["status"])
print("elapsed_seconds", report["elapsed_seconds"])
for gate in report["gates"]:
    print(gate["name"], gate["status"])
    if gate["name"] == "matched_multi_seed_training":
        for identifier, summary in gate["details"]["summary"].items():
            print("quality", identifier, summary["final_validation_loss"]["mean"], summary["final_validation_loss"]["std"], summary["parameter_count"])
    elif gate["name"] == "sequence_scaling":
        print("time_exponent", gate["details"]["forward_time_fit"]["exponent"])
        print("persistent_memory_exponent", gate["details"]["persistent_memory_fit"]["exponent"])
    elif gate["name"] == "streaming_and_allocation_audit":
        print("state_bytes", gate["details"]["state_bytes"])
        print("warm_latency_ms", gate["details"]["warm_latency_ms_mean"])
    elif gate["name"] == "long_context_harmonic_retention":
        print("harmonic_retention", gate["details"]["full_retained_ratio"])
    elif gate["name"] == "fresh_reproducibility_rerun":
        print("repro_loss_error", gate["details"]["loss_max_abs"])
print("decision", report["analysis"]["details"]["decision"])
print("stage_f_allowed", report["stage_f_implementation_allowed"])
