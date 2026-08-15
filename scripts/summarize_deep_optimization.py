from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


eager = load("results/deep_optimization_eager/latest.json")
matrix = load("results/deep_optimization_matrix/latest.json")
compiled_short = load("results/deep_optimization_compiled_short/latest.json")
pilot = load("results/deep_optimization_pilot/latest.json")

rows = []
for length in (16, 64, 256):
    cdi = float(eager["models"]["dcss_residual_cdi"][str(length)]["tokens_per_second"])
    gru = float(eager["models"]["gru_baseline"][str(length)]["tokens_per_second"])
    transformer = float(eager["models"]["transformer"][str(length)]["tokens_per_second"])
    rows.append({"length": length, "cdi_eager": cdi, "gru_cell": gru, "transformer": transformer, "cdi_vs_gru": cdi / gru, "cdi_vs_transformer": cdi / transformer})

matrix_rows = []
for length in (16, 64, 256):
    cdi = float(matrix["models"]["dcss_residual_cdi"][str(length)]["tokens_per_second"])
    cell = float(matrix["models"]["gru_cell_adapter"][str(length)]["tokens_per_second"])
    fused = float(matrix["models"]["torch_nn_gru_fused"][str(length)]["tokens_per_second"])
    transformer = float(matrix["models"]["transformer"][str(length)]["tokens_per_second"])
    matrix_rows.append({"length": length, "cdi": cdi, "gru_cell": cell, "gru_fused": fused, "transformer": transformer, "cdi_vs_gru_cell": cdi / cell, "cdi_vs_gru_fused": cdi / fused, "cdi_vs_transformer": cdi / transformer})

pilot_summary = pilot["summary"]
summary = {
    "eager_rows": rows,
    "matrix_rows": matrix_rows,
    "compiled_short": compiled_short["models"]["dcss_residual_cdi"]["16"],
    "pilot_summary": pilot_summary,
    "peak_rss_gib": {
        "eager": eager["peak_rss_gib"],
        "matrix": matrix["peak_rss_gib"],
        "compiled_short": compiled_short["peak_rss_gib"],
        "pilot": pilot["host_memory"]["peak_gb"],
    },
}

print(json.dumps(summary, indent=2))
