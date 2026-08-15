from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch

from cdi.v3 import EthioBBPETokenizer, TokenizerConfig

smoke_root = Path("/tmp/cdi-compiled-smoke-drive")
smoke_root.mkdir(parents=True, exist_ok=True)
os.environ["CDI_DRIVE_ROOT"] = str(smoke_root)
os.environ["CDI_STAGE"] = "m1.2"
os.environ["CDI_PARENT_STAGE"] = ""
os.environ["CDI_PARENT_DATA_VARIANT"] = "base"
os.environ["CDI_SESSION_ID"] = "compiled_smoke"
os.environ["CDI_DATA_VARIANT"] = "base"
os.environ["CDI_DATA_ROOT"] = str(smoke_root / "dataset")
os.environ["CDI_RUN_ROOT"] = str(smoke_root / "module1" / "M1.2" / "sessions" / "compiled_smoke")
os.environ["CDI_CHECKPOINT_ROOT"] = str(Path(os.environ["CDI_RUN_ROOT"]) / "checkpoints")
os.environ["CDI_REPORT_ROOT"] = str(Path(os.environ["CDI_RUN_ROOT"]) / "reports")
os.environ["CDI_LOG_ROOT"] = str(Path(os.environ["CDI_RUN_ROOT"]) / "logs")
os.environ["CDI_CACHE_ROOT"] = str(smoke_root / "cache")
os.environ["CDI_SKIP_INSTALL"] = "1"

spec = importlib.util.spec_from_file_location("cdi_colab_embedded", "/tmp/cdi_bash_embedded.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load embedded bash.sh Python program")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.CONFIG["dataset"]["train_tokens"] = 63
module.CONFIG["optimization"]["chunk_length"] = 64
module.CONFIG["execution"]["compiled"] = True

tokenizer = EthioBBPETokenizer.from_pretrained(TokenizerConfig(max_chunk_length=64, embedding_dim=4))
model = module.build_model(tokenizer, "cpu", 42)
runner = module.build_runner(model, True, "reduce-overhead")
rows = [[tokenizer.bos_id] + [int(value) for value in torch.randint(1, tokenizer.vocab_size, (63,)).tolist()]]
manifest = {"fingerprint": "compiled-smoke"}
loss_before = module.loss_on(model, runner, rows, 2, "cpu", True, max_batches=1)
_, _, step, digest = module.train_phase(
    "compiled_smoke",
    model,
    runner,
    rows,
    tokenizer,
    "cpu",
    0.001,
    2,
    63,
    Path("/tmp/cdi-compiled-smoke-drive/checkpoint.pt"),
    manifest,
    0,
    True,
)
loss_after = module.loss_on(model, runner, rows, 2, "cpu", True, max_batches=1)
checkpoint = Path("/tmp/cdi-compiled-smoke-drive/checkpoint.pt")
print({"loss_before": loss_before, "loss_after": loss_after, "step": step, "checkpoint_exists": checkpoint.is_file(), "checkpoint_sha256": digest})
assert step == 1
assert checkpoint.is_file()
assert len(digest) == 64
assert torch.isfinite(torch.tensor([loss_before, loss_after])).all()
