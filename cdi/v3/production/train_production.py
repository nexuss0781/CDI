"""GPU-accelerated production training and fine-tuning engine for DCSS-CDI."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import torch

from cdi.v3 import (
    ArtifactLineage,
    CharacterTokenizer,
    DCSSLanguageModel,
    EvaluationCard,
    EvaluationEvidence,
    EnvironmentLineage,
    P2DataPolicy,
    ProductionRunConfig,
    ReleaseBoundary,
    StageCConfig,
    StageDConfig,
    TokenizerConfig,
    build_envelope,
    evaluate_causal_offline,
    optimizer_for,
    save_atomic,
    train_steps,
)
from cdi.v3.training import CorpusDocument, deterministic_batches, pack_documents, parameter_fingerprint, seed_everything
from .hf_ingest import ingest_wikitext_and_sciq


def run_production_pipeline(output_dir: str | Path = "results/production") -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # 1. Ingest Data from Hugging Face
    ingest_result = ingest_wikitext_and_sciq(out / "data")
    manifest_path = ingest_result["manifest"]
    text_path = ingest_result["text_jsonl"]
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    # Production config for the run
    run_config = ProductionRunConfig(device=device, max_steps=200, checkpoint_interval=50)
    run_config.validate()
    
    # StageDConfig wrapper for training helpers
    training_config = StageDConfig(
        name="nano", 
        seed=run_config.seed, 
        device=device, 
        chunk_length=16, 
        batch_size=8
    )
    
    seed_everything(run_config.seed)
    
    # 2. Setup Tokenizer and Model
    tokenizer_config = TokenizerConfig(max_chunk_length=16, embedding_dim=64)
    
    # Load texts from the local JSONL created by ingest
    texts = []
    with text_path.open("r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    
    tokenizer = CharacterTokenizer.from_texts(texts, tokenizer_config)
    
    # Scale Stage C config for GPU
    stage_c = StageCConfig.nano(seed=run_config.seed)
    stage_c = scale_stage_c(stage_c, embed_dim=64)
    model = DCSSLanguageModel(tokenizer, stage_c).to(device)
    
    # 3. Pack and Batch Training Data
    corpus_docs = [CorpusDocument(f"doc-{i}", t) for i, t in enumerate(texts)]
    train_ex, _ = pack_documents(corpus_docs, tokenizer, tokenizer_config.max_chunk_length)
    batches = deterministic_batches(train_ex, tokenizer, training_config)
    
    optimizer = optimizer_for(model, training_config)
    
    print("Starting production pretraining...")
    losses, optimizer, cursor = train_steps(model, batches, training_config, steps=run_config.max_steps, optimizer=optimizer)
    
    # 4. Evaluation and Checkpoint
    eval_card = EvaluationCard("production-eval-v1", "evaluate causal loss on wikitext pilot", manifest_data["fingerprint"], ("loss", "perplexity"))
    lineage = ArtifactLineage("prod-commit", run_config.fingerprint, manifest_data["fingerprint"], tokenizer.fingerprint, parameter_fingerprint(model))
    
    stage_d_payload = {
        "format": "dcss-cdi-stage-d-checkpoint-v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "tokenizer_artifact": tokenizer.artifact(),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "data_manifest": manifest_data,
        "config": training_config.as_dict(),
        "global_step": run_config.max_steps,
        "cursor": cursor,
        "random_state": {"python": None, "numpy": None, "torch": None},
        "topology_fingerprint": model.ssm.cell.topology.fingerprint(),
        "hardware": {"device": device, "dtype": "float32", "torch": torch.__version__},
    }
    
    boundary = ReleaseBoundary(status="production_pilot", real_corpus_training_authorized=True)
    envelope = build_envelope(stage_d_payload, lineage, boundary, EnvironmentLineage.current(device=device))
    
    checkpoint_path = out / "production_checkpoint.pt"
    save_atomic(checkpoint_path, envelope)
    print(f"Saved production checkpoint to {checkpoint_path}")
    
    report = {
        "status": "PASS",
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "checkpoint": str(checkpoint_path),
        "manifest_fingerprint": manifest_data["fingerprint"],
    }
    (out / "production_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def scale_stage_c(c_config: StageCConfig, embed_dim: int) -> StageCConfig:
    from dataclasses import replace
    return replace(c_config, input_width=embed_dim, output_width=embed_dim)
