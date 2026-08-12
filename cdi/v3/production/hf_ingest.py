"""Production Hugging Face dataset ingestion with content deduplication and HF_TOKEN support."""
from __future__ import annotations

import os
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set
import sys

from datasets import load_dataset
import requests
from huggingface_hub import login

from .data import DataManifest, GovernedDocument, P2DataPolicy


def ingest_wikitext_and_sciq(output_dir: str | Path = "data/production") -> Dict[str, Path]:
    """Download WikiText-103 and SciQ from Hugging Face with content deduplication."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    token = os.environ.get("HF_TOKEN")
    if token:
        print("[INFO] Authenticating with Hugging Face Hub using HF_TOKEN...")
        try:
            login(token=token)
        except Exception as e:
            print(f"[WARNING] HF Login failed: {e}")

    print("Verifying Hugging Face Hub connectivity...")
    try:
        r = requests.get("https://huggingface.co", timeout=10)
        print(f"Hub status: {r.status_code}")
    except Exception as e:
        print(f"WARNING: Cannot reach huggingface.co: {e}")

    print("Attempting to load real datasets from Hugging Face Hub...")
    
    wikitext = None
    sciq = None
    errors = []

    # 1. Load WikiText-103
    try:
        print("Loading Salesforce/wikitext (wikitext-103-raw-v1)...")
        wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", token=token, trust_remote_code=True)
    except Exception as e:
        errors.append(f"Primary wikitext load failed: {e}")
        try:
            print("Retrying wikitext with legacy ID...")
            wikitext = load_dataset("wikitext", "wikitext-103-raw-v1", token=token)
        except Exception as e2:
            errors.append(f"Legacy wikitext load failed: {e2}")

    # 2. Load SciQ
    try:
        print("Loading allenai/sciq...")
        sciq = load_dataset("allenai/sciq", token=token, trust_remote_code=True)
    except Exception as e:
        errors.append(f"Primary sciq load failed: {e}")
        try:
            print("Retrying sciq with legacy ID...")
            sciq = load_dataset("sciq", token=token)
        except Exception as e2:
            errors.append(f"Legacy sciq load failed: {e2}")

    if not wikitext or not sciq:
        print("\n".join(errors))
        print("CRITICAL: Could not load real datasets from Hugging Face.")
        print("Please verify your HF_TOKEN and internet connection.")
        sys.exit(1)

    # Use a set to track content hashes for deduplication across the entire ingestion
    seen_hashes: Set[str] = set()

    def get_unique_docs(dataset_split, prefix: str, limit: int, data_source: str, license_info: str, usage: str) -> List[GovernedDocument]:
        docs = []
        for idx, item in enumerate(dataset_split):
            if "text" in item:
                text = item["text"].strip()
            elif "question" in item and "correct_answer" in item:
                text = f"Q: {item['question'].strip()} A: {item['correct_answer'].strip()}"
            else:
                continue
                
            if len(text) > 50:
                content_hash = sha256(text.encode("utf-8")).hexdigest()
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    doc_id = f"{prefix}-{idx:06d}"
                    docs.append(GovernedDocument(doc_id, text, data_source, license_info, usage, data_class="rights_cleared_pilot", pii_review="reviewed_no_pii"))
                    if len(docs) >= limit:
                        break
        return docs

    print("Ingesting WikiText-103 documents...")
    train_docs = get_unique_docs(wikitext["train"], "wt-train", 50000, "hf://Salesforce/wikitext", "CC-BY-SA-4.0", "retained_for_pretraining")
    val_docs = get_unique_docs(wikitext["validation"], "wt-val", 5000, "hf://Salesforce/wikitext", "CC-BY-SA-4.0", "retained_for_validation")

    print("Ingesting SciQ documents...")
    finetune_docs = get_unique_docs(sciq["train"], "sciq-train", 10000, "hf://allenai/sciq", "MIT", "retained_for_finetuning")

    all_docs = train_docs + val_docs + finetune_docs
    policy = P2DataPolicy()
    
    val_split_ids = [d.identifier for d in val_docs[:len(val_docs)//2]]
    test_split_ids = [d.identifier for d in val_docs[len(val_docs)//2:]]
    
    manifest = DataManifest.build(
        all_docs,
        {
            "train": [d.identifier for d in train_docs + finetune_docs],
            "validation": val_split_ids,
            "test": test_split_ids,
        },
        policy=policy,
    )
    
    manifest_path = out / "production_corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    
    text_path = out / "production_corpus_text.jsonl"
    with text_path.open("w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps({"id": doc.identifier, "text": doc.text}) + "\n")
            
    print(f"Manifest written to {manifest_path} with fingerprint {manifest.fingerprint}")
    return {"manifest": manifest_path, "text_jsonl": text_path}
