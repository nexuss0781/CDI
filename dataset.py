"""
CDI Dataset — Real Knowledge for Next-Gen LLM
===============================================

This is NOT a toy numerical regression setup.
CDI is a next-generation language model architecture.

Datasets (all from HuggingFace)
-------------------------------
**Training corpus** — ``wikitext-2-raw-v1``
    Real Wikipedia articles: biology, physics, history, mathematics.
    Teaches CDI general language structure and world knowledge.
    The belief complex learns hierarchical abstraction of text.

**Fine-tuning corpus** — ``sciq`` (SciQ)
    Real science QA: biology, chemistry, physics questions + answers.
    Teaches CDI to shape outputs for question-answering.
    The Dirac operator learns to route queries to relevant knowledge.

**Test set** — Hand-crafted science questions
    Generated manually to probe CDI's learned knowledge.
    Tests whether Hodge-theoretic inference (§5) retrieves
    the right information from the belief complex.

Architecture Mapping
--------------------
    text → HF tokenizer → token IDs → embedding[IDs] → (n_points, embed_dim)
         → CDI Engine (manifold, belief, Dirac, heat equation, Hodge)
         → (n_points, embed_dim)
         → output @ embedding.T → logits → next token prediction

Each manifold point = one token position (replaces positional encoding).
Belief connection = information flow between positions (replaces attention).
Heat equation = convergent learning (replaces gradient descent dynamics).

Complexity: O(n) encode, O(1) per-token access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════════
#  Text Dataset — sequences of token IDs
# ═══════════════════════════════════════════════════════════════════════════

class CDITextDataset(Dataset):
    """Dataset of token-ID windows for next-token prediction.

    Each sample:  X = token_ids[i : i+seq_len]      (input)
                  y = token_ids[i+1 : i+seq_len+1]  (target = shifted by 1)

    Complexity: O(1) per sample.
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int, name: str = ""):
        self.token_ids = token_ids.to(torch.long)
        self.seq_len = seq_len
        self.name = name

    def __len__(self) -> int:
        return max(len(self.token_ids) - self.seq_len - 1, 0)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (input_ids, target_ids) each of shape (seq_len,)."""
        x = self.token_ids[idx : idx + self.seq_len]
        y = self.token_ids[idx + 1 : idx + self.seq_len + 1]
        return x, y

    def __repr__(self) -> str:
        return (f"CDITextDataset(name={self.name!r}, "
                f"n_tokens={len(self.token_ids):,}, "
                f"n_windows={len(self):,}, seq_len={self.seq_len})")


# ═══════════════════════════════════════════════════════════════════════════
#  QA Dataset — question+answer tokenised
# ═══════════════════════════════════════════════════════════════════════════

class CDIQADataset(Dataset):
    """QA dataset: input = question tokens, target = answer's next tokens.

    Each sample:
        X = token_ids of "Q: {question} A:" padded to seq_len
        y = token_ids shifted by 1 (next-token prediction through answer)
    """

    def __init__(self, input_ids: torch.Tensor, target_ids: torch.Tensor, name: str = ""):
        assert input_ids.shape == target_ids.shape
        self.input_ids = input_ids.to(torch.long)
        self.target_ids = target_ids.to(torch.long)
        self.name = name

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

    def __repr__(self):
        return (f"CDIQADataset(name={self.name!r}, n={len(self)}, "
                f"seq_len={self.input_ids.shape[1]})")


# ═══════════════════════════════════════════════════════════════════════════
#  1. TRAINING — WikiText-2 (real Wikipedia knowledge)
# ═══════════════════════════════════════════════════════════════════════════

def download_wikitext(seq_len: int = 32, max_tokens: int = 200_000) -> CDITextDataset:
    """Download wikitext-2-raw-v1 from HuggingFace and tokenise.

    Contains real Wikipedia articles covering biology, physics,
    mathematics, history, geography, and more.

    Parameters
    ----------
    seq_len : int    Context window (= CDI n_points).
    max_tokens : int  Cap total tokens for current scale.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("  [TRAIN] Downloading wikitext-2-raw-v1 from HuggingFace...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                       trust_remote_code=True)

    print("  [TRAIN] Tokenising with GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Concatenate all text and tokenise
    all_text = "\n".join([row["text"] for row in ds if row["text"].strip()])
    all_ids = tokenizer.encode(all_text)

    # Cap at max_tokens
    if len(all_ids) > max_tokens:
        all_ids = all_ids[:max_tokens]

    token_tensor = torch.tensor(all_ids, dtype=torch.long)
    print(f"  [TRAIN] {len(all_ids):,} tokens → {len(all_ids) - seq_len - 1:,} windows")

    return CDITextDataset(token_tensor, seq_len, name="wikitext2_train")


# ═══════════════════════════════════════════════════════════════════════════
#  2. FINE-TUNING — SciQ (real science questions + answers)
# ═══════════════════════════════════════════════════════════════════════════

def download_sciq(
    seq_len: int = 32,
    max_samples: int = 3000,
) -> CDIQADataset:
    """Download SciQ from HuggingFace — real science QA.

    Biology, chemistry, physics, earth science questions with answers.
    Fine-tunes CDI to shape its inference for question-answering.

    Parameters
    ----------
    seq_len : int     Sequence length (= CDI n_points).
    max_samples : int  Maximum QA pairs.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("  [FINE-TUNE] Downloading allenai/sciq from HuggingFace...")
    ds = load_dataset("allenai/sciq", split="train", trust_remote_code=True)

    print("  [FINE-TUNE] Tokenising QA pairs...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    all_input = []
    all_target = []

    for i, row in enumerate(ds):
        if i >= max_samples:
            break

        question = row.get("question", "")
        answer = row.get("correct_answer", "")
        if not question or not answer:
            continue

        # Format: "Q: {question} A: {answer}"
        qa_text = f"Q: {question} A: {answer}"
        ids = tokenizer.encode(qa_text)

        # Pad or truncate to seq_len + 1 (need shifted target)
        if len(ids) < seq_len + 1:
            ids = ids + [pad_id] * (seq_len + 1 - len(ids))
        else:
            ids = ids[:seq_len + 1]

        all_input.append(ids[:seq_len])
        all_target.append(ids[1:seq_len + 1])

    input_t = torch.tensor(all_input, dtype=torch.long)
    target_t = torch.tensor(all_target, dtype=torch.long)
    print(f"  [FINE-TUNE] {len(all_input):,} QA pairs")

    return CDIQADataset(input_t, target_t, name="sciq_finetune")


# ═══════════════════════════════════════════════════════════════════════════
#  3. TEST — Hand-crafted science questions
# ═══════════════════════════════════════════════════════════════════════════

# These are real questions to probe CDI's learned knowledge.
# Generated here, not downloaded — tests generalisation.

TEST_QUESTIONS = [
    # Biology
    ("Q: What is the basic unit of life? A:", "The cell"),
    ("Q: What organelle produces energy in cells? A:", "Mitochondria"),
    ("Q: What molecule carries genetic information? A:", "DNA"),
    ("Q: What process do plants use to make food from sunlight? A:", "Photosynthesis"),
    ("Q: What is the powerhouse of the cell? A:", "The mitochondrion"),
    ("Q: What type of bond holds the two strands of DNA together? A:", "Hydrogen bonds"),
    ("Q: What is the process of cell division called? A:", "Mitosis"),
    ("Q: What organ pumps blood through the body? A:", "The heart"),
    ("Q: What gas do humans exhale? A:", "Carbon dioxide"),
    ("Q: What is the largest organ in the human body? A:", "The skin"),
    # Physics
    ("Q: What is Newton's second law of motion? A:", "Force equals mass times acceleration"),
    ("Q: What is the speed of light in vacuum? A:", "Approximately 300,000 kilometers per second"),
    ("Q: What force keeps planets in orbit? A:", "Gravity"),
    ("Q: What is the unit of electrical resistance? A:", "Ohm"),
    ("Q: What particle carries a positive charge? A:", "Proton"),
    ("Q: What is the first law of thermodynamics? A:", "Energy cannot be created or destroyed"),
    ("Q: What is the formula for kinetic energy? A:", "One half times mass times velocity squared"),
    ("Q: What phenomenon explains why the sky is blue? A:", "Rayleigh scattering"),
    # Chemistry
    ("Q: What is the chemical symbol for water? A:", "H2O"),
    ("Q: What is the pH of a neutral solution? A:", "Seven"),
    ("Q: What is the most abundant element in the universe? A:", "Hydrogen"),
    ("Q: What type of bond involves sharing electrons? A:", "Covalent bond"),
    ("Q: What is Avogadro's number? A:", "Approximately 6.022 times 10 to the 23rd"),
    # Earth Science
    ("Q: What are the three types of rocks? A:", "Igneous, sedimentary, and metamorphic"),
    ("Q: What causes tides on Earth? A:", "The gravitational pull of the Moon"),
    ("Q: What layer of the atmosphere contains the ozone layer? A:", "The stratosphere"),
    # Mathematics
    ("Q: What is the derivative of x squared? A:", "Two x"),
    ("Q: What is the value of pi to two decimal places? A:", "3.14"),
    ("Q: What is the integral of one over x? A:", "Natural logarithm of x"),
    ("Q: What is Euler's number approximately equal to? A:", "2.718"),
]


def make_test_set(seq_len: int = 32) -> CDIQADataset:
    """Generate test dataset from hand-crafted science questions.

    These questions were NOT in the training or fine-tuning data.
    They test whether CDI's belief complex actually learned knowledge.
    """
    from transformers import AutoTokenizer

    print("  [TEST] Generating test set from hand-crafted science questions...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    all_input = []
    all_target = []

    for question, answer in TEST_QUESTIONS:
        full_text = question + " " + answer
        ids = tokenizer.encode(full_text)

        if len(ids) < seq_len + 1:
            ids = ids + [pad_id] * (seq_len + 1 - len(ids))
        else:
            ids = ids[:seq_len + 1]

        all_input.append(ids[:seq_len])
        all_target.append(ids[1:seq_len + 1])

    input_t = torch.tensor(all_input, dtype=torch.long)
    target_t = torch.tensor(all_target, dtype=torch.long)
    print(f"  [TEST] {len(TEST_QUESTIONS)} science questions")

    return CDIQADataset(input_t, target_t, name="science_test")


# ═══════════════════════════════════════════════════════════════════════════
#  DataLoader factory — O(1) per batch
# ═══════════════════════════════════════════════════════════════════════════

def make_dataloader(dataset, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ═══════════════════════════════════════════════════════════════════════════
#  Prepare all
# ═══════════════════════════════════════════════════════════════════════════

def prepare_all(seq_len: int = 32, data_dir: str = "./data") -> Dict:
    """Download and prepare all datasets."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    train_ds = download_wikitext(seq_len=seq_len)
    ft_ds = download_sciq(seq_len=seq_len)
    test_ds = make_test_set(seq_len=seq_len)

    return {"train": train_ds, "finetune": ft_ds, "test": test_ds}


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  CDI Dataset — Real Knowledge for Next-Gen LLM")
    print("  Training:    wikitext-2 (Wikipedia knowledge)")
    print("  Fine-tuning: SciQ (science QA)")
    print("  Test:        Hand-crafted science questions")
    print("=" * 65)
    datasets = prepare_all(seq_len=32)
    for k, v in datasets.items():
        print(f"  {k:12s} → {v}")
    print("\n  Done.")
