"""
cgt/distillation/dataset_v2.py
================================
WikiTextTokenDataset — replaces the broken 10-sentence TEXTS dataset.

ROOT CAUSE FIXED HERE
----------------------
The original TEXTS dataset had only 10 unique sentences repeated 20x.
With SEQ_LEN=32, the model memorized in ~80 epochs (PPL → 1.05),
then produced incoherent output for any out-of-distribution input.

PATCH v2: HuggingFace 504 timeout resilience + local text fallback
-------------------------------------------------------------------
The HuggingFace Hub can time out in Colab with flaky connectivity.
Fixes applied:
    1. Retry logic (3 attempts, 5s backoff) around load_dataset()
    2. Local text fallback: if all HF attempts fail, download the raw
       WikiText-2 .tokens files directly from the official mirror and
       parse them manually — no HF API call needed.
    3. Explicit dataset_name="wikitext-2-raw-v1" (prevents accidental
       wikitext-103 resolution).

Usage
-----
    from cgt.distillation.dataset_v2 import build_wikitext_loaders
    train_loader, val_loader = build_wikitext_loaders(
        tokenizer, seq_len=128, batch_size=16
    )
"""

from __future__ import annotations

import time
import os
import urllib.request
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# WikiText-2 raw text mirror (official S3, no auth required)
# Used as fallback if HuggingFace Hub is unreachable
# ---------------------------------------------------------------------------
_WIKITEXT2_URLS = {
    "train":      "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "validation": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    "valid":      "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    "test":       "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
}

# Backup mirror (Hugging Face raw files, no API)
_WIKITEXT2_HF_RAW = {
    "train":      "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet",
}


def _load_wikitext_hf(split: str, retries: int = 3, backoff: float = 5.0) -> str:
    """Load WikiText-2 via HuggingFace datasets with retry logic."""
    from datasets import load_dataset

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            print(f"  [HF] Loading wikitext-2-raw-v1 split={split!r} (attempt {attempt}/{retries})")
            ds = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split=split,
                trust_remote_code=False,
            )
            text = "\n".join(
                line.strip()
                for line in ds["text"]
                if line.strip() and len(line.strip()) > 10
            )
            print(f"  [HF] OK — {len(text):,} chars")
            return text
        except Exception as e:
            last_exc = e
            print(f"  [HF] Attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < retries:
                print(f"  [HF] Retrying in {backoff}s...")
                time.sleep(backoff)

    raise last_exc


def _load_wikitext_raw(split: str) -> str:
    """Load WikiText-2 directly from raw text files (no HF API)."""
    # Normalize split name
    split_key = "valid" if split == "validation" else split
    if split_key not in _WIKITEXT2_URLS:
        raise ValueError(f"Unknown split: {split!r}. Choose from train/validation/test.")

    url = _WIKITEXT2_URLS[split_key]
    cache_path = f"/tmp/wikitext2_{split_key}.txt"

    if not os.path.exists(cache_path):
        print(f"  [Fallback] Downloading {url}")
        try:
            urllib.request.urlretrieve(url, cache_path)
            print(f"  [Fallback] Saved to {cache_path}")
        except Exception as e:
            raise RuntimeError(
                f"Could not download WikiText-2 from {url}: {e}\n"
                "Check your internet connection or upload the data manually."
            ) from e

    with open(cache_path, encoding="utf-8") as f:
        lines = f.readlines()

    text = "\n".join(
        line.strip()
        for line in lines
        if line.strip() and len(line.strip()) > 10
    )
    print(f"  [Fallback] Loaded {len(text):,} chars from {cache_path}")
    return text


def _load_text(split: str) -> str:
    """Try HF first; fall back to direct download if HF fails."""
    try:
        return _load_wikitext_hf(split)
    except Exception as hf_err:
        print(f"\n  [Warning] HuggingFace load failed: {hf_err}")
        print("  [Warning] Falling back to direct text download...")
        return _load_wikitext_raw(split)


class WikiTextTokenDataset(Dataset):
    """
    Sliding-window token dataset built from WikiText-2.

    Args:
        tokenizer   : HuggingFace tokenizer (e.g. GPT2Tokenizer)
        seq_len     : length of each token window (default 128)
        split       : 'train', 'validation', or 'test'
        overlap     : overlap between consecutive windows (default 64)
        min_len     : skip windows shorter than this after tokenization
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 128,
        split: str = "train",
        overlap: int = 64,
        min_len: int = 16,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
        self.overlap   = overlap
        self.min_len   = min_len

        # Load text with retry + fallback
        text = _load_text(split)

        # Tokenize the entire corpus in one shot
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )
        token_ids = encoded["input_ids"].squeeze(0)  # [N_total_tokens]

        # Build sliding windows
        stride = seq_len - overlap
        self.samples: List[dict] = []

        total = token_ids.shape[0]
        for start in range(0, total - seq_len + 1, stride):
            end   = start + seq_len
            ids   = token_ids[start:end]

            if ids.shape[0] < min_len:
                continue

            if ids.shape[0] < seq_len:
                pad_len = seq_len - ids.shape[0]
                pad_tok = tokenizer.eos_token_id or 0
                ids = torch.cat([ids, torch.full((pad_len,), pad_tok, dtype=ids.dtype)])

            lbl = ids.clone()
            if tokenizer.pad_token_id is not None:
                lbl[lbl == tokenizer.pad_token_id] = -100

            self.samples.append({"input_ids": ids, "labels": lbl})

        if len(self.samples) == 0:
            raise RuntimeError(
                f"WikiTextTokenDataset: no samples from split={split!r}. "
                "Check that the dataset downloaded correctly."
            )

        print(f"  WikiText-2 [{split}]: {len(self.samples):,} windows "
              f"(seq_len={seq_len}, overlap={overlap})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def build_wikitext_loaders(
    tokenizer,
    seq_len: int = 128,
    batch_size: int = 16,
    overlap: int = 64,
    val_fraction: float = 0.05,   # unused — kept for API compat
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from WikiText-2.

    Includes retry logic and direct-download fallback for HuggingFace timeouts.
    """
    print("Loading WikiText-2 train split...")
    train_ds = WikiTextTokenDataset(
        tokenizer, seq_len=seq_len, split="train", overlap=overlap
    )
    print("Loading WikiText-2 validation split...")
    val_ds = WikiTextTokenDataset(
        tokenizer, seq_len=seq_len, split="validation", overlap=overlap
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\n✅ Dataset ready:")
    print(f"   train: {len(train_ds):,} windows | val: {len(val_ds):,} windows")
    print(f"   seq_len={seq_len}  batch_size={batch_size}  overlap={overlap}")

    return train_loader, val_loader
