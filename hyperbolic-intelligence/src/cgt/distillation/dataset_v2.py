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


# Persistent cache: Drive → local /tmp copy (avoids re-download every session)
_DRIVE_WIKITEXT_DIR = "/content/drive/MyDrive/HydraPaper_VariantF/data/wikitext2"
_LOCAL_CACHE_DIR    = "/tmp/wikitext2_cache"


def _get_wikitext_cache_path(split_key: str) -> tuple:
    """
    Return (drive_path, local_path) for a WikiText-2 split.

    Drive path: persistent across Colab sessions.
    Local path: fast /tmp copy used at runtime.
    """
    os.makedirs(_LOCAL_CACHE_DIR, exist_ok=True)
    drive_path = os.path.join(_DRIVE_WIKITEXT_DIR, f"wikitext2_{split_key}.txt")
    local_path = os.path.join(_LOCAL_CACHE_DIR,    f"wikitext2_{split_key}.txt")
    return drive_path, local_path


def _load_wikitext_raw(split: str) -> str:
    """
    Load WikiText-2 with 3-tier cache:
      1. Local /tmp cache  (fastest — already copied this session)
      2. Drive cache       (fast  — copy to /tmp then use)
      3. Download + save   (slow  — save to Drive AND /tmp for next time)
    """
    split_key  = "valid" if split == "validation" else split
    if split_key not in _WIKITEXT2_URLS:
        raise ValueError(f"Unknown split: {split!r}. Choose from train/validation/test.")

    drive_path, local_path = _get_wikitext_cache_path(split_key)

    # ── Tier 1: local /tmp cache ───────────────────────────────────────────
    if os.path.exists(local_path):
        print(f"  [Cache] /tmp hit → {local_path}")
        with open(local_path, encoding="utf-8") as f:
            lines = f.readlines()
        text = "\n".join(l.strip() for l in lines if l.strip() and len(l.strip()) > 10)
        print(f"  [Cache] Loaded {len(text):,} chars")
        return text

    # ── Tier 2: Drive cache → copy to /tmp ────────────────────────────────
    if os.path.exists(drive_path):
        print(f"  [Cache] Drive hit → {drive_path}")
        import shutil
        shutil.copy2(drive_path, local_path)
        print(f"  [Cache] Copied to /tmp")
        with open(local_path, encoding="utf-8") as f:
            lines = f.readlines()
        text = "\n".join(l.strip() for l in lines if l.strip() and len(l.strip()) > 10)
        print(f"  [Cache] Loaded {len(text):,} chars from Drive cache")
        return text

    # ── Tier 3: download → save to Drive + /tmp ───────────────────────────
    url = _WIKITEXT2_URLS[split_key]
    print(f"  [Fallback] Downloading {url}")
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"  [Fallback] Saved to {local_path}")
    except Exception as e:
        raise RuntimeError(
            f"Could not download WikiText-2 from {url}: {e}\n"
            "Check your internet connection or upload the data manually."
        ) from e

    # Persist to Drive for next session
    try:
        os.makedirs(_DRIVE_WIKITEXT_DIR, exist_ok=True)
        import shutil
        shutil.copy2(local_path, drive_path)
        print(f"  [Cache] Saved to Drive → {drive_path}")
    except Exception as _drive_err:
        print(f"  [Cache] Drive save skipped: {_drive_err}")  # non-critical

    with open(local_path, encoding="utf-8") as f:
        lines = f.readlines()
    text = "\n".join(l.strip() for l in lines if l.strip() and len(l.strip()) > 10)
    print(f"  [Fallback] Loaded {len(text):,} chars")
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


def build_openwebtext_loaders(
    tokenizer,
    seq_len:    int = 1024,
    batch_size: int = 8,
    overlap:    int = 512,
    num_workers:int = 2,
    max_train_samples: int = 100_000,   # ~100M tokens — safe for Colab free (~30GB disk)
) -> "tuple[DataLoader, DataLoader]":
    """
    OpenWebText DataLoader for competitive HyDRA training.

    ~38GB of Reddit-curated web text — same distribution as GPT-2 training data.
    Caches to Drive automatically on first load.

    Args:
        max_train_samples: cap training samples to control run time.
            500_000 × 1024 tokens = ~512M tokens ≈ 3h on T4.
            Use None for full dataset (~38B tokens).
    """
    from datasets import load_dataset
    import os

    _DRIVE_CACHE = "/content/drive/MyDrive/HydraPaper_VariantF/data/openwebtext"
    _LOCAL_CACHE = "/tmp/openwebtext_cache"

    print(f"  [OpenWebText] Loading dataset...")

    # Try local cache first, then Drive, then download
    for cache_path in [_LOCAL_CACHE, _DRIVE_CACHE]:
        if os.path.isdir(cache_path) and os.listdir(cache_path):
            print(f"  [OpenWebText] Cache hit: {cache_path}")
            try:
                ds = load_dataset("openwebtext", split="train",
                                  cache_dir=cache_path, trust_remote_code=False)
                break
            except Exception:
                pass
    else:
        print(f"  [OpenWebText] Downloading (~12GB compressed)...")
        os.makedirs(_LOCAL_CACHE, exist_ok=True)
        ds = load_dataset("openwebtext", split="train",
                          cache_dir=_LOCAL_CACHE, trust_remote_code=False)
        # Save to Drive
        try:
            import shutil
            os.makedirs(_DRIVE_CACHE, exist_ok=True)
            if not os.listdir(_DRIVE_CACHE):
                shutil.copytree(_LOCAL_CACHE, _DRIVE_CACHE, dirs_exist_ok=True)
                print(f"  [OpenWebText] Saved to Drive: {_DRIVE_CACHE}")
        except Exception as _e:
            print(f"  [OpenWebText] Drive save skipped: {_e}")

    # Cap training samples
    if max_train_samples and len(ds) > max_train_samples:
        ds = ds.select(range(max_train_samples))
        print(f"  [OpenWebText] Capped to {max_train_samples:,} samples")

    print(f"  [OpenWebText] {len(ds):,} documents")

    # Tokenise in-memory (no disk cache) — avoids /tmp disk exhaustion on Colab
    # Streams through documents and builds flat token array directly
    from torch.utils.data import DataLoader
    import torch as _torch

    all_ids = []
    _eos = tokenizer.eos_token_id
    _report_every = max(1, len(ds) // 10)
    print(f"  [OpenWebText] Tokenising {len(ds):,} docs in-memory...")
    for _i, _doc in enumerate(ds):
        _enc = tokenizer(_doc["text"], truncation=False,
                         padding=False, return_attention_mask=False)
        all_ids.extend(_enc["input_ids"])
        all_ids.append(_eos)
        if (_i + 1) % _report_every == 0:
            print(f"  [OpenWebText]   {_i+1:,}/{len(ds):,}  "
                  f"tokens so far: {len(all_ids):,}")

    import torch
    all_ids = torch.tensor(all_ids, dtype=torch.long)

    # Split 95/5 train/val
    split = int(len(all_ids) * 0.95)
    train_ids, val_ids = all_ids[:split], all_ids[split:]

    stride = seq_len - overlap
    def _make_samples(ids):
        samples = []
        for i in range(0, len(ids) - seq_len, stride):
            chunk = ids[i:i + seq_len]
            samples.append({"input_ids": chunk, "labels": chunk})
        return samples

    train_samples = _make_samples(train_ids)
    val_samples   = _make_samples(val_ids)
    print(f"  [OpenWebText] {len(train_samples):,} train / {len(val_samples):,} val windows")

    from torch.utils.data import Dataset as _DS
    class _TokDataset(_DS):
        def __init__(self, s): self.s = s
        def __len__(self): return len(self.s)
        def __getitem__(self, i): return self.s[i]

    train_loader = DataLoader(_TokDataset(train_samples), batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(_TokDataset(val_samples),   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

