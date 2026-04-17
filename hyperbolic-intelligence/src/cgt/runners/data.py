"""cgt.runners.data — WikiText-2 byte-level, deterministic per-seed.

Design
------
- Dataset is loaded once per process (HuggingFace `datasets` with direct
  HTTP fallback) and cached in memory as raw byte arrays.
- `make_batches(seed)` produces a deterministic batch order for a given
  seed and sequence length. Identical call → identical output, across
  Python versions, because we use `numpy.random.default_rng(seed)` for
  the permutation and `np.frombuffer` for tokenization.
- Each `train_batches` element is `(inputs, targets)` where
  targets is inputs shifted by 1 (next-byte prediction).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Raw text loader (module-level cache)
# ─────────────────────────────────────────────────────────────────────────────

_TEXT_CACHE: Optional[Tuple[str, str]] = None


def set_text_override(train_text: str, valid_text: str) -> None:
    """Testing hook: inject synthetic text, bypassing any download.

    After calling this, `load_wikitext2_raw()` returns the injected pair
    until the process dies. Used to run integration tests offline.
    """
    global _TEXT_CACHE, _IDS_CACHE
    _TEXT_CACHE = (train_text, valid_text)
    _IDS_CACHE = None      # force re-encode on next get_byte_ids()


def clear_caches() -> None:
    global _TEXT_CACHE, _IDS_CACHE
    _TEXT_CACHE = None
    _IDS_CACHE = None


def load_wikitext2_raw() -> Tuple[str, str]:
    """Return (train_text, valid_text). Cached after first call."""
    global _TEXT_CACHE
    if _TEXT_CACHE is not None:
        return _TEXT_CACHE

    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        tr = "\n".join(ds["train"]["text"])
        vl = "\n".join(ds["validation"]["text"])
    except Exception:
        import urllib.request
        URL = {
            "train": "https://raw.githubusercontent.com/pytorch/examples/"
                     "main/word_language_model/data/wikitext-2/train.txt",
            "valid": "https://raw.githubusercontent.com/pytorch/examples/"
                     "main/word_language_model/data/wikitext-2/valid.txt",
        }
        tr = urllib.request.urlopen(URL["train"]).read().decode("utf-8")
        vl = urllib.request.urlopen(URL["valid"]).read().decode("utf-8")

    _TEXT_CACHE = (tr, vl)
    return _TEXT_CACHE


def _byte_encode(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Byte-id tensor cache (module-level)
# ─────────────────────────────────────────────────────────────────────────────

_IDS_CACHE: Optional[Tuple[np.ndarray, np.ndarray]] = None


def get_byte_ids() -> Tuple[np.ndarray, np.ndarray]:
    global _IDS_CACHE
    if _IDS_CACHE is None:
        tr, vl = load_wikitext2_raw()
        _IDS_CACHE = (_byte_encode(tr), _byte_encode(vl))
    return _IDS_CACHE


# ─────────────────────────────────────────────────────────────────────────────
# Batch generator
# ─────────────────────────────────────────────────────────────────────────────

def _pack(ids: np.ndarray, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = (len(ids) - 1) // seq_len
    ids = ids[: n * seq_len + 1]
    inputs  = torch.from_numpy(ids[:-1].reshape(n, seq_len).copy())
    targets = torch.from_numpy(ids[1:].reshape(n, seq_len).copy())
    return inputs, targets


def make_batches(
    seed: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]],
           List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Deterministic (train_batches, valid_batches) for a given seed.

    Both lists are lists of (inputs, targets) on `device`, each batch of
    shape [batch_size, seq_len].
    """
    train_ids, valid_ids = get_byte_ids()

    train_in, train_tg = _pack(train_ids, seq_len)
    valid_in, valid_tg = _pack(valid_ids, seq_len)

    # Only the train set is shuffled; validation order is fixed for
    # apples-to-apples eval across runs.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(train_in.shape[0])
    train_in, train_tg = train_in[idx], train_tg[idx]

    def _batch(inp: torch.Tensor, tg: torch.Tensor):
        out = []
        for s in range(0, inp.shape[0] - batch_size + 1, batch_size):
            out.append((inp[s:s+batch_size].to(device),
                        tg [s:s+batch_size].to(device)))
        return out

    return _batch(train_in, train_tg), _batch(valid_in, valid_tg)


if __name__ == "__main__":
    # Self-test (offline path used if datasets is not installed / no network)
    import os
    # Mock the text to avoid network in sandbox
    _TEXT_CACHE = ("hello world " * 5000, "hello world " * 500)
    globals()["_TEXT_CACHE"] = _TEXT_CACHE
    _IDS_CACHE = None

    tr_b, vl_b = make_batches(seed=42, seq_len=32, batch_size=8,
                              device=torch.device("cpu"))
    print(f"train batches: {len(tr_b)}   valid batches: {len(vl_b)}")
    # Determinism: same seed → identical first batch
    tr_b2, _ = make_batches(seed=42, seq_len=32, batch_size=8,
                            device=torch.device("cpu"))
    assert torch.equal(tr_b[0][0], tr_b2[0][0]), "seed determinism broken"
    # Different seed → different first batch
    tr_b3, _ = make_batches(seed=7, seq_len=32, batch_size=8,
                            device=torch.device("cpu"))
    assert not torch.equal(tr_b[0][0], tr_b3[0][0]), "seeds too similar"
    print("✓ data module self-test passed")
