"""cgt.runners.logger — append-only JSONL metrics log + console echo.

Design
------
- Line-buffered JSONL: each call to `.log()` produces one line that is
  flushed immediately. Safe against kill: the log up to the last complete
  line is always readable.
- Console echo is optional and formatted for human inspection; the JSONL
  is the canonical record.
- A small heartbeat row is written once per session so you can tell when
  a run started/resumed without grepping the training loop.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class MetricsLogger:
    """Append-only JSONL logger, one file per experiment."""

    def __init__(
        self,
        path: str | Path,
        console: bool = True,
        console_every: int = 1,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Line buffering: newline flushes to disk
        self._fp = open(self.path, "a", buffering=1, encoding="utf-8")
        self.console = console
        self.console_every = max(1, console_every)
        self._n_writes = 0

    # ── session markers ──────────────────────────────────────────────────

    def heartbeat(self, message: str, **extra: Any) -> None:
        """Write a session-level event (start, resume, end, abort)."""
        self._write({
            "event":     message,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **extra,
        }, echo=True)

    # ── step logging ─────────────────────────────────────────────────────

    def log(self, step: int, **metrics: Any) -> None:
        """Log a step-level metrics row."""
        row = {"step": step, **metrics}
        self._write(row, echo=(self._n_writes % self.console_every == 0))
        self._n_writes += 1

    # ── internals ────────────────────────────────────────────────────────

    def _write(self, row: Dict[str, Any], echo: bool = False) -> None:
        # Coerce non-JSON-safe scalars (numpy/torch) to Python floats
        row = {k: _coerce(v) for k, v in row.items()}
        line = json.dumps(row, separators=(",", ":"))
        self._fp.write(line + "\n")

        if echo and self.console:
            _print_row(row)

    def close(self) -> None:
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass

    # ── context manager ──────────────────────────────────────────────────

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            # Best-effort: record the failure
            try:
                self.heartbeat("aborted",
                               exception=f"{exc_type.__name__}: {exc}")
            except Exception:
                pass
        self.close()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coerce(v: Any) -> Any:
    """Convert numpy/torch scalars to JSON-safe Python primitives."""
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
    except ImportError:
        pass
    try:
        import torch
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return v.item()
            return v.tolist()
    except ImportError:
        pass
    if isinstance(v, float):
        if v != v:          # NaN
            return None
        if v in (float("inf"), float("-inf")):
            return None
    return v


def _print_row(row: Dict[str, Any]) -> None:
    """Short human-readable rendering for console echo."""
    if "event" in row:
        ts = row.get("timestamp", "")
        extra = " ".join(f"{k}={v}" for k, v in row.items()
                         if k not in ("event", "timestamp"))
        print(f"[{ts}] {row['event']:<10} {extra}")
        return

    step = row.get("step", "?")
    bits = [f"step={step:>5}"]
    for k in ("train_loss", "val_loss", "ppl",
              "cos_align", "gate_mean", "gate_std", "corr_attn_K", "lr"):
        v = row.get(k)
        if v is None:
            continue
        if k == "ppl":
            bits.append(f"ppl={v:7.2f}")
        elif k in ("cos_align", "corr_attn_K"):
            bits.append(f"{k}={v:+.3f}")
        elif k in ("gate_mean", "gate_std"):
            bits.append(f"{k}={v:.3f}")
        elif k == "lr":
            bits.append(f"lr={v:.2e}")
        else:
            bits.append(f"{k}={v:.4f}")
    print("  " + "  ".join(bits))


# ─────────────────────────────────────────────────────────────────────────────
# Reader utility (for aggregation)
# ─────────────────────────────────────────────────────────────────────────────

def read_jsonl(path: str | Path) -> list[dict]:
    """Read a jsonl log, skipping malformed lines (e.g. partial final line
    after a kill)."""
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "metrics.jsonl"
        with MetricsLogger(p, console=True) as lg:
            lg.heartbeat("started", variant="v9", seed=42)
            for step in range(0, 300, 50):
                lg.log(step, train_loss=5.5 - step*0.001,
                       val_loss=5.5 - step*0.0011,
                       cos_align=0.05 + step*1e-5,
                       gate_std=0.06, gate_mean=0.53,
                       ppl=245.0 - step*0.3, lr=3e-4)
            lg.heartbeat("completed", final_step=300)

        rows = read_jsonl(p)
        print(f"\nread back {len(rows)} rows; first event: {rows[0].get('event')}")
        # Check that NaN/inf coerce to null
        with MetricsLogger(p, console=False) as lg:
            lg.log(999, train_loss=float("nan"), gate_std=float("inf"))
        rows = read_jsonl(p)
        last = rows[-1]
        assert last["train_loss"] is None, f"NaN not coerced: {last}"
        assert last["gate_std"] is None, f"inf not coerced: {last}"
        print("✓ MetricsLogger self-test passed")
