"""cgt.runners.checkpoint — atomic checkpoint save/load.

Guarantees
----------
- **Atomic writes**: write to a temp path, then os.replace to the target.
  Never leaves a half-written `latest.pt` after a kill/OOM.
- **RNG state capture**: python/numpy/torch CPU/torch CUDA. Resuming
  reproduces the exact data order and dropout masks from the point of save.
- **Separate best/latest**: `latest.pt` updates every ckpt_every and
  end-of-run; `best.pt` only updates when val_loss improves.
- **Forward-compatible load**: checkpoints carry a schema version; missing
  fields fall back to safe defaults so old checkpoints still load.
"""
from __future__ import annotations

import os
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# RNG state capture
# ─────────────────────────────────────────────────────────────────────────────

def capture_rng_state() -> Dict[str, Any]:
    """Snapshot all relevant RNG streams."""
    state = {
        "python":  random.getstate(),
        "numpy":   np.random.get_state(),
        "torch":   torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG streams from a previous capture. Tolerant to missing keys."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda_all" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
        except RuntimeError:
            # Device count mismatch between save and load — not fatal
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Atomic I/O
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_save(obj: Any, path: Path) -> None:
    """Save to a tempfile in the same directory, then rename. The rename is
    atomic on POSIX for same-filesystem paths, which covers our use case."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(tmp_fd)
    try:
        torch.save(obj, tmp_name)
        os.replace(tmp_name, str(path))
    except Exception:
        if os.path.exists(tmp_name):
            try: os.remove(tmp_name)
            except OSError: pass
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckpointMetadata:
    """Minimal bookkeeping that persists across resumes."""
    global_step: int       = 0
    best_val_loss: float   = float("inf")
    best_val_step: int     = 0
    total_wall_s: float    = 0.0               # accumulated across resumes
    consecutive_nans: int  = 0
    extra: Dict[str, Any]  = field(default_factory=dict)


class CheckpointManager:
    """Manages `latest.pt` and `best.pt` for a single experiment directory."""

    def __init__(self, ckpt_dir: str | Path):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.ckpt_dir / "latest.pt"

    @property
    def best_path(self) -> Path:
        return self.ckpt_dir / "best.pt"

    def step_path(self, step: int) -> Path:
        return self.ckpt_dir / f"step_{step:06d}.pt"

    # ── save ─────────────────────────────────────────────────────────────

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        meta: CheckpointMetadata,
        config_dict: Dict[str, Any],
        is_best: bool = False,
        tag: str = "latest",
    ) -> Path:
        """Write a checkpoint. `tag` selects the target file:
            - 'latest'        → latest.pt   (always overwritten)
            - 'best'          → best.pt     (only if is_best)
            - 'step'          → step_{N}.pt (archival)
        """
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "global_step":    meta.global_step,
            "best_val_loss":  meta.best_val_loss,
            "best_val_step":  meta.best_val_step,
            "total_wall_s":   meta.total_wall_s,
            "consecutive_nans": meta.consecutive_nans,
            "extra":          meta.extra,
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict() if optimizer is not None else None,
            "scheduler":      (scheduler.state_dict() if scheduler is not None
                               and hasattr(scheduler, "state_dict") else None),
            "rng_state":      capture_rng_state(),
            "config":         config_dict,
        }
        if tag == "latest":
            target = self.latest_path
        elif tag == "best":
            target = self.best_path
        elif tag == "step":
            target = self.step_path(meta.global_step)
        else:
            raise ValueError(f"unknown tag {tag!r}")

        _atomic_save(payload, target)

        # If this is also the best-so-far, mirror to best.pt in the same call
        if is_best and tag != "best":
            _atomic_save(payload, self.best_path)

        return target

    # ── load ─────────────────────────────────────────────────────────────

    def load(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        restore_rng: bool = True,
        map_location: Optional[str | torch.device] = None,
        strict: bool = True,
    ) -> CheckpointMetadata:
        """Restore model/optimizer/scheduler/RNG from `path`. Returns metadata."""
        path = Path(path)
        payload = torch.load(path, map_location=map_location, weights_only=False)

        ver = payload.get("schema_version", 0)
        if ver > CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(
                f"checkpoint schema version {ver} is newer than supported "
                f"({CHECKPOINT_SCHEMA_VERSION}); upgrade the runner")

        model.load_state_dict(payload["model"], strict=strict)

        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])

        if scheduler is not None and payload.get("scheduler") is not None:
            try:
                scheduler.load_state_dict(payload["scheduler"])
            except Exception:
                pass   # scheduler is optional and may not be resumable

        if restore_rng and "rng_state" in payload:
            restore_rng_state(payload["rng_state"])

        return CheckpointMetadata(
            global_step      = payload.get("global_step", 0),
            best_val_loss    = payload.get("best_val_loss", float("inf")),
            best_val_step    = payload.get("best_val_step", 0),
            total_wall_s     = payload.get("total_wall_s", 0.0),
            consecutive_nans = payload.get("consecutive_nans", 0),
            extra            = payload.get("extra", {}),
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def has_latest(self) -> bool:
        return self.latest_path.exists()

    def prune_step_checkpoints(self, keep: int = 0) -> None:
        """Keep the `keep` most-recent step_XXXX.pt files; delete the rest.
        Useful to cap disk usage in long runs.
        """
        files = sorted(self.ckpt_dir.glob("step_*.pt"))
        if len(files) > keep:
            for f in files[:-keep] if keep > 0 else files:
                try: f.unlink()
                except OSError: pass


if __name__ == "__main__":
    # Self-test: save, modify, load, verify restoration
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td) / "ckpts"
        mgr = CheckpointManager(ckpt_dir)

        # Build a trivial model + optimizer
        m = torch.nn.Linear(4, 2)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        # Perturb optimizer state so it's not all zeros
        for _ in range(3):
            out = m(torch.randn(8, 4))
            out.sum().backward()
            opt.step(); opt.zero_grad()

        # Snapshot reference state
        ref_weight = m.weight.detach().clone()
        ref_opt_state = {k: v for k, v in opt.state_dict()["state"].items()}

        # Save checkpoint (captures RNG state at this point)
        meta = CheckpointMetadata(global_step=42, best_val_loss=2.1, best_val_step=30)
        mgr.save(m, opt, None, meta, {"dummy": True}, is_best=True, tag="latest")
        assert mgr.has_latest(), "latest.pt should exist"
        assert mgr.best_path.exists(), "best.pt should exist"

        # Generate RNG fingerprint from the state *at save time* — this is what
        # we expect to reproduce after restore_rng_state.
        rng_fp_1 = torch.randn(3).tolist()

        # Mutate model + advance RNG past the fingerprint
        with torch.no_grad():
            m.weight.add_(1.0)
        _ = torch.randn(100)                # advance RNG

        # Reload
        m2 = torch.nn.Linear(4, 2)
        opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
        meta_loaded = mgr.load(mgr.latest_path, m2, opt2, restore_rng=True)

        assert meta_loaded.global_step == 42
        assert abs(meta_loaded.best_val_loss - 2.1) < 1e-9
        # Weights match the pre-mutation state
        assert torch.allclose(m2.weight, ref_weight), "weight mismatch"
        # Optimizer state restored (shapes match at least)
        s2 = opt2.state_dict()["state"]
        assert set(s2.keys()) == set(ref_opt_state.keys()), "opt state keys"

        # RNG restored → same fingerprint
        rng_fp_2 = torch.randn(3).tolist()
        assert rng_fp_1 == rng_fp_2, f"RNG not restored: {rng_fp_1} vs {rng_fp_2}"

        # Atomic save corner case: simulate crash mid-save (write fails)
        # -> latest.pt should be unchanged
        class BadObj:
            def __reduce__(self): raise RuntimeError("simulated crash")
        try:
            _atomic_save(BadObj(), mgr.latest_path)
        except Exception:
            pass
        # latest.pt still exists and loads cleanly
        meta_loaded_2 = mgr.load(mgr.latest_path, m2)
        assert meta_loaded_2.global_step == 42, "atomic save broke latest.pt"

        print("✓ CheckpointManager self-test passed")
