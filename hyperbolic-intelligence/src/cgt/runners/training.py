"""cgt.runners.training — generic training loop.

Decoupled from any specific architecture. Experiment-specific metrics
(e.g. AngularPhysicsGate diagnostics) are injected via the
`metrics_callback` argument of `train()`.

Features
--------
- Device-aware: uses CUDA if available, CPU otherwise. Autocast on CUDA
  (bfloat16 by default; float16 optional). No autocast on CPU.
- Resumable: if `latest.pt` exists in the experiment directory, loads it
  and continues from `global_step`. All RNG streams are restored.
- NaN-resilient: skip or stop on NaN (configurable). Always checkpoints
  before stopping.
- Periodic evaluation, best-checkpoint tracking, final checkpoint.
"""
from __future__ import annotations

import math
import signal
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from .checkpoint import CheckpointManager, CheckpointMetadata
from .config     import ExperimentConfig
from .data       import make_batches
from .logger     import MetricsLogger


# ─────────────────────────────────────────────────────────────────────────────
# Metrics callback — experiment-specific diagnostics plug in here
# ─────────────────────────────────────────────────────────────────────────────

MetricsCallback = Callable[[nn.Module], Dict[str, float]]


def null_metrics(_model: nn.Module) -> Dict[str, float]:
    """Default: no architecture-specific metrics collected."""
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def _lr_at(step: int, total: int, warmup_frac: float) -> float:
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * t))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module,
             batches: List[Tuple[torch.Tensor, torch.Tensor]],
             max_batches: int,
             device: torch.device) -> float:
    model.eval()
    losses = []
    n = min(max_batches, len(batches))
    for xb, yb in batches[:n]:
        out = model(xb, labels=yb)
        losses.append(float(out["loss"].item()))
    model.train()
    if not losses:
        return float("nan")
    return float(np.mean(losses))


# ─────────────────────────────────────────────────────────────────────────────
# Graceful shutdown on SIGINT/SIGTERM
# ─────────────────────────────────────────────────────────────────────────────

class _GracefulShutdown:
    def __init__(self):
        self.requested = False
        self._orig_int  = None
        self._orig_term = None

    def __enter__(self):
        def _handler(signum, frame):
            if not self.requested:
                print(f"\n[signal {signum}] shutdown requested; will checkpoint "
                      f"and exit after current step")
            self.requested = True
        try:
            self._orig_int  = signal.signal(signal.SIGINT,  _handler)
            self._orig_term = signal.signal(signal.SIGTERM, _handler)
        except (ValueError, OSError):
            pass
        return self

    def __exit__(self, *exc):
        try:
            if self._orig_int  is not None: signal.signal(signal.SIGINT,  self._orig_int)
            if self._orig_term is not None: signal.signal(signal.SIGTERM, self._orig_term)
        except (ValueError, OSError):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    completed_steps: int
    final_val_loss:  float
    final_ppl:       float
    best_val_loss:   float
    best_val_step:   int
    wall_s:          float
    aborted:         bool = False
    reason:          str  = ""


def train(
    cfg: ExperimentConfig,
    model: nn.Module,
    device: torch.device,
    logger: MetricsLogger,
    ckpt_mgr: CheckpointManager,
    metrics_callback: MetricsCallback = null_metrics,
) -> TrainResult:
    """Generic training loop with resume, autocast, NaN handling.

    Args
    ----
    cfg              : ExperimentConfig
    model            : nn.Module whose forward takes (input_ids, labels=)
                       and returns {'loss': scalar}
    device           : cuda or cpu
    logger           : MetricsLogger
    ckpt_mgr         : CheckpointManager
    metrics_callback : optional hook called after eval forwards to collect
                       arch-specific diagnostics; returns a dict merged into
                       the log row. Must be fast and side-effect free.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(
        trainable_params, lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )

    # Resume?
    meta = CheckpointMetadata()
    if cfg.resume and ckpt_mgr.has_latest():
        logger.heartbeat("resuming", from_path=str(ckpt_mgr.latest_path))
        meta = ckpt_mgr.load(
            ckpt_mgr.latest_path, model, opt, scheduler=None,
            restore_rng=True, map_location=device,
        )
        logger.heartbeat("resumed",
                         global_step=meta.global_step,
                         best_val_loss=meta.best_val_loss)
    else:
        logger.heartbeat("started",
                         variant=cfg.variant, seed=cfg.seed,
                         total_steps=cfg.total_steps, device=str(device))

    start_step = meta.global_step
    if start_step >= cfg.total_steps:
        logger.heartbeat("already_completed", step=start_step)
        return TrainResult(
            completed_steps=start_step,
            final_val_loss=meta.best_val_loss,
            final_ppl=math.exp(meta.best_val_loss) if meta.best_val_loss < float("inf") else float("nan"),
            best_val_loss=meta.best_val_loss,
            best_val_step=meta.best_val_step,
            wall_s=meta.total_wall_s,
        )

    train_b, valid_b = make_batches(
        seed=cfg.seed, seq_len=cfg.seq_len,
        batch_size=cfg.batch_size, device=device,
    )

    use_ac = cfg.use_autocast and device.type == "cuda"
    if use_ac:
        ac_dtype = torch.bfloat16 if cfg.autocast_dtype == "bfloat16" else torch.float16
        autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=ac_dtype)
    else:
        from contextlib import nullcontext
        autocast_ctx = nullcontext

    grad_scaler = (
        torch.amp.GradScaler() if (use_ac and ac_dtype == torch.float16) else None
    )

    model.train()
    t_session = time.time()
    step = start_step
    aborted = False; abort_reason = ""
    train_iter = _skip_iter(_cycle(train_b), start_step % max(1, len(train_b)))

    with _GracefulShutdown() as shutdown:
        while step < cfg.total_steps:
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(_cycle(train_b))
                xb, yb = next(train_iter)

            lr = cfg.lr * _lr_at(step, cfg.total_steps, cfg.warmup_frac)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            try:
                with autocast_ctx():
                    out = model(xb, labels=yb)
                    loss = out["loss"]

                loss_val = float(loss.item())
                if not math.isfinite(loss_val):
                    meta.consecutive_nans += 1
                    if meta.consecutive_nans >= cfg.max_consecutive_nans:
                        abort_reason = f"{meta.consecutive_nans} consecutive NaN batches"
                        aborted = True; break
                    if cfg.nan_action == "stop":
                        abort_reason = "NaN loss (nan_action=stop)"
                        aborted = True; break
                    logger.log(step, event="nan_skip",
                               consecutive_nans=meta.consecutive_nans)
                    step += 1
                    continue
                meta.consecutive_nans = 0

                if grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
                    grad_scaler.step(opt)
                    grad_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
                    opt.step()
            except RuntimeError as e:
                meta.total_wall_s += (time.time() - t_session)
                _save(ckpt_mgr, model, opt, meta, cfg, tag="latest")
                logger.heartbeat("runtime_error", exception=str(e))
                raise

            if (step % cfg.log_every == 0) or (step + 1 == cfg.total_steps):
                extra = metrics_callback(model)
                logger.log(step, train_loss=loss_val, lr=lr, **extra)

            is_eval_step = (
                (step > 0 and step % cfg.eval_every == 0)
                or (step + 1 == cfg.total_steps)
            )
            if is_eval_step:
                val_loss = evaluate(model, valid_b, cfg.eval_batches, device)
                ppl = math.exp(val_loss) if math.isfinite(val_loss) else float("nan")
                extra = metrics_callback(model)
                logger.log(step, phase="eval",
                           train_loss=loss_val, val_loss=val_loss,
                           ppl=ppl, lr=lr, **extra)
                is_best = val_loss < meta.best_val_loss
                if is_best:
                    meta.best_val_loss = val_loss
                    meta.best_val_step = step

                meta.global_step = step + 1
                meta.total_wall_s = _accumulate_wall(meta.total_wall_s, t_session)
                t_session = time.time()
                _save(ckpt_mgr, model, opt, meta, cfg,
                      tag="latest", is_best=is_best)

            elif cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0:
                meta.global_step = step + 1
                meta.total_wall_s = _accumulate_wall(meta.total_wall_s, t_session)
                t_session = time.time()
                _save(ckpt_mgr, model, opt, meta, cfg, tag="latest")

            if shutdown.requested:
                abort_reason = "SIGINT/SIGTERM"
                aborted = True
                break

            step += 1

    meta.total_wall_s = _accumulate_wall(meta.total_wall_s, t_session)
    meta.global_step = step if aborted else cfg.total_steps

    final_val = evaluate(model, valid_b, cfg.final_eval_batches, device)
    final_ppl = math.exp(final_val) if math.isfinite(final_val) else float("nan")

    is_best_final = math.isfinite(final_val) and final_val < meta.best_val_loss
    if is_best_final:
        meta.best_val_loss = final_val
        meta.best_val_step = meta.global_step

    _save(ckpt_mgr, model, opt, meta, cfg,
          tag="latest", is_best=is_best_final)

    if aborted:
        logger.heartbeat("aborted",
                         reason=abort_reason,
                         global_step=meta.global_step,
                         wall_s=meta.total_wall_s)
    else:
        logger.heartbeat("completed",
                         global_step=meta.global_step,
                         final_ppl=final_ppl,
                         best_val_loss=meta.best_val_loss,
                         wall_s=meta.total_wall_s)

    return TrainResult(
        completed_steps = meta.global_step,
        final_val_loss  = final_val,
        final_ppl       = final_ppl,
        best_val_loss   = meta.best_val_loss,
        best_val_step   = meta.best_val_step,
        wall_s          = meta.total_wall_s,
        aborted         = aborted,
        reason          = abort_reason,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(ckpt_mgr, model, opt, meta, cfg, tag, is_best=False):
    ckpt_mgr.save(model, opt, None, meta, cfg.to_dict(),
                  is_best=is_best, tag=tag)


def _accumulate_wall(prev: float, t_session_start: float) -> float:
    return prev + (time.time() - t_session_start)


def _cycle(xs):
    while True:
        for x in xs:
            yield x


def _skip_iter(iterator, n: int):
    for _ in range(n):
        try: next(iterator)
        except StopIteration: break
    return iterator


# ─────────────────────────────────────────────────────────────────────────────
# RNG snapshot helpers (used around eval calls that consume RNG)
# ─────────────────────────────────────────────────────────────────────────────

def _capture_rng_snapshot():
    """Snapshot just torch RNG (python/numpy are not consumed by eval)."""
    import torch as _t
    snap = {"torch": _t.get_rng_state()}
    if _t.cuda.is_available():
        snap["cuda"] = _t.cuda.get_rng_state_all()
    return snap


def _restore_rng_snapshot(snap):
    import torch as _t
    _t.set_rng_state(snap["torch"])
    if "cuda" in snap and _t.cuda.is_available():
        try:
            _t.cuda.set_rng_state_all(snap["cuda"])
        except RuntimeError:
            pass
