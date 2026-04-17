"""cgt.runners.runner — single-experiment orchestrator.

The runner is architecture-agnostic. v9-specific wiring lives in a small
registry that maps variant names → (model_builder, metrics_callback).
Adding v10 later is a single registry entry.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .checkpoint import CheckpointManager
from .config     import ExperimentConfig
from .logger     import MetricsLogger
from .training   import MetricsCallback, TrainResult, null_metrics, train


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

def setup_env(cgt_root: Optional[str] = None) -> torch.device:
    """Global numeric policy + sys.path. Returns the device."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cgt_root is not None and cgt_root not in sys.path:
        sys.path.insert(0, cgt_root)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Variant registry
# ─────────────────────────────────────────────────────────────────────────────

ModelBuilder = Callable[[ExperimentConfig, torch.device], nn.Module]

_MODEL_BUILDERS: Dict[str, ModelBuilder] = {}
_METRICS_CALLBACKS: Dict[str, MetricsCallback] = {}


def register_variant(name: str,
                     builder: ModelBuilder,
                     metrics: Optional[MetricsCallback] = None) -> None:
    """Register a (variant_name → builder, metrics_callback) entry.

    Use this to add new architectures without modifying the runner.
    """
    _MODEL_BUILDERS[name] = builder
    _METRICS_CALLBACKS[name] = metrics or null_metrics


def known_variants() -> list[str]:
    return sorted(_MODEL_BUILDERS.keys())


def _get_builder(variant: str) -> ModelBuilder:
    if variant not in _MODEL_BUILDERS:
        raise KeyError(
            f"unknown variant {variant!r}; registered: {known_variants()}. "
            f"Did you call register_default_v9_variants()?"
        )
    return _MODEL_BUILDERS[variant]


def _get_metrics(variant: str) -> MetricsCallback:
    return _METRICS_CALLBACKS.get(variant, null_metrics)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in: v9 registration (call once per process)
# ─────────────────────────────────────────────────────────────────────────────

def register_default_v9_variants() -> None:
    """Register v7 + v9 + ablation variants using cgt.models.*. Import-time
    dependencies on cgt.* are deferred here so the runner module itself
    remains importable even without cgt on the path."""
    from cgt.models.transformer_v2        import (
        HyperbolicTransformerV2, HyperbolicTransformerConfigV2,
    )
    from cgt.models.angular_physics       import AngularPhysicsConfig
    from cgt.models.angular_physics_layer import upgrade_transformer_to_v9
    from .metrics_v9 import collect_v9_gate_stats

    def _base(cfg: ExperimentConfig) -> HyperbolicTransformerV2:
        torch.manual_seed(cfg.seed)
        mc = cfg.model
        hconfig = HyperbolicTransformerConfigV2(
            vocab_size          = mc.vocab_size,
            n_embd              = mc.n_embd,
            n_layer             = mc.n_layer,
            n_head              = mc.n_head,
            n_positions         = mc.n_positions,
            tie_word_embeddings = mc.tie_word_embeddings,
            learnable_curvature = mc.learnable_curvature,
            dropout             = mc.dropout,
            attention_dropout   = mc.attention_dropout,
        )
        return HyperbolicTransformerV2(hconfig)

    def build_v7(cfg, device):
        return _base(cfg).to(device)

    def build_v9(cfg, device):
        m = _base(cfg)
        upgrade_transformer_to_v9(
            m, angular_gate_config=AngularPhysicsConfig(
                detach_attn_for_alignment=True))
        return m.to(device)

    def build_v9_no_detach(cfg, device):
        m = _base(cfg)
        upgrade_transformer_to_v9(
            m, angular_gate_config=AngularPhysicsConfig(
                detach_attn_for_alignment=False))
        return m.to(device)

    def build_v9_gate_off(cfg, device):
        m = _base(cfg)
        upgrade_transformer_to_v9(
            m, angular_gate_config=AngularPhysicsConfig(
                detach_attn_for_alignment=True))
        # Force gate ≡ ~1.0 and freeze its parameters.
        with torch.no_grad():
            for layer in m.encoder.layers:
                layer.angular_gate.W.zero_()
                layer.angular_gate.b.fill_(20.0)
        for layer in m.encoder.layers:
            layer.angular_gate.W.requires_grad_(False)
            layer.angular_gate.b.requires_grad_(False)
        return m.to(device)

    register_variant("v7",            build_v7,           metrics=null_metrics)
    register_variant("v9",            build_v9,           metrics=collect_v9_gate_stats)
    register_variant("v9_no_detach",  build_v9_no_detach, metrics=collect_v9_gate_stats)
    register_variant("v9_gate_off",   build_v9_gate_off,  metrics=collect_v9_gate_stats)


# ─────────────────────────────────────────────────────────────────────────────
# Single-experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def _summarize(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{total:,} params (trainable: {trainable:,}, frozen: {total-trainable:,})"


def run_experiment(
    cfg: ExperimentConfig,
    device: Optional[torch.device] = None,
    cgt_root: Optional[str] = None,
    verbose: bool = True,
) -> TrainResult:
    """Run one (variant, seed) experiment end-to-end.

    Requires that the variant was registered via `register_variant` or
    `register_default_v9_variants`. The runner calls the builder and the
    matching metrics callback automatically.
    """
    if device is None:
        device = setup_env(cgt_root=cgt_root)
    elif cgt_root is not None and cgt_root not in sys.path:
        sys.path.insert(0, cgt_root)

    # Auto-register v9 variants if none registered yet
    if not _MODEL_BUILDERS:
        register_default_v9_variants()

    builder  = _get_builder(cfg.variant)
    metrics  = _get_metrics(cfg.variant)

    cfg.ensure_dirs()
    cfg.save()

    ckpt_mgr = CheckpointManager(cfg.ckpt_dir)
    logger   = MetricsLogger(cfg.log_path, console=verbose,
                              console_every=max(1, cfg.log_every // 5))

    if verbose:
        print(f"\n{'═' * 72}")
        print(f"  experiment: {cfg.experiment_name}")
        print(f"  variant   : {cfg.variant}  |  seed: {cfg.seed}  |  mode: {cfg.run_mode}")
        print(f"  steps     : {cfg.total_steps}  |  eval every: {cfg.eval_every}  |  "
              f"ckpt every: {cfg.ckpt_every}")
        print(f"  device    : {device}  |  autocast: "
              f"{cfg.autocast_dtype if cfg.use_autocast and device.type == 'cuda' else 'off'}")
        print(f"  exp dir   : {cfg.exp_dir}")
        print(f"{'═' * 72}")

    with logger:
        model = builder(cfg, device)
        if verbose:
            print(f"  {cfg.variant}: {_summarize(model)}")

        result = train(
            cfg=cfg, model=model, device=device,
            logger=logger, ckpt_mgr=ckpt_mgr,
            metrics_callback=metrics,
        )

    if verbose:
        print(f"\n  → completed_steps : {result.completed_steps}")
        print(f"  → final PPL       : {result.final_ppl:.3f}")
        print(f"  → best val loss   : {result.best_val_loss:.4f}  (step {result.best_val_step})")
        print(f"  → wall time       : {result.wall_s / 60:.2f} min")
        if result.aborted:
            print(f"  → aborted         : {result.reason}")

    return result
