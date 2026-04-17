"""cgt.runners.config — centralized, serializable configuration.

Design rules
------------
- Everything that affects a training run lives in `ExperimentConfig`.
- Configs are round-trippable to JSON (reviewer reproducibility).
- Run modes (SHORT/MID/LONG) set step budgets but leave all other
  hyperparameters identical across modes, so comparisons are valid.
- Variant flags encode which v9 ablation is being run.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Run-scale presets
# ─────────────────────────────────────────────────────────────────────────────

RUN_MODES = {
    "SHORT":  dict(total_steps=300,    eval_every=25,   log_every=10,  ckpt_every=150),
    "MID":    dict(total_steps=1000,   eval_every=50,   log_every=25,  ckpt_every=250),
    "LONG":   dict(total_steps=3000,   eval_every=100,  log_every=50,  ckpt_every=500),
    "XLONG":  dict(total_steps=10000,  eval_every=250,  log_every=100, ckpt_every=1000),
}


# ─────────────────────────────────────────────────────────────────────────────
# Variant / Model
# ─────────────────────────────────────────────────────────────────────────────

VARIANTS = ("v7", "v9", "v9_no_detach", "v9_gate_off")


@dataclass
class ModelConfig:
    """HLLM-paper-matched scale: ~329K params."""
    vocab_size: int           = 256
    n_embd: int               = 64
    n_layer: int              = 6
    n_head: int               = 4
    n_positions: int          = 128
    tie_word_embeddings: bool = True
    learnable_curvature: bool = False
    dropout: float            = 0.0
    attention_dropout: float  = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Experiment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    # Identity
    variant: str              = "v9"           # one of VARIANTS
    seed: int                 = 42
    run_mode: str             = "LONG"         # key of RUN_MODES

    # Training
    total_steps: int          = 3000
    eval_every: int           = 100
    log_every: int            = 50
    ckpt_every: int           = 500
    batch_size: int           = 16
    seq_len: int              = 128
    lr: float                 = 3e-4
    weight_decay: float       = 0.0
    warmup_frac: float        = 0.10
    grad_clip: float          = 1.0
    adam_beta1: float         = 0.9
    adam_beta2: float         = 0.95
    eval_batches: int         = 20              # validation minibatches per eval
    final_eval_batches: int   = 40              # at end-of-run

    # Numerics
    use_autocast: bool        = True            # enabled when cuda is available
    autocast_dtype: str       = "bfloat16"      # 'bfloat16' or 'float16'
    nan_action: str           = "skip"          # 'skip' or 'stop'
    max_consecutive_nans: int = 5               # stop after this many consecutive NaN batches

    # Model
    model: ModelConfig        = field(default_factory=ModelConfig)

    # I/O
    experiment_root: str      = "experiments"
    experiment_name: Optional[str] = None      # if None → auto: "{variant}_seed{seed}"
    resume: bool              = True            # resume from latest.pt if present

    # ── derived ──────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        assert self.variant in VARIANTS, f"unknown variant {self.variant!r}; expected one of {VARIANTS}"
        assert self.run_mode in RUN_MODES, f"unknown run_mode {self.run_mode!r}"
        assert self.autocast_dtype in ("bfloat16", "float16")
        assert self.nan_action in ("skip", "stop")

        # Apply run_mode preset (only if fields still at their defaults — user
        # overrides always win).
        preset = RUN_MODES[self.run_mode]
        if self.total_steps == 3000 and self.run_mode != "LONG":
            self.total_steps = preset["total_steps"]
        if self.eval_every == 100 and preset["eval_every"] != 100:
            self.eval_every = preset["eval_every"]
        if self.log_every == 50 and preset["log_every"] != 50:
            self.log_every = preset["log_every"]
        if self.ckpt_every == 500 and preset["ckpt_every"] != 500:
            self.ckpt_every = preset["ckpt_every"]

        if self.experiment_name is None:
            self.experiment_name = f"{self.variant}_seed{self.seed}_{self.run_mode.lower()}"

    # ── paths ────────────────────────────────────────────────────────────
    @property
    def exp_dir(self) -> Path:
        return Path(self.experiment_root) / self.experiment_name

    @property
    def ckpt_dir(self) -> Path:
        return self.exp_dir / "checkpoints"

    @property
    def log_path(self) -> Path:
        return self.exp_dir / "logs" / "metrics.jsonl"

    @property
    def config_path(self) -> Path:
        return self.exp_dir / "config.json"

    def ensure_dirs(self) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── serialization ────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Paths handled by properties; not stored
        return d

    def save(self) -> Path:
        self.ensure_dirs()
        with open(self.config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        return self.config_path

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            d = json.load(f)
        # Reconstruct nested ModelConfig
        model_d = d.pop("model", {})
        mc = ModelConfig(**model_d)
        # Drop any keys that are no longer fields (forward-compat)
        valid = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in valid}
        return cls(model=mc, **d)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-canned experiment grid
# ─────────────────────────────────────────────────────────────────────────────

def default_experiment_grid(
    seeds=(42, 7, 123),
    run_mode: str = "LONG",
    variants=VARIANTS,
    experiment_root: str = "experiments",
) -> list[ExperimentConfig]:
    """Produce the standard (variant × seed) experiment grid for the paper."""
    configs = []
    for variant in variants:
        for seed in seeds:
            configs.append(ExperimentConfig(
                variant=variant, seed=seed, run_mode=run_mode,
                experiment_root=experiment_root,
            ))
    return configs


if __name__ == "__main__":
    # Self-test: round-trip a few configs
    import tempfile, os
    cfg = ExperimentConfig(variant="v9", seed=42, run_mode="SHORT")
    print(f"exp name  : {cfg.experiment_name}")
    print(f"total steps: {cfg.total_steps}  eval_every: {cfg.eval_every}")
    print(f"exp_dir   : {cfg.exp_dir}")

    with tempfile.TemporaryDirectory() as td:
        cfg.experiment_root = td
        cfg.save()
        loaded = ExperimentConfig.load(cfg.config_path)
        assert loaded.to_dict() == cfg.to_dict(), "round-trip mismatch"
        print("✓ round-trip OK")

    grid = default_experiment_grid(seeds=(42, 7, 123), run_mode="SHORT")
    print(f"grid size: {len(grid)} experiments ({len(set(c.variant for c in grid))} variants × "
          f"{len(set(c.seed for c in grid))} seeds)")
