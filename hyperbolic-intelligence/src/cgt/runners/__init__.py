"""cgt.runners — research-grade training system.

Generic training loop with resume, autocast, NaN handling, multi-seed
support, and persistent JSONL logging. Architecture-specific logic lives
in pluggable metrics callbacks and a variant registry.

Public entry points
-------------------
    from cgt.runners import (
        ExperimentConfig, default_experiment_grid,
        setup_env, register_default_v9_variants, run_experiment,
        summarize_experiment, aggregate_by_variant, print_final_report,
    )

Example
-------
    setup_env()                         # cuda/cpu + TF32 off
    register_default_v9_variants()      # v7, v9, v9_no_detach, v9_gate_off
    cfg = ExperimentConfig(variant="v9", seed=42, run_mode="LONG")
    result = run_experiment(cfg)
"""
from .config       import (
    ExperimentConfig, ModelConfig, VARIANTS, RUN_MODES,
    default_experiment_grid,
)
from .checkpoint   import CheckpointManager, CheckpointMetadata
from .logger       import MetricsLogger, read_jsonl
from .data         import (
    make_batches, load_wikitext2_raw,
    set_text_override, clear_caches,
)
from .training     import (
    train, TrainResult, evaluate, MetricsCallback, null_metrics,
)
from .runner       import (
    setup_env, run_experiment,
    register_variant, register_default_v9_variants, known_variants,
)
from .metrics_v9   import collect_v9_gate_stats
from .aggregate    import (
    ExperimentSummary, VariantStats,
    summarize_experiment, aggregate_by_variant,
    print_final_report, save_report,
)

__all__ = [
    # config
    "ExperimentConfig", "ModelConfig", "VARIANTS", "RUN_MODES",
    "default_experiment_grid",
    # checkpoint
    "CheckpointManager", "CheckpointMetadata",
    # logger
    "MetricsLogger", "read_jsonl",
    # data
    "make_batches", "load_wikitext2_raw", "set_text_override", "clear_caches",
    # training
    "train", "TrainResult", "evaluate", "MetricsCallback", "null_metrics",
    # runner
    "setup_env", "run_experiment",
    "register_variant", "register_default_v9_variants", "known_variants",
    # v9 metrics
    "collect_v9_gate_stats",
    # aggregate
    "ExperimentSummary", "VariantStats",
    "summarize_experiment", "aggregate_by_variant",
    "print_final_report", "save_report",
]
