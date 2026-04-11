# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Evaluation Module
=====================

Scientific validation and evaluation metrics.

Exports
-------
FalsificationProtocols
    F1-F3 scientific validation tests.
evaluate_stsb
    STS-B benchmark evaluation.
compute_effective_rank
    Embedding dimensionality metric.
compute_gromov_delta
    Hyperbolicity estimation.
compute_distortion
    Distance preservation metric.
"""

from cgt.evaluation.metrics import (
    FalsificationProtocols,
    compute_distortion,
    compute_effective_rank,
    compute_gromov_delta,
    evaluate_stsb,
)

from cgt.evaluation.benchmarks import (
    evaluate_stsb_quick,
    evaluate_all_datasets,
    run_falsification,
    EuclideanStudent,
    eval_pca_baseline,
    eval_random_baseline,
    eval_mrl_baseline,
    train_euclidean_mlp,
    benchmark_latency,
    set_seed,
)

from cgt.evaluation.benchmarks import (
    evaluate_stsb_quick,
    evaluate_all_datasets,
    run_falsification,
    eval_pca_baseline,
    eval_random_baseline,
    eval_mrl_baseline,
    train_euclidean_mlp,
    benchmark_latency,
    EuclideanStudent,
)

__all__ = [
    "FalsificationProtocols",
    "evaluate_stsb",
    "compute_effective_rank",
    "compute_gromov_delta",
    "compute_distortion",
    "evaluate_stsb_quick",
    "evaluate_all_datasets",
    "run_falsification",
    "EuclideanStudent",
    "eval_pca_baseline",
    "eval_random_baseline",
    "eval_mrl_baseline",
    "train_euclidean_mlp",
    "benchmark_latency",
    "set_seed",
]
