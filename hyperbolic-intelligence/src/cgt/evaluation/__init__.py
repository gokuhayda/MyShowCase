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

__all__ = [
    "FalsificationProtocols",
    "evaluate_stsb",
    "compute_effective_rank",
    "compute_gromov_delta",
    "compute_distortion",
]
