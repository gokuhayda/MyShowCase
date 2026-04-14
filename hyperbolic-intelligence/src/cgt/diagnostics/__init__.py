"""
cgt.diagnostics
===============
Post-training diagnostic tools for hyperbolic model analysis.

Purely additive — no existing training code is modified.
Import and use independently after a training run.

Main entry point:
    from cgt.diagnostics import DegEqDiagnostics
    diag = DegEqDiagnostics(trainer, tokenizer)
    report = diag.run()
"""
from .degeq import (
    DegEqDiagnostics,
    k_equilibrium_from_zipf,
    estimate_zipf_exponent,
    freq_radius_correlation,
    radial_collapse_score,
    build_token_counts,
)

__all__ = [
    "DegEqDiagnostics",
    "k_equilibrium_from_zipf",
    "estimate_zipf_exponent",
    "freq_radius_correlation",
    "radial_collapse_score",
    "build_token_counts",
]
from cgt.diagnostics.gw_monitor import GWDivergenceMonitor
