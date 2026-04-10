"""
cgt/diagnostics/degeq.py
========================
Post-training diagnostic suite for Degenerate Equilibrium (DegEq) analysis.

**Purely additive — zero modifications to existing training code.**

This module provides:
  - Krioukov curvature-Zipf equilibrium analysis
  - Frequency-radius correlation (Khrulkov/Yang criterion)
  - Radial collapse scoring
  - DegEqDiagnostics: unified post-training report

References
----------
  Krioukov et al. (2010). Hyperbolic geometry of complex networks.
    Physical Review E 82:036106. DOI:10.1103/PhysRevE.82.036106

  Khrulkov et al. (2020). Hyperbolic Image Embeddings. CVPR.

  Yang et al. (2025). HypLoRA. NeurIPS 2025.

Usage
-----
    # After training completes:
    from cgt.diagnostics import DegEqDiagnostics

    diag = DegEqDiagnostics(
        trainer     = trainer,          # DistillationTrainerV2 instance
        tokenizer   = tokenizer,        # HuggingFace tokenizer
        token_counts= token_counts_tensor,  # optional [V] counts
        device      = DEVICE,
    )
    report = diag.run(verbose=True)
    # => dict with rho, K*, gamma, risk level, radial collapse score, etc.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Krioukov equilibrium analysis
# ─────────────────────────────────────────────────────────────────────────────

def k_equilibrium_from_zipf(gamma: float, eps: float = 1e-6) -> float:
    """
    Equilibrium Lorentz curvature K* for a Zipf corpus with exponent gamma.

    From Krioukov et al. (2010, Eq. 10):
        gamma = 1 + 1 / (2*zeta),   K = -zeta^2   =>  |K*| = 1/(4*(gamma-1)^2)

    Interpretation:
      gamma=1.0  →  K* → inf  (perfect Zipf needs infinite curvature)
      gamma=1.5  →  K* = 1.0  (matches K=1 default)
      gamma=2.0  →  K* = 0.25 (flat distribution, less curvature)

    A model trained with fixed K=1 on a corpus with gamma < 1.5 is in geometric
    mismatch with its data — the curvature is too low for the vocabulary hierarchy.

    Args:
        gamma:  Zipf exponent, typically 1.0–2.0 for natural language.
        eps:    Floor for (gamma-1) to prevent division by zero.

    Returns:
        K* (float, positive).
    """
    delta = max(gamma - 1.0, eps)
    return 1.0 / (4.0 * delta ** 2)


def estimate_zipf_exponent(
    token_counts: torch.Tensor,
    min_count: int = 5,
) -> float:
    """
    Estimate Zipf exponent gamma from observed token counts via OLS on
    log-rank vs log-count.

    Args:
        token_counts:  [V] tensor of token occurrence counts (any dtype).
        min_count:     Exclude tokens with fewer than min_count occurrences.

    Returns:
        gamma (float): |slope| of the log-log regression.
    """
    counts = token_counts.float()
    counts = counts[counts >= min_count]
    if counts.numel() < 10:
        warnings.warn("[DegEqDiag] Too few tokens for Zipf estimation — using gamma=1.0")
        return 1.0

    sorted_c, _ = counts.sort(descending=True)
    ranks  = torch.arange(1, sorted_c.numel() + 1, dtype=torch.float32)
    log_r  = torch.log(ranks)
    log_c  = torch.log(sorted_c)

    lr_m = log_r.mean(); lc_m = log_c.mean()
    cov  = ((log_r - lr_m) * (log_c - lc_m)).sum()
    var  = ((log_r - lr_m) ** 2).sum()
    return abs((cov / var).item())


# ─────────────────────────────────────────────────────────────────────────────
# Radial structure analysis (Khrulkov criterion)
# ─────────────────────────────────────────────────────────────────────────────

def freq_radius_correlation(
    embeddings: torch.Tensor,
    token_counts: torch.Tensor,
) -> Dict[str, float]:
    """
    Pearson correlation between log-token-frequency and embedding radius.

    From Khrulkov et al. (2020) and Yang et al. (2025):
      rho < -0.1  →  hierarchy preserved (frequent tokens near origin)
      rho ≈  0.0  →  DegEq: radial structure collapsed

    Args:
        embeddings:   [V, d] or [V, d+1] tensor.
                      If last dim > 16, assumed ambient [V, n+1] → radius = ||x[1:]||
                      Otherwise tangent [V, n] → radius = ||x||
        token_counts: [V] token occurrence counts.

    Returns:
        Dict with: rho, p_value_approx, mean_radius, std_radius,
                   hierarchy_intact (bool), interpretation (str).
    """
    emb = embeddings.float()
    tc  = token_counts.float()

    # Align vocab sizes
    min_v = min(emb.shape[0], tc.shape[0])
    emb   = emb[:min_v]
    tc    = tc[:min_v]

    # Compute radii
    if emb.shape[-1] > 16:
        radii = emb[:, 1:].norm(dim=-1)   # ambient: skip time coordinate
    else:
        radii = emb.norm(dim=-1)

    # Pearson correlation on log-frequency vs radius
    log_f  = torch.log(tc.clamp(min=1.0))
    lf_m   = log_f.mean();  r_m = radii.mean()
    cov    = ((log_f - lf_m) * (radii - r_m)).mean()
    std_lf = log_f.std().clamp(min=1e-8)
    std_r  = radii.std().clamp(min=1e-8)
    rho    = (cov / (std_lf * std_r)).item()

    # Approximate p-value (t-distribution, df = n-2)
    n   = min_v
    t   = rho * math.sqrt((n - 2) / max(1 - rho**2, 1e-8))
    # Rough p-value approximation (not scipy, just indicative)
    p_approx = 2.0 * math.exp(-0.717 * abs(t) - 0.416 * t**2) if abs(t) < 20 else 0.0

    if rho < -0.3:
        interp = "Strong hierarchy (frequent→origin, rare→boundary)"
    elif rho < -0.1:
        interp = "Weak hierarchy (partially preserved)"
    elif abs(rho) < 0.1:
        interp = "DegEq detected: radial structure collapsed"
    else:
        interp = "Inverted hierarchy (unexpected)"

    return {
        "rho":              round(rho, 4),
        "p_value_approx":   round(p_approx, 6),
        "mean_radius":      round(radii.mean().item(), 4),
        "std_radius":       round(radii.std().item(), 4),
        "min_radius":       round(radii.min().item(), 4),
        "max_radius":       round(radii.max().item(), 4),
        "hierarchy_intact": rho < -0.1,
        "interpretation":   interp,
    }


def radial_collapse_score(
    embeddings: torch.Tensor,
    expected_radius: float = 1.5,
) -> Dict[str, float]:
    """
    Measure how collapsed the radial distribution is.

    DegEq drives all embeddings to a near-uniform radius (rdc*≈10 means
    all tokens sit at roughly the same distance from the origin).
    The collapse score quantifies this: 0 = fully collapsed, 1 = healthy spread.

    Args:
        embeddings:      [V, d] or [V, d+1] embedding matrix.
        expected_radius: Target radius (default 1.5 from OTED anchor).

    Returns:
        Dict with: collapse_score, cv (coefficient of variation),
                   mean_dev_from_target, degeq_active (bool).
    """
    emb = embeddings.float()
    if emb.shape[-1] > 16:
        radii = emb[:, 1:].norm(dim=-1)
    else:
        radii = emb.norm(dim=-1)

    mean_r = radii.mean().item()
    std_r  = radii.std().item()
    cv     = std_r / max(mean_r, 1e-8)              # coefficient of variation
    dev    = abs(mean_r - expected_radius) / expected_radius

    # Collapse score: low std + uniform radius = collapsed
    collapse_score = min(1.0, cv * 5.0)             # 0=collapsed, 1=healthy

    return {
        "collapse_score":      round(collapse_score, 4),
        "cv":                  round(cv, 4),
        "mean_radius":         round(mean_r, 4),
        "std_radius":          round(std_r, 4),
        "mean_dev_from_target":round(dev, 4),
        "degeq_active":        cv < 0.05,            # very uniform = DegEq
    }


# ─────────────────────────────────────────────────────────────────────────────
# DegEqDiagnostics — unified post-training report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DegEqReport:
    """
    Complete DegEq diagnostic report.

    Fields set after calling DegEqDiagnostics.run():
      rho_freq_radius:  Pearson r(log f_k, r_k). < -0.1 = hierarchy intact.
      k_equilibrium:    K* from Krioukov (2010) for the measured Zipf exponent.
      k_model:          Curvature K of the trained model (from optimizer or config).
      k_mismatch:       |K_model - K*|. > 5 = HIGH mismatch.
      gamma_zipf:       Estimated Zipf exponent from token_counts.
      rdc_star:         Final rdc_ema value from val_hist (DegEq attractor).
      collapse_score:   Radial collapse score (0=collapsed, 1=healthy).
      degeq_risk:       "HIGH" | "MEDIUM" | "LOW"
      hierarchy_intact: bool
      interpretation:   Human-readable summary.
    """
    rho_freq_radius:  Optional[float] = None
    k_equilibrium:    Optional[float] = None
    k_model:          float           = 1.0
    k_mismatch:       Optional[float] = None
    gamma_zipf:       Optional[float] = None
    rdc_star:         Optional[float] = None
    collapse_score:   Optional[float] = None
    cv_radius:        Optional[float] = None
    degeq_risk:       str             = "UNKNOWN"
    hierarchy_intact: bool            = False
    interpretation:   str             = ""
    raw:              Dict            = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "═══ DegEq Diagnostic Report ═══",
            f"  rho(log f, r)   : {self.rho_freq_radius}  {'✅ hierarchy intact' if self.hierarchy_intact else '❌ collapsed'}",
            f"  gamma_Zipf      : {self.gamma_zipf}",
            f"  K* (Krioukov)   : {self.k_equilibrium}",
            f"  K_model         : {self.k_model}",
            f"  K mismatch      : {self.k_mismatch}  (risk: {self.degeq_risk})",
            f"  rdc*            : {self.rdc_star}",
            f"  collapse_score  : {self.collapse_score}  (cv={self.cv_radius})",
            f"  → {self.interpretation}",
            "═══════════════════════════════",
        ]
        return "\n".join(lines)


class DegEqDiagnostics:
    """
    Post-training DegEq diagnostics. Reads from a completed trainer and produces
    a DegEqReport without modifying any training state.

    Usage:
        diag   = DegEqDiagnostics(trainer, tokenizer, token_counts=counts)
        report = diag.run(verbose=True)
        print(report.summary())
    """

    def __init__(
        self,
        trainer,
        tokenizer,
        token_counts: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            trainer:      Completed DistillationTrainerV2 instance.
            tokenizer:    HuggingFace tokenizer (or None).
            token_counts: [V] int/float tensor of per-token occurrence counts.
                          Build with: counts[tok_id] += 1 during corpus iteration.
                          If None, only geometry metrics are computed.
            device:       Computation device. Defaults to trainer's device.
        """
        self.trainer      = trainer
        self.tokenizer    = tokenizer
        self.token_counts = token_counts
        self.device       = device or getattr(trainer, 'device',
                                              torch.device('cpu'))

    def _get_embeddings(self) -> Optional[torch.Tensor]:
        """Extract embedding weight matrix from the student model."""
        student = self.trainer.student
        # Strategy: try known attribute names at depth 0 and 1
        for path in [
            ['embed'],
            ['embedding'],
            ['wte'],
            ['embed_tokens'],
            ['core_model', 'embed'],
            ['core_model', 'embedding'],
            ['core_model', 'wte'],
            ['core_model', 'token_embedding'],
        ]:
            obj = student
            for part in path:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and isinstance(obj, nn.Embedding):
                return obj.weight.detach().to(self.device).float()

        # Fallback: scan all modules for the largest Embedding
        best = None
        for _, module in student.named_modules():
            if isinstance(module, nn.Embedding):
                if best is None or module.weight.shape[0] > best.shape[0]:
                    best = module.weight
        return best.detach().to(self.device).float() if best is not None else None

    def _get_k_model(self) -> float:
        """Get curvature K from the model (substrate or learnable parameter)."""
        student = self.trainer.student
        # 1. From loss_fn substrate
        substrate = getattr(getattr(self.trainer, 'loss_fn', None), 'substrate', None)
        if substrate is not None:
            K = getattr(substrate, 'K', None)
            if K is not None:
                return K.item() if hasattr(K, 'item') else float(K)
        # 2. From learnable log_K in model
        for _, module in student.named_modules():
            if hasattr(module, 'log_K') and module.log_K is not None:
                return torch.exp(module.log_K).item()
        return 1.0  # default

    def _get_rdc_star(self) -> Optional[float]:
        """Get rdc* from val_hist (mean of last 5 evals)."""
        val_hist = getattr(self.trainer, 'val_hist', [])
        rdcs = [float(r['rdc_ema']) for r in val_hist
                if r.get('rdc_ema') not in (None, '')]
        if not rdcs:
            return None
        return round(sum(rdcs[-5:]) / min(5, len(rdcs)), 2)

    def run(self, verbose: bool = True) -> DegEqReport:
        """
        Run all diagnostics and return a DegEqReport.

        Args:
            verbose: If True, print report to stdout.

        Returns:
            DegEqReport with all computed fields.
        """
        report = DegEqReport()
        report.k_model = self._get_k_model()
        report.rdc_star = self._get_rdc_star()

        emb = self._get_embeddings()
        if emb is None:
            report.interpretation = "Could not find embedding layer"
            report.degeq_risk = "UNKNOWN"
            if verbose:
                print(report.summary())
            return report

        # Radial collapse (no token counts needed)
        collapse = radial_collapse_score(emb)
        report.collapse_score = collapse["collapse_score"]
        report.cv_radius      = collapse["cv"]
        report.raw["collapse"] = collapse

        if self.token_counts is not None:
            tc = self.token_counts.to(self.device)

            # Freq-radius correlation
            frc = freq_radius_correlation(emb, tc)
            report.rho_freq_radius = frc["rho"]
            report.hierarchy_intact = frc["hierarchy_intact"]
            report.raw["freq_radius"] = frc

            # Zipf exponent + Krioukov K*
            report.gamma_zipf    = round(estimate_zipf_exponent(tc), 4)
            report.k_equilibrium = round(k_equilibrium_from_zipf(report.gamma_zipf), 4)
            report.k_mismatch    = round(abs(report.k_model - report.k_equilibrium), 4)

            # Risk assessment
            if report.k_mismatch > 5 or (report.rho_freq_radius is not None and
               abs(report.rho_freq_radius) < 0.05):
                report.degeq_risk = "HIGH"
            elif report.k_mismatch > 1:
                report.degeq_risk = "MEDIUM"
            else:
                report.degeq_risk = "LOW"
        else:
            # Risk from collapse score only
            report.degeq_risk = "HIGH" if collapse["degeq_active"] else \
                                "MEDIUM" if report.cv_radius < 0.1 else "LOW"

        # Human-readable interpretation
        parts = []
        if report.rho_freq_radius is not None:
            if report.hierarchy_intact:
                parts.append(f"radial hierarchy preserved (ρ={report.rho_freq_radius})")
            else:
                parts.append(f"radial hierarchy COLLAPSED (ρ={report.rho_freq_radius}≈0)")
        if report.k_mismatch is not None and report.k_mismatch > 5:
            parts.append(
                f"curvature mismatch K={report.k_model} vs K*={report.k_equilibrium} "
                f"(γ={report.gamma_zipf}) — geometric mismatch is DegEq root cause"
            )
        if report.rdc_star is not None and report.rdc_star > 8:
            parts.append(f"rdc*={report.rdc_star} confirms DegEq attractor active")
        report.interpretation = "; ".join(parts) if parts else "Insufficient data"

        if verbose:
            print(report.summary())

        return report


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build token_counts from a DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def build_token_counts(
    dataloader,
    vocab_size: int,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Count token occurrences from a DataLoader to enable Zipf estimation.

    Args:
        dataloader:   PyTorch DataLoader yielding dicts with 'input_ids'.
        vocab_size:   Vocabulary size (e.g. 50257 for GPT-2).
        max_batches:  If set, stop after this many batches (for speed).
        device:       Device for accumulation.

    Returns:
        counts: [vocab_size] int64 tensor.

    Usage:
        counts = build_token_counts(train_loader, VOCAB_SIZE, max_batches=100)
        diag   = DegEqDiagnostics(trainer, tokenizer, token_counts=counts)
        report = diag.run()
    """
    dev    = device or torch.device('cpu')
    counts = torch.zeros(vocab_size, dtype=torch.long, device=dev)
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        ids = batch.get('input_ids')
        if ids is None:
            continue
        ids = ids.to(dev).reshape(-1)
        valid = ids[(ids >= 0) & (ids < vocab_size)]
        counts.scatter_add_(0, valid, torch.ones_like(valid))
    return counts
