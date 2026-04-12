# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Thesis Falsification Protocol (F1–F4)
======================================

Implements the four-test falsification protocol defined in the
doctoral project (PPGC/UFRGS) Section 5.5.

These tests are designed to DISPROVE (not prove) that a hyperbolic
embedding preserves geometric and structural properties. Each test
has a quantitative threshold; failure on any test falsifies the
corresponding property claim.

Protocol Definition
-------------------
F1 — Minkowski Constraint Fidelity
    Test: max |⟨x, x⟩_L + 1/K| across all embeddings
    Pass: violation < 10^{-5}
    Falsifies: "representations lie on the Lorentz hyperboloid"

F2 — Geodesic Distance Preservation
    Test: Spearman ρ between teacher pairwise distances and
          student geodesic distances
    Pass: ρ ≥ 0.8
    Falsifies: "embedding preserves distance ordering"

F3 — Neighborhood Preservation (k-NN Overlap)
    Test: fraction of k-nearest neighbors preserved between
          teacher (Euclidean) and student (geodesic) spaces
    Pass: overlap ≥ 50%
    Falsifies: "embedding preserves local structure"

F4 — Radial Hierarchy Preservation
    Test: Pearson ρ between log-token-frequency and geodesic radius
    Pass: ρ < -0.1 (anti-correlation: frequent → origin, rare → boundary)
    Falsifies: "embedding preserves frequency→radius hierarchy"
    Note: F4 failure indicates DegEq (Degenerate Equilibrium)

Relationship to existing code:
    - cgt/evaluation/metrics.py has DIFFERENT F1–F3 (homotopy, stability,
      Forman-Ricci) — those are the CGT research falsification tests.
    - This module implements the THESIS-SPECIFIC F1–F4 defined for the
      doctoral committee evaluation.
    - F4 reuses freq_radius_correlation from cgt/diagnostics/degeq.py.

Design: ADDITIVE ONLY. New module, no modifications to existing code.

Usage:
    from cgt.evaluation.thesis_falsification import ThesisFalsificationSuite

    suite = ThesisFalsificationSuite(substrate=substrate)
    report = suite.run(
        student_embeddings=student_emb,    # [N, D+1] on H^n
        teacher_embeddings=teacher_emb,    # [N, D_teacher] in R^n
        token_counts=token_counts,         # [V] for F4
    )
    print(report.summary())
    # → F1: PASS (violation=2.3e-07)
    #   F2: PASS (ρ=0.84)
    #   F3: PASS (overlap=0.62)
    #   F4: FAIL (ρ=-0.03, DegEq active)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds (from doctoral project Section 5.5)
# ─────────────────────────────────────────────────────────────────────────────

F1_THRESHOLD = 1e-5      # Minkowski violation
F2_THRESHOLD = 0.8       # Spearman ρ (distance preservation)
F3_THRESHOLD = 0.50      # k-NN overlap fraction
F4_THRESHOLD = -0.1      # Pearson ρ (frequency-radius, must be negative)


# ─────────────────────────────────────────────────────────────────────────────
# Individual Tests
# ─────────────────────────────────────────────────────────────────────────────

def f1_minkowski_fidelity(
    embeddings: torch.Tensor,
    K: float = 1.0,
) -> Dict[str, float]:
    """
    F1 — Minkowski Constraint Fidelity.

    Tests whether embeddings satisfy ⟨x, x⟩_L = -1/K on the
    Lorentz hyperboloid.

    Args:
        embeddings: [N, D+1] points in ambient Minkowski space.
            x[..., 0] = time coordinate, x[..., 1:] = spatial.
        K: Curvature parameter (default 1.0).

    Returns:
        Dict with max_violation, mean_violation, passed (bool).
    """
    x = embeddings.double()
    # ⟨x, x⟩_L = -x_0² + Σ x_i²
    time_sq = x[..., 0:1].pow(2)
    space_sq = x[..., 1:].pow(2).sum(dim=-1, keepdim=True)
    inner = space_sq - time_sq  # Should equal -1/K

    target = -1.0 / K
    violation = (inner - target).abs()

    max_viol = violation.max().item()
    mean_viol = violation.mean().item()

    return {
        "max_violation": max_viol,
        "mean_violation": mean_viol,
        "threshold": F1_THRESHOLD,
        "passed": max_viol < F1_THRESHOLD,
    }


def f2_distance_preservation(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    substrate=None,
    n_samples: int = 500,
    K: float = 1.0,
) -> Dict[str, float]:
    """
    F2 — Geodesic Distance Preservation.

    Computes Spearman rank correlation between teacher pairwise
    Euclidean distances and student pairwise geodesic distances.

    Args:
        student_embeddings: [N, D+1] on H^n (Lorentz ambient).
        teacher_embeddings: [N, D_t] in R^n (Euclidean).
        substrate: LorentzSubstrate (optional, for geodesic dist).
        n_samples: Number of pairs to sample (for efficiency).
        K: Curvature parameter.

    Returns:
        Dict with spearman_rho, p_value, passed (bool).
    """
    N = min(student_embeddings.shape[0], teacher_embeddings.shape[0])
    s_emb = student_embeddings[:N].double()
    t_emb = teacher_embeddings[:N].float()

    # Sample pairs
    max_pairs = min(n_samples, N * (N - 1) // 2)
    idx_i = torch.randint(0, N, (max_pairs,))
    idx_j = torch.randint(0, N, (max_pairs,))
    # Avoid self-pairs
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    if idx_i.numel() < 10:
        return {"spearman_rho": 0.0, "p_value": 1.0,
                "threshold": F2_THRESHOLD, "passed": False}

    # Teacher: Euclidean distances
    t_dists = (t_emb[idx_i] - t_emb[idx_j]).norm(dim=-1)

    # Student: geodesic distances on H^n
    if substrate is not None:
        s_dists = substrate.dist(s_emb[idx_i], s_emb[idx_j])
    else:
        # Manual Lorentz distance: d = (1/√K) * arccosh(-K * ⟨x, y⟩_L)
        x = s_emb[idx_i]
        y = s_emb[idx_j]
        inner = (x[..., 1:] * y[..., 1:]).sum(-1) - x[..., 0] * y[..., 0]
        clamped = (-K * inner).clamp(min=1.0 + 1e-7)
        s_dists = torch.acosh(clamped) / (K ** 0.5)

    s_dists = s_dists.float()

    # Spearman rank correlation
    def _rank(t: torch.Tensor) -> torch.Tensor:
        _, order = t.sort()
        ranks = torch.zeros_like(t)
        ranks[order] = torch.arange(len(t), dtype=t.dtype, device=t.device)
        return ranks

    r_t = _rank(t_dists)
    r_s = _rank(s_dists)
    n = r_t.numel()
    d = r_t - r_s
    rho = 1.0 - 6.0 * (d * d).sum().item() / (n * (n * n - 1))

    return {
        "spearman_rho": round(rho, 4),
        "n_pairs": n,
        "threshold": F2_THRESHOLD,
        "passed": rho >= F2_THRESHOLD,
    }


def f3_knn_overlap(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    k: int = 10,
    substrate=None,
    n_samples: int = 200,
    K: float = 1.0,
) -> Dict[str, float]:
    """
    F3 — Neighborhood Preservation (k-NN Overlap).

    For each of n_samples anchor points, computes the k-nearest
    neighbors in both teacher (Euclidean) and student (geodesic) spaces.
    The overlap is the fraction of neighbors that appear in both sets.

    Args:
        student_embeddings: [N, D+1] on H^n.
        teacher_embeddings: [N, D_t] in R^n.
        k: Number of nearest neighbors.
        substrate: LorentzSubstrate (optional).
        n_samples: Number of anchor points to test.
        K: Curvature parameter.

    Returns:
        Dict with mean_overlap, std_overlap, passed (bool).
    """
    N = min(student_embeddings.shape[0], teacher_embeddings.shape[0])
    s_emb = student_embeddings[:N].double()
    t_emb = teacher_embeddings[:N].float()
    n_samples = min(n_samples, N)
    k = min(k, N - 1)

    if k < 1 or N < 3:
        return {"mean_overlap": 0.0, "std_overlap": 0.0,
                "threshold": F3_THRESHOLD, "passed": False}

    anchors = torch.randperm(N)[:n_samples]
    overlaps = []

    for anchor_idx in anchors:
        # Teacher k-NN (Euclidean)
        t_dists = (t_emb - t_emb[anchor_idx].unsqueeze(0)).norm(dim=-1)
        t_dists[anchor_idx] = float('inf')
        _, t_knn = t_dists.topk(k, largest=False)

        # Student k-NN (geodesic)
        s_anchor = s_emb[anchor_idx].unsqueeze(0).expand(N, -1)
        if substrate is not None:
            s_dists = substrate.dist(s_anchor, s_emb)
        else:
            inner = (s_anchor[..., 1:] * s_emb[..., 1:]).sum(-1) - \
                     s_anchor[..., 0] * s_emb[..., 0]
            clamped = (-K * inner).clamp(min=1.0 + 1e-7)
            s_dists = torch.acosh(clamped) / (K ** 0.5)
        s_dists = s_dists.float()
        s_dists[anchor_idx] = float('inf')
        _, s_knn = s_dists.topk(k, largest=False)

        # Overlap
        t_set = set(t_knn.tolist())
        s_set = set(s_knn.tolist())
        overlap = len(t_set & s_set) / k
        overlaps.append(overlap)

    overlaps_t = torch.tensor(overlaps)
    mean_overlap = overlaps_t.mean().item()
    std_overlap = overlaps_t.std().item()

    return {
        "mean_overlap": round(mean_overlap, 4),
        "std_overlap": round(std_overlap, 4),
        "k": k,
        "n_samples": n_samples,
        "threshold": F3_THRESHOLD,
        "passed": mean_overlap >= F3_THRESHOLD,
    }


def f4_radial_hierarchy(
    embeddings: torch.Tensor,
    token_counts: torch.Tensor,
) -> Dict[str, float]:
    """
    F4 — Radial Hierarchy Preservation.

    Tests whether the embedding preserves the frequency→radius
    anti-correlation: frequent tokens near the origin, rare tokens
    near the boundary. Failure indicates DegEq.

    Reuses the logic from cgt.diagnostics.degeq.freq_radius_correlation
    but with the thesis-specific threshold.

    Args:
        embeddings: [V, D+1] or [V, D] token embeddings.
        token_counts: [V] per-token occurrence counts.

    Returns:
        Dict with pearson_rho, hierarchy_intact, passed (bool).
    """
    emb = embeddings.float()
    tc = token_counts.float()

    V = min(emb.shape[0], tc.shape[0])
    emb = emb[:V]
    tc = tc[:V]

    # Radii
    if emb.shape[-1] > 16:
        radii = emb[:, 1:].norm(dim=-1)  # ambient: skip time
    else:
        radii = emb.norm(dim=-1)

    # Filter tokens with at least 1 occurrence
    valid = tc >= 1
    radii = radii[valid]
    log_f = torch.log(tc[valid].clamp(min=1.0))

    if radii.numel() < 10:
        return {"pearson_rho": 0.0, "threshold": F4_THRESHOLD, "passed": False}

    # Pearson correlation
    lf_m = log_f.mean()
    r_m = radii.mean()
    cov = ((log_f - lf_m) * (radii - r_m)).mean()
    std_lf = log_f.std().clamp(min=1e-8)
    std_r = radii.std().clamp(min=1e-8)
    rho = (cov / (std_lf * std_r)).item()

    return {
        "pearson_rho": round(rho, 4),
        "mean_radius": round(radii.mean().item(), 4),
        "std_radius": round(radii.std().item(), 4),
        "n_tokens": radii.numel(),
        "threshold": F4_THRESHOLD,
        "passed": rho < F4_THRESHOLD,  # Must be negative
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unified Suite
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FalsificationReport:
    """Report from the thesis F1–F4 falsification suite."""
    f1: Dict[str, float] = field(default_factory=dict)
    f2: Dict[str, float] = field(default_factory=dict)
    f3: Dict[str, float] = field(default_factory=dict)
    f4: Dict[str, float] = field(default_factory=dict)

    all_passed: bool = False
    n_passed: int = 0
    n_tests: int = 0
    interpretation: str = ""

    def summary(self) -> str:
        def _status(d: Dict) -> str:
            if not d:
                return "SKIP"
            return "PASS ✅" if d.get("passed", False) else "FAIL ❌"

        def _detail(d: Dict, key: str) -> str:
            if not d:
                return ""
            return f"({key}={d.get(key, '?')})"

        lines = [
            "═══ Thesis Falsification Protocol (F1–F4) ═══",
            f"  F1 Minkowski fidelity  : {_status(self.f1)} {_detail(self.f1, 'max_violation')}",
            f"  F2 Distance preserv.   : {_status(self.f2)} {_detail(self.f2, 'spearman_rho')}",
            f"  F3 k-NN overlap        : {_status(self.f3)} {_detail(self.f3, 'mean_overlap')}",
            f"  F4 Radial hierarchy    : {_status(self.f4)} {_detail(self.f4, 'pearson_rho')}",
            f"  ─────────────────────────",
            f"  Result: {self.n_passed}/{self.n_tests} passed",
        ]
        if not self.all_passed:
            lines.append(f"  ⚠ {self.interpretation}")
        else:
            lines.append("  ✅ All properties validated (not falsified)")
        lines.append("═══════════════════════════════════════════════")
        return "\n".join(lines)


class ThesisFalsificationSuite:
    """
    Unified F1–F4 falsification suite for the doctoral project.

    Runs all four tests and produces a consolidated report.

    Args:
        substrate: LorentzSubstrate/LorentzSubstrateV2 instance.
        K: Manifold curvature (default 1.0, overridden by substrate if provided).
        knn_k: k for F3 neighborhood test.
        f2_n_samples: Number of pairs for F2 distance test.
        f3_n_samples: Number of anchors for F3 k-NN test.
    """

    def __init__(
        self,
        substrate=None,
        K: float = 1.0,
        knn_k: int = 10,
        f2_n_samples: int = 500,
        f3_n_samples: int = 200,
    ):
        self.substrate = substrate
        if substrate is not None and hasattr(substrate, 'K'):
            K_val = substrate.K
            self.K = K_val.item() if hasattr(K_val, 'item') else float(K_val)
        else:
            self.K = K
        self.knn_k = knn_k
        self.f2_n_samples = f2_n_samples
        self.f3_n_samples = f3_n_samples

    def run(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: Optional[torch.Tensor] = None,
        token_counts: Optional[torch.Tensor] = None,
        skip: Optional[list] = None,
    ) -> FalsificationReport:
        """
        Run the full F1–F4 protocol.

        Args:
            student_embeddings: [N, D+1] student embeddings on H^n.
            teacher_embeddings: [N, D_t] teacher embeddings in R^n.
                Required for F2 and F3. If None, F2/F3 are skipped.
            token_counts: [V] token occurrence counts.
                Required for F4. If None, F4 is skipped.
            skip: List of tests to skip, e.g. ['F2', 'F4'].

        Returns:
            FalsificationReport with all results.
        """
        skip = set(s.upper() for s in (skip or []))
        report = FalsificationReport()

        # F1 — always runnable (needs only student embeddings)
        if 'F1' not in skip:
            report.f1 = f1_minkowski_fidelity(student_embeddings, K=self.K)

        # F2 — needs teacher embeddings
        if 'F2' not in skip and teacher_embeddings is not None:
            report.f2 = f2_distance_preservation(
                student_embeddings, teacher_embeddings,
                substrate=self.substrate,
                n_samples=self.f2_n_samples,
                K=self.K,
            )

        # F3 — needs teacher embeddings
        if 'F3' not in skip and teacher_embeddings is not None:
            report.f3 = f3_knn_overlap(
                student_embeddings, teacher_embeddings,
                k=self.knn_k,
                substrate=self.substrate,
                n_samples=self.f3_n_samples,
                K=self.K,
            )

        # F4 — needs token counts
        if 'F4' not in skip and token_counts is not None:
            report.f4 = f4_radial_hierarchy(student_embeddings, token_counts)

        # Consolidate
        tests = [report.f1, report.f2, report.f3, report.f4]
        active = [t for t in tests if t]
        report.n_tests = len(active)
        report.n_passed = sum(1 for t in active if t.get("passed", False))
        report.all_passed = report.n_passed == report.n_tests

        # Interpretation
        failures = []
        if report.f1 and not report.f1.get("passed"):
            failures.append("F1: embeddings OFF manifold")
        if report.f2 and not report.f2.get("passed"):
            failures.append("F2: distance ordering NOT preserved")
        if report.f3 and not report.f3.get("passed"):
            failures.append("F3: local neighborhoods DISRUPTED")
        if report.f4 and not report.f4.get("passed"):
            failures.append("F4: radial hierarchy COLLAPSED (DegEq)")
        report.interpretation = "; ".join(failures) if failures else "All tests passed"

        return report
