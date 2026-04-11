# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
benchmarks.py
=============
Reusable evaluation and baseline utilities for CGT experiments.

Contents
--------
- evaluate_stsb_quick:     Fast validation during training (single dataset)
- evaluate_all_datasets:   Full eval across N STS datasets
- run_falsification:       F1/F2/F3 geometric falsification protocol
- EuclideanStudent:        Baseline MLP projector (no hyperbolic geometry)
- eval_pca_baseline:       PCA compression baseline
- eval_random_baseline:    Random projection baseline
- eval_mrl_baseline:       Matryoshka truncation baseline
- train_euclidean_mlp:     Train Euclidean MLP baseline
- benchmark_latency:       Forward pass latency measurement

Author: Éric Gustavo Reis de Sena
Date: April 2026
"""

from __future__ import annotations

import gc
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

try:
    from sklearn.decomposition import PCA
    from sklearn.random_projection import GaussianRandomProjection
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


# ═══════════════════════════════════════════════════════════════════════════
# SEED UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════
# QUICK VALIDATION (for use inside training loops)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_stsb_quick(
    model: nn.Module,
    substrate: nn.Module,
    teacher_name: str,
    eval_data: Dict,
    device: str = 'cuda',
    dataset_key: str = 'STSBenchmark',
) -> float:
    """
    Quick single-dataset validation. Returns Spearman ρ.

    Args:
        model: CGT student model
        substrate: LorentzSubstrate for geodesic similarity
        teacher_name: HuggingFace model name for encoding
        eval_data: Dict with dataset_key containing 's1', 's2', 'scores'
        device: torch device
        dataset_key: Which dataset to evaluate on

    Returns:
        Spearman correlation (float)
    """
    assert HAS_ST, "sentence-transformers required"

    ds = eval_data[dataset_key]
    teacher = SentenceTransformer(teacher_name)
    teacher_dim = teacher.get_sentence_embedding_dimension()

    model.eval()
    with torch.no_grad():
        s1_emb = teacher.encode(ds['s1'], batch_size=512,
                                convert_to_tensor=True, device=device)
        s2_emb = teacher.encode(ds['s2'], batch_size=512,
                                convert_to_tensor=True, device=device)
        h1 = model(s1_emb[:, :teacher_dim].to(torch.float64))
        h2 = model(s2_emb[:, :teacher_dim].to(torch.float64))
        sim = substrate.geodesic_similarity(h1, h2).cpu().numpy()

    rho, _ = spearmanr(sim, ds['scores'])
    del teacher; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(rho)


# ═══════════════════════════════════════════════════════════════════════════
# FULL EVALUATION (all datasets)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_all_datasets(
    model: nn.Module,
    substrate: nn.Module,
    teacher_name: str,
    eval_data: Dict[str, Dict],
    device: str = 'cuda',
) -> Dict[str, Dict]:
    """
    Evaluate CGT model on all datasets.

    Args:
        model: CGT student model
        substrate: LorentzSubstrate
        teacher_name: HuggingFace model name
        eval_data: Dict of {dataset_name: {'s1':[], 's2':[], 'scores':[]}}
        device: torch device

    Returns:
        Dict of {dataset_name: {'teacher_rho', 'student_rho', 'retention'}}
    """
    assert HAS_ST, "sentence-transformers required"

    teacher = SentenceTransformer(teacher_name)
    teacher_dim = teacher.get_sentence_embedding_dimension()
    model.eval()

    results = {}
    for ds_name, ds in eval_data.items():
        if not isinstance(ds, dict) or 's1' not in ds:
            continue
        with torch.no_grad():
            s1_e = teacher.encode(ds['s1'], batch_size=512,
                                  convert_to_tensor=True, device=device)
            s2_e = teacher.encode(ds['s2'], batch_size=512,
                                  convert_to_tensor=True, device=device)

            # Teacher performance (cosine)
            t_cos = F.cosine_similarity(s1_e, s2_e).cpu().numpy()
            t_rho, _ = spearmanr(t_cos, ds['scores'])

            # Student performance (geodesic)
            h1 = model(s1_e[:, :teacher_dim].to(torch.float64))
            h2 = model(s2_e[:, :teacher_dim].to(torch.float64))
            sim = substrate.geodesic_similarity(h1, h2).cpu().numpy()
            s_rho, _ = spearmanr(sim, ds['scores'])

        retention = (s_rho / t_rho * 100) if t_rho > 0 else 0
        results[ds_name] = {
            'teacher_rho': round(float(t_rho), 4),
            'student_rho': round(float(s_rho), 4),
            'retention': round(float(retention), 1),
        }

    del teacher; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# FALSIFICATION PROTOCOL (F1, F2, F3)
# ═══════════════════════════════════════════════════════════════════════════

def run_falsification(
    model: nn.Module,
    substrate: nn.Module,
    teacher_name: str,
    eval_data: Dict,
    device: str = 'cuda',
    dataset_key: str = 'STSBenchmark',
    f2_threshold: float = 0.8,
    f3_threshold: float = 0.5,
    f3_k: int = 10,
    n_samples: int = 500,
) -> Dict[str, Any]:
    """
    Run F1/F2/F3 geometric falsification protocol.

    F1: Manifold integrity — |⟨h,h⟩_L + 1/K| < 1e-5
    F2: Distance preservation — Spearman(geodesic, cosine) > f2_threshold
    F3: Neighbourhood consistency — k-NN overlap > f3_threshold

    Returns:
        Dict with F1_violation, F1_pass, F2_rho, F2_pass,
        F3_overlap, F3_pass, pass_count
    """
    assert HAS_ST, "sentence-transformers required"

    teacher = SentenceTransformer(teacher_name)
    teacher_dim = teacher.get_sentence_embedding_dimension()
    ds = eval_data[dataset_key]

    model.eval()
    with torch.no_grad():
        s_emb = teacher.encode(ds['s1'][:n_samples], batch_size=512,
                               convert_to_tensor=True, device=device)
        t_emb = s_emb[:, :teacher_dim].to(torch.float64)
        h_emb = model(t_emb)

        # F1: Manifold violation
        f1_viol = substrate.manifold_violation(h_emb).item()
        f1_pass = f1_viol < 1e-5

        # F2: Distance preservation (sample pairs)
        n = min(n_samples, h_emb.shape[0])
        D_h = substrate.distance_matrix_points(h_emb[:n], h_emb[:n], pairwise=True)
        D_t = torch.cdist(t_emb[:n].float(), t_emb[:n].float())

        # Upper triangle
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        h_dists = D_h[mask].cpu().numpy()
        t_dists = D_t[mask].cpu().numpy()
        f2_rho, _ = spearmanr(h_dists, t_dists)
        f2_pass = f2_rho > f2_threshold

        # F3: k-NN overlap
        k = min(f3_k, n - 1)
        knn_t = t_dists_matrix_knn(D_t[:n, :n], k)
        knn_s = t_dists_matrix_knn(D_h[:n, :n], k)
        overlaps = []
        for i in range(n):
            overlap = len(set(knn_t[i]) & set(knn_s[i])) / k
            overlaps.append(overlap)
        f3_overlap = np.mean(overlaps)
        f3_pass = f3_overlap > f3_threshold

    pass_count = sum([f1_pass, f2_pass, f3_pass])

    del teacher; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'F1_violation': f1_viol, 'F1_pass': f1_pass,
        'F2_rho': round(float(f2_rho), 4), 'F2_pass': f2_pass,
        'F3_overlap': round(float(f3_overlap), 4), 'F3_pass': f3_pass,
        'pass_count': pass_count,
    }


def t_dists_matrix_knn(D: torch.Tensor, k: int) -> List[List[int]]:
    """Extract k nearest neighbours from distance matrix."""
    _, indices = D.topk(k + 1, largest=False, dim=-1)
    # Remove self (index 0 is self with dist=0)
    result = []
    for i in range(D.shape[0]):
        neighbours = [idx.item() for idx in indices[i] if idx.item() != i][:k]
        result.append(neighbours)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# EUCLIDEAN BASELINE MODEL
# ═══════════════════════════════════════════════════════════════════════════

class EuclideanStudent(nn.Module):
    """
    Euclidean MLP projector — same architecture as CGTStudent but
    WITHOUT exponential map. Outputs stay in R^d, similarity via cosine.

    Use as ablation baseline to isolate the contribution of
    hyperbolic geometry vs. the MLP projector alone.
    """

    def __init__(self, teacher_dim: int, student_dim: int, hidden_dim: int = 256):
        super().__init__()
        l1 = nn.Linear(teacher_dim, hidden_dim)
        l2 = nn.Linear(hidden_dim, hidden_dim)
        l3 = nn.Linear(hidden_dim, student_dim)
        nn.init.orthogonal_(l1.weight, gain=1.0)
        nn.init.orthogonal_(l2.weight, gain=1.0)
        nn.init.normal_(l3.weight, std=1e-4)
        if l3.bias is not None:
            nn.init.zeros_(l3.bias)
        self.projector = nn.Sequential(
            l1, nn.LayerNorm(hidden_dim), nn.GELU(),
            l2, nn.LayerNorm(hidden_dim), nn.GELU(),
            l3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


# ═══════════════════════════════════════════════════════════════════════════
# BASELINE EVALUATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _eval_projected_baseline(
    teacher_name: str,
    eval_data: Dict,
    transform_fn,
    device: str = 'cuda',
) -> Dict[str, Dict]:
    """
    Generic baseline evaluation: encode with teacher, project, eval cosine.

    Args:
        teacher_name: HuggingFace teacher model name
        eval_data: {dataset_name: {'s1','s2','scores'}}
        transform_fn: callable(embeddings_np) → projected_np
        device: torch device

    Returns:
        {dataset_name: {'teacher_rho','student_rho','retention'}}
    """
    assert HAS_ST, "sentence-transformers required"
    teacher = SentenceTransformer(teacher_name)

    results = {}
    for ds_name, ds in eval_data.items():
        if not isinstance(ds, dict) or 's1' not in ds:
            continue
        s1_emb = teacher.encode(ds['s1'], batch_size=512, convert_to_tensor=False)
        s2_emb = teacher.encode(ds['s2'], batch_size=512, convert_to_tensor=False)

        # Teacher
        t_cos = F.cosine_similarity(
            torch.tensor(s1_emb), torch.tensor(s2_emb)).numpy()
        t_rho, _ = spearmanr(t_cos, ds['scores'])

        # Projected
        s1_p = transform_fn(s1_emb)
        s2_p = transform_fn(s2_emb)
        s_cos = F.cosine_similarity(
            torch.tensor(s1_p, dtype=torch.float32),
            torch.tensor(s2_p, dtype=torch.float32)).numpy()
        s_rho, _ = spearmanr(s_cos, ds['scores'])

        retention = (s_rho / t_rho * 100) if t_rho > 0 else 0
        results[ds_name] = {
            'teacher_rho': round(float(t_rho), 4),
            'student_rho': round(float(s_rho), 4),
            'retention': round(float(retention), 1),
        }

    del teacher; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def eval_pca_baseline(
    teacher_name: str,
    eval_data: Dict,
    train_embs: np.ndarray,
    target_dim: int = 32,
    device: str = 'cuda',
) -> Tuple[Dict, float]:
    """PCA baseline. Returns (results_dict, explained_variance_ratio)."""
    assert HAS_SKLEARN, "scikit-learn required"
    pca = PCA(n_components=target_dim)
    pca.fit(train_embs[:, :train_embs.shape[1]])
    explained = float(pca.explained_variance_ratio_.sum())
    results = _eval_projected_baseline(
        teacher_name, eval_data, pca.transform, device)
    return results, explained


def eval_random_baseline(
    teacher_name: str,
    eval_data: Dict,
    target_dim: int = 32,
    seed: int = 42,
    device: str = 'cuda',
) -> Dict:
    """Gaussian random projection baseline."""
    assert HAS_SKLEARN, "scikit-learn required"
    assert HAS_ST, "sentence-transformers required"

    teacher = SentenceTransformer(teacher_name)
    t_dim = teacher.get_sentence_embedding_dimension()
    del teacher; gc.collect()

    rp = GaussianRandomProjection(n_components=target_dim, random_state=seed)
    rp.fit(np.random.randn(100, t_dim))
    return _eval_projected_baseline(
        teacher_name, eval_data, rp.transform, device)


def eval_mrl_baseline(
    teacher_name: str,
    eval_data: Dict,
    target_dim: int = 32,
    device: str = 'cuda',
) -> Optional[Dict]:
    """Matryoshka (MRL) truncation baseline."""
    def truncate(emb):
        return emb[:, :target_dim]
    return _eval_projected_baseline(
        teacher_name, eval_data, truncate, device)


def train_euclidean_mlp(
    train_embs: torch.Tensor,
    teacher_name: str,
    eval_data: Dict,
    teacher_dim: int = 384,
    student_dim: int = 32,
    hidden_dim: int = 256,
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 256,
    device: str = 'cuda',
    seed: int = 42,
    patience: int = 25,
) -> Tuple[Dict, float]:
    """
    Train Euclidean MLP baseline with cosine InfoNCE.
    Same architecture as CGTStudent but without exp_map.

    Returns:
        (results_dict, best_stsb_rho)
    """
    set_seed(seed)
    model = EuclideanStudent(teacher_dim, student_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    embs = train_embs[:, :teacher_dim].to(device, dtype=torch.float32)
    best_rho = 0.0
    stall = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(embs))
        for i in range(0, len(embs), batch_size):
            batch = embs[idx[i:i + batch_size]]
            if len(batch) < 8:
                continue
            proj = model(batch)
            proj_norm = F.normalize(proj, dim=-1)
            sim = proj_norm @ proj_norm.T / 0.07
            sim.fill_diagonal_(-50)
            labels = torch.arange(len(batch), device=device)
            loss = F.cross_entropy(sim, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Quick val
        rho_val = _eval_euclidean_quick(model, teacher_name, eval_data,
                                         teacher_dim, device)
        if rho_val > best_rho:
            best_rho = rho_val
            stall = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
        if stall >= patience:
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Full eval
    results = _eval_euclidean_full(model, teacher_name, eval_data,
                                    teacher_dim, device)
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results, best_rho


def _eval_euclidean_quick(
    model: nn.Module,
    teacher_name: str,
    eval_data: Dict,
    teacher_dim: int,
    device: str,
) -> float:
    """Quick STSb validation for Euclidean model."""
    assert HAS_ST, "sentence-transformers required"
    teacher = SentenceTransformer(teacher_name)
    ds = eval_data['STSBenchmark']
    model.eval()
    with torch.no_grad():
        s1_e = teacher.encode(ds['s1'], batch_size=512,
                              convert_to_tensor=True, device=device)
        s2_e = teacher.encode(ds['s2'], batch_size=512,
                              convert_to_tensor=True, device=device)
        s1_p = model(s1_e[:, :teacher_dim].float())
        s2_p = model(s2_e[:, :teacher_dim].float())
        cos = F.cosine_similarity(s1_p, s2_p).cpu().numpy()
    rho, _ = spearmanr(cos, ds['scores'])
    del teacher; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(rho)


def _eval_euclidean_full(
    model: nn.Module,
    teacher_name: str,
    eval_data: Dict,
    teacher_dim: int,
    device: str,
) -> Dict[str, Dict]:
    """Full eval for Euclidean baseline."""
    assert HAS_ST, "sentence-transformers required"
    teacher = SentenceTransformer(teacher_name)
    model.eval()
    results = {}
    for ds_name, ds in eval_data.items():
        if not isinstance(ds, dict) or 's1' not in ds:
            continue
        with torch.no_grad():
            s1_e = teacher.encode(ds['s1'], batch_size=512,
                                  convert_to_tensor=True, device=device)
            s2_e = teacher.encode(ds['s2'], batch_size=512,
                                  convert_to_tensor=True, device=device)
            t_cos = F.cosine_similarity(s1_e, s2_e).cpu().numpy()
            t_rho, _ = spearmanr(t_cos, ds['scores'])
            s1_p = model(s1_e[:, :teacher_dim].float())
            s2_p = model(s2_e[:, :teacher_dim].float())
            s_cos = F.cosine_similarity(s1_p, s2_p).cpu().numpy()
            s_rho, _ = spearmanr(s_cos, ds['scores'])
        retention = (s_rho / t_rho * 100) if t_rho > 0 else 0
        results[ds_name] = {
            'teacher_rho': round(float(t_rho), 4),
            'student_rho': round(float(s_rho), 4),
            'retention': round(float(retention), 1),
        }
    del teacher; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# LATENCY BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_latency(
    model_fn,
    input_tensor: torch.Tensor,
    n_warmup: int = 10,
    n_runs: int = 100,
    label: str = '',
) -> float:
    """
    Benchmark forward pass latency.

    Args:
        model_fn: Callable taking input_tensor
        input_tensor: Input for the model
        n_warmup: Warmup iterations
        n_runs: Timed iterations
        label: Label for printing

    Returns:
        Mean latency in milliseconds
    """
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model_fn(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model_fn(input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    if label:
        print(f"  {label:25s}: {mean_ms:.2f} ± {std_ms:.2f} ms  "
              f"(shape: {tuple(out.shape)})")
    return mean_ms
