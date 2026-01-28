# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Tree Embedding Protocols - DUAL REGIME VERSION
===============================================

REGIME A (exact):  N ≤ 10,000 - Full D_tree, exact stress
REGIME B (stochastic): N > 10,000 - No D_tree, pair sampling

These regimes NEVER mix in the same execution flow.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Literal
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# OBRIGATÓRIO: Disable TorchDynamo
import torch._dynamo
torch._dynamo.disable()

from scipy.stats import linregress
from scipy.spatial.distance import pdist


# ═══════════════════════════════════════════════════════════════════════════════
#                    CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

EXACT_THRESHOLD = 10_000
K_MAX = 2_000_000
K_MULTIPLIER = 100


# ═══════════════════════════════════════════════════════════════════════════════
#                    DEVICE + SEED + MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
#                    TREE STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_tree_structure(b: int, d: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """Generate tree structure (parent, depth arrays). NO distance matrix."""
    if b == 1:
        n_nodes = d + 1
    else:
        n_nodes = (b ** (d + 1) - 1) // (b - 1)
    
    parent = np.full(n_nodes, -1, dtype=np.int32)
    depth = np.zeros(n_nodes, dtype=np.int32)
    
    for i in range(1, n_nodes):
        parent[i] = (i - 1) // b
        depth[i] = depth[parent[i]] + 1
    
    return n_nodes, parent, depth


def tree_distance(i: int, j: int, parent: np.ndarray, depth: np.ndarray) -> int:
    """Tree distance via LCA. O(depth) time, O(1) memory."""
    di, dj = int(depth[i]), int(depth[j])
    ii, jj = i, j
    
    while di > dj:
        ii = parent[ii]
        di -= 1
    while dj > di:
        jj = parent[jj]
        dj -= 1
    
    steps = 0
    while ii != jj:
        ii = parent[ii]
        jj = parent[jj]
        steps += 1
    
    return (int(depth[i]) - di) + (int(depth[j]) - dj) + 2 * steps


def generate_distance_matrix(b: int, d: int) -> Tuple[int, np.ndarray]:
    """Generate full D_tree. ONLY for REGIME A (exact)."""
    n_nodes, parent, depth = generate_tree_structure(b, d)
    
    D = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist = tree_distance(i, j, parent, depth)
            D[i, j] = dist
            D[j, i] = dist
    
    return n_nodes, D


# ═══════════════════════════════════════════════════════════════════════════════
#                    LORENTZ GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

def safe_acosh(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    delta = x - 1.0
    mask = delta < eps
    val_std = torch.acosh(torch.clamp(x, min=1.0 + eps))
    val_taylor = torch.sqrt(2.0 * torch.clamp(delta, min=0.0) + 1e-15)
    return torch.where(mask, val_taylor, val_std)


class LorentzManifold(nn.Module):
    def __init__(self, dim: int, initial_K: float = 1.0):
        super().__init__()
        self.dim = dim
        self.eps = 1e-6
        self._log_K = nn.Parameter(torch.tensor(np.log(initial_K), dtype=torch.float64))
    
    @property
    def K(self) -> torch.Tensor:
        return torch.exp(self._log_K).clamp(min=0.1, max=10.0)
    
    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        K = self.K
        v_norm_sq = (v ** 2).sum(dim=-1, keepdim=True) + self.eps
        v_norm = torch.sqrt(v_norm_sq)
        scale = (v_norm * torch.sqrt(K)).clamp(max=15.0)
        x0 = torch.cosh(scale) / torch.sqrt(K)
        xi = torch.sinh(scale) * v / (v_norm + self.eps)
        return torch.cat([x0, xi], dim=-1)
    
    def dist_pairs(self, x: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        K = self.K
        xi, xj = x[i], x[j]
        mink = -xi[:, 0] * xj[:, 0] + (xi[:, 1:] * xj[:, 1:]).sum(dim=-1)
        arg = (-K * mink).clamp(min=1.0, max=1e5)
        d = safe_acosh(arg) / (torch.sqrt(K) + 1e-9)
        return torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    
    def distance_matrix(self, x: torch.Tensor) -> torch.Tensor:
        K = self.K
        x0 = x[:, :1]
        xi = x[:, 1:]
        time_inner = torch.mm(x0, x0.t())
        space_inner = torch.mm(xi, xi.t())
        mink_inner = -time_inner + space_inner
        arg = (-K * mink_inner).clamp(min=1.0, max=1e5)
        d = safe_acosh(arg) / (torch.sqrt(K) + 1e-9)
        return torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#                    STOCHASTIC STRESS (REGIME B ONLY)
# ═══════════════════════════════════════════════════════════════════════════════

def stochastic_stress_euclidean(
    X: torch.Tensor,
    parent: np.ndarray,
    depth: np.ndarray,
    K: int,
    device: torch.device
) -> torch.Tensor:
    """Stochastic stress for Euclidean embedding. REGIME B only."""
    N = X.shape[0]
    i = torch.randint(0, N, (K,), device=device)
    j = torch.randint(0, N, (K,), device=device)
    mask = i != j
    i, j = i[mask], j[mask]
    
    dij_emb = torch.norm(X[i] - X[j], dim=1)
    
    dij_tree = torch.tensor(
        [tree_distance(ii, jj, parent, depth) for ii, jj in zip(i.tolist(), j.tolist())],
        device=device,
        dtype=X.dtype
    )
    
    return ((dij_emb - dij_tree) ** 2).mean()


def stochastic_stress_hyperbolic(
    X: torch.Tensor,
    manifold: LorentzManifold,
    parent: np.ndarray,
    depth: np.ndarray,
    K: int,
    device: torch.device
) -> torch.Tensor:
    """Stochastic stress for hyperbolic embedding. REGIME B only."""
    N = X.shape[0]
    i = torch.randint(0, N, (K,), device=device)
    j = torch.randint(0, N, (K,), device=device)
    mask = i != j
    i, j = i[mask], j[mask]
    
    dij_emb = manifold.dist_pairs(X, i, j)
    
    dij_tree = torch.tensor(
        [tree_distance(ii, jj, parent, depth) for ii, jj in zip(i.tolist(), j.tolist())],
        device=device,
        dtype=X.dtype
    )
    
    return ((dij_emb - dij_tree) ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#                    REGIME A: EXACT EMBEDDINGS (N ≤ 10k)
# ═══════════════════════════════════════════════════════════════════════════════

def embed_euclidean_exact(
    D_tree: np.ndarray,
    dim: int,
    n_epochs: int,
    lr: float = 0.01,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """REGIME A: Exact Euclidean embedding."""
    device = get_device()
    set_seed(seed)
    
    N = D_tree.shape[0]
    D_target = torch.tensor(D_tree, dtype=torch.float64, device=device)
    
    X = nn.Parameter(torch.randn(N, dim, dtype=torch.float64, device=device))
    with torch.no_grad():
        X.mul_(0.1)
    
    optimizer = optim.Adam([X], lr=lr)
    mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
    
    for _ in range(n_epochs):
        optimizer.zero_grad()
        D_emb = torch.cdist(X, X, p=2)
        loss = ((D_emb[mask] - D_target[mask]) ** 2).mean()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        D_final = torch.cdist(X, X, p=2)
    
    X_np = X.detach().cpu().numpy()
    D_np = D_final.detach().cpu().numpy()
    
    del X, D_emb, D_target, D_final
    clear_gpu()
    
    return X_np, D_np


def embed_hyperbolic_exact(
    D_tree: np.ndarray,
    dim: int,
    n_epochs: int,
    lr: float = 0.001,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """REGIME A: Exact hyperbolic embedding."""
    device = get_device()
    set_seed(seed)
    
    N = D_tree.shape[0]
    D_target = torch.tensor(D_tree, dtype=torch.float64, device=device)
    
    manifold = LorentzManifold(dim).to(device).double()
    
    V = nn.Parameter(torch.randn(N, dim, dtype=torch.float64, device=device))
    with torch.no_grad():
        V.mul_(0.1)
    
    params = [V] + list(manifold.parameters())
    optimizer = optim.Adam(params, lr=lr)
    mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
    
    for _ in range(n_epochs):
        optimizer.zero_grad()
        X = manifold.exp_map_zero(V)
        D_emb = manifold.distance_matrix(X)
        loss = ((D_emb[mask] - D_target[mask]) ** 2).mean()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        X_final = manifold.exp_map_zero(V)
        D_final = manifold.distance_matrix(X_final)
        K_final = manifold.K.item()
    
    X_np = X_final.detach().cpu().numpy()
    D_np = D_final.detach().cpu().numpy()
    
    del V, X_final, D_final, D_target, manifold
    clear_gpu()
    
    return X_np, D_np, K_final


# ═══════════════════════════════════════════════════════════════════════════════
#                    REGIME B: STOCHASTIC EMBEDDINGS (N > 10k)
# ═══════════════════════════════════════════════════════════════════════════════

def embed_euclidean_stochastic(
    N: int,
    parent: np.ndarray,
    depth: np.ndarray,
    dim: int,
    n_epochs: int,
    lr: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """REGIME B: Stochastic Euclidean embedding. NO D_tree."""
    device = get_device()
    set_seed(seed)
    
    K = min(K_MULTIPLIER * N, K_MAX)
    
    X = nn.Parameter(torch.randn(N, dim, dtype=torch.float64, device=device))
    with torch.no_grad():
        X.mul_(0.1)
    
    optimizer = optim.Adam([X], lr=lr)
    
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = stochastic_stress_euclidean(X, parent, depth, K, device)
        loss.backward()
        optimizer.step()
    
    X_np = X.detach().cpu().numpy()
    
    del X
    clear_gpu()
    
    return X_np


def embed_hyperbolic_stochastic(
    N: int,
    parent: np.ndarray,
    depth: np.ndarray,
    dim: int,
    n_epochs: int,
    lr: float = 0.001,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """REGIME B: Stochastic hyperbolic embedding. NO D_tree."""
    device = get_device()
    set_seed(seed)
    
    K = min(K_MULTIPLIER * N, K_MAX)
    
    manifold = LorentzManifold(dim).to(device).double()
    
    V = nn.Parameter(torch.randn(N, dim, dtype=torch.float64, device=device))
    with torch.no_grad():
        V.mul_(0.1)
    
    params = [V] + list(manifold.parameters())
    optimizer = optim.Adam(params, lr=lr)
    
    for _ in range(n_epochs):
        optimizer.zero_grad()
        X = manifold.exp_map_zero(V)
        loss = stochastic_stress_hyperbolic(X, manifold, parent, depth, K, device)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        X_final = manifold.exp_map_zero(V)
        K_final = manifold.K.item()
    
    X_np = X_final.detach().cpu().numpy()
    
    del V, X_final, manifold
    clear_gpu()
    
    return X_np, K_final


# ═══════════════════════════════════════════════════════════════════════════════
#                    DISTORTION COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_distortion_exact(D_original: np.ndarray, D_embedded: np.ndarray) -> float:
    """REGIME A: Exact distortion."""
    N = D_original.shape[0]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_orig = np.maximum(D_original[mask], 1e-10)
    d_emb = D_embedded[mask]
    return float(np.mean(np.abs(d_emb / d_orig - 1.0)))


def compute_distortion_sampled(
    X: np.ndarray,
    parent: np.ndarray,
    depth: np.ndarray,
    geometry: str = "euclidean",
    K_curv: float = 1.0,
    n_samples: int = 100_000,
) -> float:
    """REGIME B: Sampled distortion."""
    N = X.shape[0]
    
    i = np.random.randint(0, N, n_samples)
    j = np.random.randint(0, N, n_samples)
    mask = i != j
    i, j = i[mask], j[mask]
    
    d_tree = np.array([tree_distance(ii, jj, parent, depth) for ii, jj in zip(i, j)])
    d_tree = np.maximum(d_tree, 1e-10)
    
    if geometry == "euclidean":
        d_emb = np.linalg.norm(X[i] - X[j], axis=1)
    else:
        xi, xj = X[i], X[j]
        mink = -xi[:, 0] * xj[:, 0] + (xi[:, 1:] * xj[:, 1:]).sum(axis=1)
        arg = np.clip(-K_curv * mink, 1.0, 1e5)
        d_emb = np.arccosh(arg) / np.sqrt(K_curv)
    
    return float(np.mean(np.abs(d_emb / d_tree - 1.0)))


# ═══════════════════════════════════════════════════════════════════════════════
#                    PROTOCOL E1
# ═══════════════════════════════════════════════════════════════════════════════

def run_protocol_e1(
    b: int = 3,
    d: int = 15,
    dimensions: List[int] = None,
    n_trials: int = 10,
    n_epochs: int = 1000,
    seed: int = 42,
    mode: Literal["exact", "stochastic"] = "stochastic",
    verbose: bool = True,
) -> Dict:
    """
    Protocol E1: Distortion vs Dimension.
    
    Args:
        mode: "exact" (N ≤ 10k only) or "stochastic" (any N)
    """
    assert mode in ["exact", "stochastic"], "mode must be 'exact' or 'stochastic'"
    
    if dimensions is None:
        dimensions = [4, 8, 16, 32, 64, 128, 256]
    
    device = get_device()
    n_nodes, parent, depth = generate_tree_structure(b, d)
    
    # REGIME GUARD
    if mode == "exact" and n_nodes > EXACT_THRESHOLD:
        raise ValueError(f"Exact mode not allowed for N={n_nodes:,} > {EXACT_THRESHOLD:,}")
    
    if verbose:
        print(f"Protocol E1: Distortion vs Dimension")
        print(f"  Tree: T_{{{b},{d}}} with {n_nodes:,} nodes")
        print(f"  Mode: {mode.upper()}")
        print(f"  Dimensions: {dimensions}")
        print(f"  Trials: {n_trials}, Epochs: {n_epochs}")
        print(f"  Device: {device}")
    
    # REGIME A: generate D_tree
    D_tree = None
    if mode == "exact":
        _, D_tree = generate_distance_matrix(b, d)
    
    results = {
        "b": b, "d": d, "n_nodes": n_nodes, "mode": mode,
        "dimensions": dimensions,
        "euclidean": {"mean": [], "std": []},
        "hyperbolic": {"mean": [], "std": []},
    }
    
    for dim in dimensions:
        if verbose:
            print(f"\n  Dimension {dim}:")
        
        euc_dist, hyp_dist = [], []
        
        for trial in range(n_trials):
            trial_seed = seed + trial
            
            if mode == "exact":
                # REGIME A
                X_euc, D_euc = embed_euclidean_exact(D_tree, dim, n_epochs, seed=trial_seed)
                euc_dist.append(compute_distortion_exact(D_tree, D_euc))
                del X_euc, D_euc
                
                X_hyp, D_hyp, K = embed_hyperbolic_exact(D_tree, dim, n_epochs, seed=trial_seed)
                hyp_dist.append(compute_distortion_exact(D_tree, D_hyp))
                del X_hyp, D_hyp
            else:
                # REGIME B
                X_euc = embed_euclidean_stochastic(n_nodes, parent, depth, dim, n_epochs, seed=trial_seed)
                euc_dist.append(compute_distortion_sampled(X_euc, parent, depth, geometry="euclidean"))
                del X_euc
                
                X_hyp, K = embed_hyperbolic_stochastic(n_nodes, parent, depth, dim, n_epochs, seed=trial_seed)
                hyp_dist.append(compute_distortion_sampled(X_hyp, parent, depth, geometry="hyperbolic", K_curv=K))
                del X_hyp
            
            clear_gpu()
        
        results["euclidean"]["mean"].append(float(np.mean(euc_dist)))
        results["euclidean"]["std"].append(float(np.std(euc_dist)))
        results["hyperbolic"]["mean"].append(float(np.mean(hyp_dist)))
        results["hyperbolic"]["std"].append(float(np.std(hyp_dist)))
        
        if verbose:
            print(f"    Euclidean:  {np.mean(euc_dist):.4f} ± {np.std(euc_dist):.4f}")
            print(f"    Hyperbolic: {np.mean(hyp_dist):.4f} ± {np.std(hyp_dist):.4f}")
    
    # LaTeX
    threshold = 0.1
    dim_euc, dim_hyp = None, None
    for i, dim in enumerate(dimensions):
        if dim_euc is None and results["euclidean"]["mean"][i] < threshold:
            dim_euc = dim
        if dim_hyp is None and results["hyperbolic"]["mean"][i] < threshold:
            dim_hyp = dim
    
    results["latex"] = {
        "RESULT_E1_DIM_EUCLIDEAN": str(dim_euc) if dim_euc else f">{dimensions[-1]}",
        "RESULT_E1_DIM_HYPERBOLIC": str(dim_hyp) if dim_hyp else f">{dimensions[-1]}",
        "RESULT_E1_RATIO": f"{dim_euc/dim_hyp:.1f}" if (dim_euc and dim_hyp) else "N/A",
        "RESULT_E1_STATUS": "CONFIRMED" if (dim_euc and dim_hyp and dim_euc > 1.5 * dim_hyp) else "NOT CONFIRMED",
    }
    
    if D_tree is not None:
        del D_tree
    clear_gpu()
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    PROTOCOL E2
# ═══════════════════════════════════════════════════════════════════════════════

def run_protocol_e2(
    b: int = 3,
    dim: int = 64,
    depths: List[int] = None,
    n_trials: int = 10,
    n_epochs: int = 1000,
    seed: int = 42,
    mode: Literal["exact", "stochastic"] = "stochastic",
    verbose: bool = True,
) -> Dict:
    """Protocol E2: Depth Scaling."""
    assert mode in ["exact", "stochastic"]
    
    if depths is None:
        depths = [5, 10, 15, 20, 25]
    
    device = get_device()
    
    if verbose:
        print(f"Protocol E2: Depth Scaling")
        print(f"  Mode: {mode.upper()}")
        print(f"  Fixed dimension: {dim}")
        print(f"  Depths: {depths}")
        print(f"  Device: {device}")
    
    results = {
        "b": b, "dim": dim, "depths": depths, "mode": mode,
        "euclidean": {"distortions": [], "std": []},
        "hyperbolic": {"distortions": [], "std": []},
    }
    
    for depth_val in depths:
        n_nodes, parent, depth_arr = generate_tree_structure(b, depth_val)
        
        # REGIME GUARD
        if mode == "exact" and n_nodes > EXACT_THRESHOLD:
            raise ValueError(f"Exact mode not allowed for d={depth_val}, N={n_nodes:,}")
        
        if verbose:
            print(f"\n  Depth {depth_val} ({n_nodes:,} nodes):")
        
        D_tree = None
        if mode == "exact":
            _, D_tree = generate_distance_matrix(b, depth_val)
        
        euc_dist, hyp_dist = [], []
        
        for trial in range(n_trials):
            trial_seed = seed + trial
            
            if mode == "exact":
                X_euc, D_euc = embed_euclidean_exact(D_tree, dim, n_epochs, seed=trial_seed)
                euc_dist.append(compute_distortion_exact(D_tree, D_euc))
                del X_euc, D_euc
                
                X_hyp, D_hyp, K = embed_hyperbolic_exact(D_tree, dim, n_epochs, seed=trial_seed)
                hyp_dist.append(compute_distortion_exact(D_tree, D_hyp))
                del X_hyp, D_hyp
            else:
                X_euc = embed_euclidean_stochastic(n_nodes, parent, depth_arr, dim, n_epochs, seed=trial_seed)
                euc_dist.append(compute_distortion_sampled(X_euc, parent, depth_arr, geometry="euclidean"))
                del X_euc
                
                X_hyp, K = embed_hyperbolic_stochastic(n_nodes, parent, depth_arr, dim, n_epochs, seed=trial_seed)
                hyp_dist.append(compute_distortion_sampled(X_hyp, parent, depth_arr, geometry="hyperbolic", K_curv=K))
                del X_hyp
            
            clear_gpu()
        
        results["euclidean"]["distortions"].append(float(np.mean(euc_dist)))
        results["euclidean"]["std"].append(float(np.std(euc_dist)))
        results["hyperbolic"]["distortions"].append(float(np.mean(hyp_dist)))
        results["hyperbolic"]["std"].append(float(np.std(hyp_dist)))
        
        if verbose:
            print(f"    Euclidean:  {np.mean(euc_dist):.4f} ± {np.std(euc_dist):.4f}")
            print(f"    Hyperbolic: {np.mean(hyp_dist):.4f} ± {np.std(hyp_dist):.4f}")
        
        if D_tree is not None:
            del D_tree
        clear_gpu()
    
    # Power-law fit
    log_d = np.log(np.array(depths))
    log_euc = np.log(np.array(results["euclidean"]["distortions"]) + 1e-10)
    log_hyp = np.log(np.array(results["hyperbolic"]["distortions"]) + 1e-10)
    
    slope_e, _, _, _, se_e = linregress(log_d, log_euc)
    slope_h, _, _, _, se_h = linregress(log_d, log_hyp)
    
    results["fit"] = {
        "alpha_euclidean": float(slope_e),
        "alpha_hyperbolic": float(slope_h),
    }
    
    results["latex"] = {
        "RESULT_E2_ALPHA_EUCLIDEAN": f"{slope_e:.3f}",
        "RESULT_E2_ALPHA_HYPERBOLIC": f"{slope_h:.3f}",
        "RESULT_E2_STATUS": "CONFIRMED" if (slope_e > 0.1 and slope_h < 0.1) else "NOT CONFIRMED",
    }
    
    if verbose:
        print(f"\n  Power-law fit:")
        print(f"    α_E = {slope_e:.3f} ± {se_e:.3f}")
        print(f"    α_H = {slope_h:.3f} ± {se_h:.3f}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    PROTOCOL E3
# ═══════════════════════════════════════════════════════════════════════════════

def run_protocol_e3(
    b: int = 3,
    depths: List[int] = None,
    dim: int = 64,
    n_trials: int = 5,
    n_epochs: int = 1000,
    persistence_threshold: float = 0.1,
    seed: int = 42,
    mode: Literal["exact", "stochastic"] = "stochastic",
    verbose: bool = True,
) -> Dict:
    """Protocol E3: Topological Integrity."""
    try:
        from ripser import ripser
    except ImportError:
        return {"error": "ripser not installed", "latex": {}}
    
    assert mode in ["exact", "stochastic"]
    
    if depths is None:
        depths = [5, 10, 15, 20]
    
    device = get_device()
    
    if verbose:
        print(f"Protocol E3: Topological Integrity")
        print(f"  Mode: {mode.upper()}")
        print(f"  Depths: {depths}, Dimension: {dim}")
        print(f"  Device: {device}")
    
    results = {
        "b": b, "dim": dim, "depths": depths, "mode": mode,
        "euclidean_betti1": {},
        "hyperbolic_betti1": {},
    }
    
    for depth_val in depths:
        n_nodes, parent, depth_arr = generate_tree_structure(b, depth_val)
        
        if mode == "exact" and n_nodes > EXACT_THRESHOLD:
            raise ValueError(f"Exact mode not allowed for d={depth_val}")
        
        if verbose:
            print(f"\n  Depth {depth_val} ({n_nodes:,} nodes):")
        
        D_tree = None
        if mode == "exact":
            _, D_tree = generate_distance_matrix(b, depth_val)
        
        euc_b1, hyp_b1 = [], []
        
        for trial in range(n_trials):
            trial_seed = seed + trial
            
            if mode == "exact":
                X_euc, _ = embed_euclidean_exact(D_tree, dim, n_epochs, seed=trial_seed)
            else:
                X_euc = embed_euclidean_stochastic(n_nodes, parent, depth_arr, dim, n_epochs, seed=trial_seed)
            
            diameter = np.max(pdist(X_euc))
            dgm = ripser(X_euc, maxdim=1)['dgms'][1]
            b1 = int(np.sum((dgm[:, 1] - dgm[:, 0]) > persistence_threshold * diameter)) if len(dgm) > 0 else 0
            euc_b1.append(b1)
            del X_euc
            
            if mode == "exact":
                X_hyp, _, K = embed_hyperbolic_exact(D_tree, dim, n_epochs, seed=trial_seed)
            else:
                X_hyp, K = embed_hyperbolic_stochastic(n_nodes, parent, depth_arr, dim, n_epochs, seed=trial_seed)
            
            x0 = X_hyp[:, 0:1]
            xi = X_hyp[:, 1:]
            X_poincare = xi / (x0 + 1.0 + 1e-8)
            
            diameter = np.max(pdist(X_poincare))
            dgm = ripser(X_poincare, maxdim=1)['dgms'][1]
            b1 = int(np.sum((dgm[:, 1] - dgm[:, 0]) > persistence_threshold * diameter)) if len(dgm) > 0 else 0
            hyp_b1.append(b1)
            del X_hyp, X_poincare
            
            clear_gpu()
        
        results["euclidean_betti1"][depth_val] = float(np.mean(euc_b1))
        results["hyperbolic_betti1"][depth_val] = float(np.mean(hyp_b1))
        
        if verbose:
            print(f"    Euclidean  β₁: {np.mean(euc_b1):.1f}")
            print(f"    Hyperbolic β₁: {np.mean(hyp_b1):.1f}")
        
        if D_tree is not None:
            del D_tree
        clear_gpu()
    
    # LaTeX
    results["latex"] = {}
    for d in depths:
        results["latex"][f"B1_E_D{d}"] = f"{results['euclidean_betti1'].get(d, 0):.0f}"
        results["latex"][f"B1_H_D{d}"] = f"{results['hyperbolic_betti1'].get(d, 0):.0f}"
    
    max_d = max(depths)
    euc_max = results["euclidean_betti1"].get(max_d, 0)
    hyp_max = results["hyperbolic_betti1"].get(max_d, 0)
    results["latex"]["RESULT_E3_STATUS"] = "CONFIRMED" if (euc_max > 1 and hyp_max == 0) else "NOT CONFIRMED"
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Tree Embedding Protocols - Dual Regime")
    print("=" * 60)
    print(f"Device: {get_device()}")
    print(f"\nREGIME A (exact):     N ≤ {EXACT_THRESHOLD:,}")
    print(f"REGIME B (stochastic): N > {EXACT_THRESHOLD:,}")
    print(f"K = min({K_MULTIPLIER} × N, {K_MAX:,})")
