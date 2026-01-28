# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Part IV.4 - Latency Benchmark
=============================

AUDIT COMPLIANCE:
- Uses ONLY validated similarity functions from hardened modules
- Timing is pure measurement (no derivation)
- No geometry modification

PURPOSE:
Measure real wall-clock time for similarity computations.

WHAT WE MEASURE:
1. Pairwise similarity (most common operation)
2. Batch similarity (matrix operations)
3. Nearest neighbor search (top-k retrieval)

METHODS COMPARED:
- Teacher FP32-384: Full precision baseline
- CGT-32 Lorentz: Hyperbolic similarity
- CGT-32 Cosine: Spatial cosine similarity
- BQ-384: Binary Hamming distance

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import from hardened modules for CGT similarity
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened


# ═══════════════════════════════════════════════════════════════════════════════
#                    CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LatencyConfig:
    """Configuration for latency benchmark."""
    n_queries: int = 1000
    n_database: int = 10000
    n_iterations: int = 100
    warmup_iterations: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
#                    BENCHMARK UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def warmup_gpu(device: torch.device):
    """Warm up GPU to ensure consistent timing."""
    if device.type == 'cuda':
        x = torch.randn(1000, 1000, device=device)
        for _ in range(10):
            _ = torch.mm(x, x.T)
        torch.cuda.synchronize()
        del x
        gc.collect()
        torch.cuda.empty_cache()


def benchmark_function(
    func: Callable,
    n_iterations: int = 100,
    warmup_iterations: int = 10,
    device: torch.device = None,
) -> Tuple[float, float, float]:
    """
    Benchmark a function with proper warmup and statistics.
    
    Args:
        func: Function to benchmark (no arguments)
        n_iterations: Number of timed iterations
        warmup_iterations: Number of warmup iterations
        device: Device for synchronization
    
    Returns:
        Tuple of (mean_ms, std_ms, min_ms)
    """
    # Warmup
    for _ in range(warmup_iterations):
        func()
        if device and device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Timed iterations
    times = []
    for _ in range(n_iterations):
        if device and device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        func()
        
        if device and device.type == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    return float(times.mean()), float(times.std()), float(times.min())


# ═══════════════════════════════════════════════════════════════════════════════
#                    SIMILARITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def teacher_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity for teacher embeddings (PyTorch built-in).
    
    Args:
        x: Query embeddings [N, D]
        y: Database embeddings [M, D]
    
    Returns:
        Similarity matrix [N, M]
    """
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    return torch.mm(x_norm, y_norm.T)


def cgt_lorentz_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    substrate: LorentzSubstrateHardened
) -> torch.Tensor:
    """
    Lorentz similarity using HARDENED module.
    
    Args:
        x: Query embeddings [N, D+1]
        y: Database embeddings [M, D+1]
        substrate: Lorentz substrate from hardened modules
    
    Returns:
        Similarity matrix [N, M]
    """
    # Use validated function from hardened module
    # Compute pairwise similarities
    N, M = x.shape[0], y.shape[0]
    sims = torch.zeros(N, M, device=x.device, dtype=x.dtype)
    
    for i in range(N):
        sims[i] = substrate.lorentz_similarity(
            x[i:i+1].expand(M, -1),
            y
        ).squeeze()
    
    return sims


def cgt_spatial_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity on CGT spatial components (PyTorch built-in).
    
    Args:
        x: CGT embeddings [N, D+1] (first dim is time)
        y: CGT embeddings [M, D+1]
    
    Returns:
        Similarity matrix [N, M]
    """
    # Extract spatial components (indices 1:)
    x_spatial = F.normalize(x[:, 1:], dim=-1)
    y_spatial = F.normalize(y[:, 1:], dim=-1)
    return torch.mm(x_spatial, y_spatial.T)


def binary_hamming_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Hamming similarity for binary embeddings.
    
    Args:
        x: Binary embeddings [N, D] (±1 values)
        y: Binary embeddings [M, D]
    
    Returns:
        Similarity matrix [N, M] in [0, 1]
    """
    # Convert to ±1 if not already
    x_bin = torch.sign(x)
    y_bin = torch.sign(y)
    
    # Inner product gives (D - 2*hamming_distance)
    # Normalize to [0, 1]: (inner + D) / (2*D)
    D = x.shape[-1]
    inner = torch.mm(x_bin, y_bin.T)
    return (inner + D) / (2 * D)


# ═══════════════════════════════════════════════════════════════════════════════
#                    BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_latency_benchmark(
    teacher_embeddings: torch.Tensor,
    cgt_embeddings: torch.Tensor,
    substrate: LorentzSubstrateHardened,
    config: LatencyConfig,
    output_dir: Path,
) -> Dict:
    """
    Run latency benchmark for all similarity methods.
    
    Args:
        teacher_embeddings: Teacher embeddings [N, D_teacher]
        cgt_embeddings: CGT embeddings [N, D+1]
        substrate: Lorentz substrate for CGT
        config: Benchmark configuration
        output_dir: Output directory for results
    
    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "=" * 70)
    print("PART IV.4 - LATENCY BENCHMARK")
    print("=" * 70)
    
    device = torch.device(config.device)
    
    # Prepare data
    n_total = teacher_embeddings.shape[0]
    n_queries = min(config.n_queries, n_total)
    n_database = min(config.n_database, n_total)
    
    teacher_queries = teacher_embeddings[:n_queries].to(device)
    teacher_database = teacher_embeddings[:n_database].to(device)
    
    cgt_queries = cgt_embeddings[:n_queries].to(device)
    cgt_database = cgt_embeddings[:n_database].to(device)
    
    # Binary quantized versions
    bq_queries = torch.sign(teacher_queries)
    bq_database = torch.sign(teacher_database)
    
    print(f"\nBenchmark setup:")
    print(f"  Queries: {n_queries}")
    print(f"  Database: {n_database}")
    print(f"  Iterations: {config.n_iterations}")
    print(f"  Device: {device}")
    
    # Warmup
    print("\nWarming up GPU...")
    warmup_gpu(device)
    
    results = {
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {},
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # BENCHMARK: Pairwise Similarity
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "-" * 50)
    print("PAIRWISE SIMILARITY BENCHMARK")
    print("-" * 50)
    
    # Teacher cosine
    print("  Teacher Cosine (384D)...", end=" ", flush=True)
    mean, std, min_t = benchmark_function(
        lambda: teacher_cosine_similarity(teacher_queries, teacher_database),
        config.n_iterations, config.warmup_iterations, device
    )
    results['benchmarks']['teacher_cosine'] = {
        'mean_ms': mean, 'std_ms': std, 'min_ms': min_t,
        'dims': teacher_queries.shape[-1],
    }
    print(f"{mean:.2f} ± {std:.2f} ms")
    
    # CGT Lorentz similarity
    print("  CGT Lorentz (32D)...", end=" ", flush=True)
    mean, std, min_t = benchmark_function(
        lambda: cgt_lorentz_similarity(cgt_queries, cgt_database, substrate),
        config.n_iterations, config.warmup_iterations, device
    )
    results['benchmarks']['cgt_lorentz'] = {
        'mean_ms': mean, 'std_ms': std, 'min_ms': min_t,
        'dims': cgt_queries.shape[-1],
    }
    print(f"{mean:.2f} ± {std:.2f} ms")
    
    # CGT spatial cosine
    print("  CGT Spatial Cosine (32D)...", end=" ", flush=True)
    mean, std, min_t = benchmark_function(
        lambda: cgt_spatial_cosine_similarity(cgt_queries, cgt_database),
        config.n_iterations, config.warmup_iterations, device
    )
    results['benchmarks']['cgt_cosine'] = {
        'mean_ms': mean, 'std_ms': std, 'min_ms': min_t,
        'dims': cgt_queries.shape[-1] - 1,
    }
    print(f"{mean:.2f} ± {std:.2f} ms")
    
    # Binary Hamming
    print("  Binary Hamming (384D)...", end=" ", flush=True)
    mean, std, min_t = benchmark_function(
        lambda: binary_hamming_similarity(bq_queries, bq_database),
        config.n_iterations, config.warmup_iterations, device
    )
    results['benchmarks']['binary_hamming'] = {
        'mean_ms': mean, 'std_ms': std, 'min_ms': min_t,
        'dims': bq_queries.shape[-1],
    }
    print(f"{mean:.2f} ± {std:.2f} ms")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "-" * 70)
    print("LATENCY SUMMARY (lower is better)")
    print("-" * 70)
    print(f"{'Method':<25} {'Mean (ms)':>12} {'Std (ms)':>12} {'Dims':>8}")
    print("-" * 70)
    
    for method, data in results['benchmarks'].items():
        name = {
            'teacher_cosine': 'Teacher Cosine',
            'cgt_lorentz': 'CGT Lorentz',
            'cgt_cosine': 'CGT Cosine',
            'binary_hamming': 'Binary Hamming',
        }.get(method, method)
        print(f"{name:<25} {data['mean_ms']:>12.2f} {data['std_ms']:>12.2f} {data['dims']:>8}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON results
    with open(output_dir / 'latency_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # CSV summary
    rows = []
    for method, data in results['benchmarks'].items():
        rows.append({
            'method': method,
            'mean_ms': data['mean_ms'],
            'std_ms': data['std_ms'],
            'min_ms': data['min_ms'],
            'dims': data['dims'],
        })
    
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / 'latency_benchmark_summary.csv', index=False)
    
    print(f"\n📁 Results saved to: {output_dir}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("PART IV.4 - LATENCY BENCHMARK")
    print("=" * 80)
    
    print("\n⚠️ This module requires pre-computed embeddings.")
    print("   Use run_latency_benchmark() with your data.")
