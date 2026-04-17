"""cgt.runners.aggregate — multi-seed aggregation & final report."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import ExperimentConfig
from .logger import read_jsonl


# ─────────────────────────────────────────────────────────────────────────────
# Per-experiment extraction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentSummary:
    variant:       str
    seed:          int
    completed:     bool
    final_step:    int
    final_ppl:     float
    final_cos_align: float
    final_gate_std:  float
    final_gate_mean: float
    final_corr:      float
    best_val_loss:   float
    best_val_step:   int
    wall_s:          float


def _last_with_key(rows: List[dict], key: str) -> Optional[dict]:
    for r in reversed(rows):
        if key in r and r.get(key) is not None:
            return r
    return None


def summarize_experiment(cfg: ExperimentConfig) -> ExperimentSummary:
    rows = read_jsonl(cfg.log_path)
    completed_event = next(
        (r for r in reversed(rows) if r.get("event") == "completed"), None)
    completed = completed_event is not None

    # Last eval row (phase=="eval") holds the final val_loss/ppl/gate/cos_align
    last_eval = next(
        (r for r in reversed(rows) if r.get("phase") == "eval"), None)
    last_train = _last_with_key(rows, "train_loss")

    # Any heartbeat with best_val_loss (written on completion)
    best_event = next(
        (r for r in reversed(rows) if r.get("best_val_loss") is not None), None)

    def _g(d, k, default=float("nan")):
        if d is None: return default
        v = d.get(k)
        return default if v is None else float(v)

    return ExperimentSummary(
        variant         = cfg.variant,
        seed            = cfg.seed,
        completed       = completed,
        final_step      = int(_g(completed_event, "global_step",
                                 _g(last_eval, "step", 0))),
        final_ppl       = _g(completed_event, "final_ppl", _g(last_eval, "ppl")),
        final_cos_align = _g(last_eval, "cos_align"),
        final_gate_std  = _g(last_eval, "gate_std"),
        final_gate_mean = _g(last_eval, "gate_mean"),
        final_corr      = _g(last_eval, "corr_attn_K"),
        best_val_loss   = _g(best_event, "best_val_loss"),
        best_val_step   = int(_g(best_event, "best_val_step", 0)),
        wall_s          = _g(completed_event, "wall_s", 0.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed aggregation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VariantStats:
    variant: str
    n_seeds: int
    ppl_mean:       float
    ppl_std:        float
    cos_mean:       float
    cos_std:        float
    gate_std_mean:  float
    gate_std_std:   float
    seeds:          List[int]


def _nanmean_std(xs) -> Tuple[float, float]:
    arr = np.array([x for x in xs if x is not None and math.isfinite(x)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_by_variant(summaries: Iterable[ExperimentSummary]) -> Dict[str, VariantStats]:
    by_variant: Dict[str, List[ExperimentSummary]] = {}
    for s in summaries:
        by_variant.setdefault(s.variant, []).append(s)
    out = {}
    for v, group in by_variant.items():
        ppl_m, ppl_s = _nanmean_std([g.final_ppl       for g in group])
        cos_m, cos_s = _nanmean_std([g.final_cos_align for g in group])
        gs_m,  gs_s  = _nanmean_std([g.final_gate_std  for g in group])
        out[v] = VariantStats(
            variant=v, n_seeds=len(group),
            ppl_mean=ppl_m, ppl_std=ppl_s,
            cos_mean=cos_m, cos_std=cos_s,
            gate_std_mean=gs_m, gate_std_std=gs_s,
            seeds=[g.seed for g in group],
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Report printing
# ─────────────────────────────────────────────────────────────────────────────

def print_final_report(stats: Dict[str, VariantStats],
                       reference_v8_1_ppl_delta: float = 1.34,
                       reference_v8_1_ppl_sigma: float = 0.11) -> None:
    print("=" * 78)
    print("  HyDRA v9 — Robust Results  (mean ± std over seeds)")
    print("=" * 78)
    print(f"  {'variant':<16} {'n':>3} {'PPL':<20} {'cos_align':<20} {'gate_std':<16}")
    print(f"  {'-'*16} {'-'*3} {'-'*20} {'-'*20} {'-'*16}")
    for v in ("v7", "v9", "v9_no_detach", "v9_gate_off"):
        if v not in stats: continue
        s = stats[v]
        ppl = f"{s.ppl_mean:7.3f} ± {s.ppl_std:.3f}" if math.isfinite(s.ppl_mean) else "n/a"
        cos = (f"{s.cos_mean:+.4f} ± {s.cos_std:.4f}"
               if math.isfinite(s.cos_mean) else "n/a (no gate)")
        gst = (f"{s.gate_std_mean:.4f} ± {s.gate_std_std:.4f}"
               if math.isfinite(s.gate_std_mean) else "n/a (no gate)")
        print(f"  {v:<16} {s.n_seeds:>3} {ppl:<20} {cos:<20} {gst:<16}")
    print(f"  {'-'*16} {'-'*3} {'-'*20} {'-'*20} {'-'*16}")
    # v8.1 reference from HLLM paper
    if "v7" in stats:
        ref_v7 = stats["v7"].ppl_mean
        ref_v81_ppl = ref_v7 + reference_v8_1_ppl_delta if math.isfinite(ref_v7) else float("nan")
        print(f"  {'v8.1 (HLLM ref)':<16} {'-':>3} "
              f"{ref_v81_ppl:7.3f} (Δ+{reference_v8_1_ppl_delta:.2f})   "
              f"{'-0.400':<20} {'0.0160':<16}")
    print("=" * 78)
    print()

    # Verdict
    if "v9" in stats and "v7" in stats:
        s9 = stats["v9"]; s7 = stats["v7"]
        cos_ok = math.isfinite(s9.cos_mean)      and s9.cos_mean > 0
        gst_ok = math.isfinite(s9.gate_std_mean) and s9.gate_std_mean > 0.05
        ppl_delta = s9.ppl_mean - s7.ppl_mean if (
            math.isfinite(s9.ppl_mean) and math.isfinite(s7.ppl_mean)) else float("nan")
        ppl_ok = math.isfinite(ppl_delta) and ppl_delta <= 0.02 * s7.ppl_mean

        max_std = max(s7.ppl_std, s9.ppl_std, 1e-6)
        sigma_gap = abs(ppl_delta) / max_std if math.isfinite(ppl_delta) else float("nan")

        print(f"  cos_align > 0    : {'✓' if cos_ok else '✗'}  "
              f"(mean = {s9.cos_mean:+.4f})")
        print(f"  gate_std > 0.05  : {'✓' if gst_ok else '✗'}  "
              f"(mean = {s9.gate_std_mean:.4f})")
        print(f"  PPL ≤ 1.02 × v7  : {'✓' if ppl_ok else '✗'}  "
              f"(Δ = {ppl_delta:+.3f}, {sigma_gap:.2f}σ)")
        print()
        if cos_ok and gst_ok:
            print("  VERDICT: SUCCESS — angular coupling resolves the v8.1 anti-alignment")
            print("  at scale, across multiple seeds and with stable gate selectivity.")
        else:
            print("  VERDICT: FAIL — hypothesis falsified at this scale; see paper for")
            print("  implications for the coupling paradigm.")
        print("=" * 78)


def save_report(stats: Dict[str, VariantStats], path: str | Path) -> Path:
    """Dump a JSON report for the paper."""
    import json
    from dataclasses import asdict
    out = {v: asdict(s) for v, s in stats.items()}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    return p
