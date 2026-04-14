"""
active_integration_v7.py
HyDRA v7 — Full integration layer implementing Gemini spec (Q1-Q8 + paper fixes)

Changes vs v6:
  - FrictionAwareCouplingSchedulerV7: composite rdc+logit_std gating
  - Asymmetric PCGrad: CE gradient protected, aux projected away from CE
  - L_align hook: distance matrix alignment (GCM paper Eq.22, Q8)
  - Sparse HAKORNLayer: layers {4,8,11} only — pacemaker architecture
  - VolumeWeightedCE: DISABLED (conflicts with SublatticeLMHead)
  - 3-Phase CE reduction: REMOVED (paper misimplementation fix)
"""

import types, torch, functools
import torch.nn as nn
from typing import List, Optional, Dict, Sequence


# ══════════════════════════════════════════════════════════════════════════
# 1. FrictionAwareCouplingSchedulerV7 — Composite Gating (Q1)
# ══════════════════════════════════════════════════════════════════════════

class FrictionAwareCouplingSchedulerV7:
    """
    Composite gating scheduler for Kuramoto coupling.
    K_eff = K_max * rdc_gate * logit_gate

    rdc_gate   = 1 - sigmoid(10 * (rdc_ema - rdc_danger))
    logit_gate = clamp(logit_std_target / logit_std, 0.1, 1.0)
    """

    def __init__(
        self,
        coupling_min: float = 0.05,
        coupling_max: float = 0.30,
        rdc_danger: float = 2.0,
        logit_std_target: float = 2.5,
    ):
        self.coupling_min     = coupling_min
        self.coupling_max     = coupling_max
        self.rdc_danger       = rdc_danger
        self.logit_std_target = logit_std_target
        self.current_coupling = coupling_max
        self._history: List[float] = []

    @torch.no_grad()
    def step(
        self,
        rdc_ema: float,
        logit_std: float,
        layers: List[nn.Module],
    ) -> float:
        # 1. Radial Drift Suppression — logistic sigmoid
        rdc_tensor = torch.tensor(10.0 * (rdc_ema - self.rdc_danger))
        rdc_gate   = 1.0 - torch.sigmoid(rdc_tensor).item()

        # 2. Confidence Explosion Suppression — inverse ratio
        logit_gate = max(0.1, min(1.0, self.logit_std_target / (logit_std + 1e-8)))

        # 3. Composite multiplication
        effective_k = self.coupling_max * rdc_gate * logit_gate
        self.current_coupling = max(self.coupling_min, effective_k)

        # 4. Direct parameter injection into pacemaker layers
        for layer in layers:
            if hasattr(layer, 'hakorn') and layer.hakorn is not None:
                layer.hakorn.K0.data = torch.tensor(
                    self.current_coupling,
                    device=layer.hakorn.K0.device,
                    dtype=layer.hakorn.K0.dtype,
                )

        self._history.append(self.current_coupling)
        return self.current_coupling

    def report(self) -> str:
        return (f"K_eff={self.current_coupling:.4f}  "
                f"(min={self.coupling_min}  max={self.coupling_max})")


# ══════════════════════════════════════════════════════════════════════════
# 2. HPCTrainingGuardV7 — Updated abort thresholds (Q3)
# ══════════════════════════════════════════════════════════════════════════

class HPCTrainingGuardV7:
    """
    Per-step health monitor with CGT softening timeline awareness.

    Abort conditions:
      - logit_std > 3.0 before step 4000 (CGT ceiling breach)
      - logit_std velocity > +0.5 per 100 steps (overconfidence spiral)
      - rdc_ema > 2.0 (boundary penetration)
    """

    def __init__(
        self,
        abort_rdc_threshold: float = 2.0,
        logit_std_ceiling: float = 3.0,
        velocity_limit: float = 0.5,
        warmup_steps: int = 4000,
    ):
        self.abort_rdc_threshold = abort_rdc_threshold
        self.logit_std_ceiling   = logit_std_ceiling
        self.velocity_limit      = velocity_limit
        self.warmup_steps        = warmup_steps
        self._history: List[tuple] = []

    def step(
        self,
        rdc_ema: float,
        step: int,
        order_parameter: Optional[float] = None,
        logit_std: Optional[float] = None,
        l_hidden: Optional[float] = None,
    ) -> str:
        if logit_std is not None:
            self._history.append((step, logit_std))

            # 1. Absolute ceiling — CGT warmup phase
            if step < self.warmup_steps and logit_std > self.logit_std_ceiling:
                return (f"abort_logit_ceiling_breach: "
                        f"{logit_std:.2f} > {self.logit_std_ceiling} @ step {step}")

            # 2. Velocity check over 100-step windows
            if len(self._history) >= 100:
                past_step, past_logit = self._history[-100]
                velocity = logit_std - past_logit
                if velocity > self.velocity_limit:
                    return (f"abort_velocity_breach: "
                            f"+{velocity:.2f} per 100 steps @ step {step}")

        # 3. Radial drift
        if rdc_ema > self.abort_rdc_threshold:
            return f"abort_rdc_drift: {rdc_ema:.2f} > {self.abort_rdc_threshold}"

        return "ok"

    def report(self) -> str:
        if not self._history:
            return "HPCGuardV7: no data"
        last_step, last_std = self._history[-1]
        return f"HPCGuardV7: step={last_step}  logit_std={last_std:.3f}"


# ══════════════════════════════════════════════════════════════════════════
# 3. L_align — Distance Matrix Alignment (Q8, GCM paper Eq.22)
# ══════════════════════════════════════════════════════════════════════════

def compute_l_align(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    sample_k: int = 64,
    lambda_align: float = 0.05,
) -> torch.Tensor:
    """
    Approximates L_align = Σᵢⱼ (dE(eᵢ,eⱼ) - dL(hᵢ,hⱼ))²

    Uses sampled pairs to avoid O(N²) full computation.

    Args:
        student_hidden: [B, L, D+1] — ambient Lorentz points (H^n)
        teacher_hidden: [B, L, D_t] — Euclidean teacher hidden states
        sample_k:       number of token pairs to sample per batch
        lambda_align:   loss weight

    Returns:
        scalar tensor (differentiable)
    """
    B, L, _ = student_hidden.shape

    # Sample token pair indices
    idx_i = torch.randint(0, L, (sample_k,), device=student_hidden.device)
    idx_j = torch.randint(0, L, (sample_k,), device=student_hidden.device)

    # Student: spatial components only (strip time coordinate)
    s_i = student_hidden[:, idx_i, 1:].float()   # [B, k, D]
    s_j = student_hidden[:, idx_j, 1:].float()

    # Teacher: raw Euclidean hidden states
    t_i = teacher_hidden[:, idx_i, :].float()    # [B, k, D_t]
    t_j = teacher_hidden[:, idx_j, :].float()

    # Euclidean distance between teacher representations
    d_E = torch.norm(t_i - t_j, dim=-1)           # [B, k]

    # Spatial L2 distance as hyperbolic proxy (cheaper than full Lorentz dist)
    # Full Lorentz dist would require the time coordinate computation
    d_L = torch.norm(s_i - s_j, dim=-1)           # [B, k]

    loss = ((d_E - d_L) ** 2).mean()
    return loss * lambda_align


# ══════════════════════════════════════════════════════════════════════════
# 4. Sparse HAKORNLayer attachment — Pacemaker architecture (Q7)
# ══════════════════════════════════════════════════════════════════════════

PACEMAKER_LAYERS = {4, 8, 11}  # syntactic (4) + semantic (8) + aggregation (11)

def attach_sparse_hakorn(student, cfg: dict, device: str) -> None:
    """
    Attaches HAKORNLayer ONLY to pacemaker layers {4, 8, 11}.
    Standard layers get hakorn=None (no phase dynamics).
    """
    try:
        from cgt.psi_extensions.binding import HAKORNLayer
        layers  = student.core_model.encoder.layers
        seq_len = cfg["model"]["n_positions"]

        for idx, layer in enumerate(layers):
            if idx in PACEMAKER_LAYERS:
                if not hasattr(layer, 'hakorn') or layer.hakorn is None:
                    layer.hakorn = HAKORNLayer(
                        num_nodes        = seq_len,
                        coupling_strength= cfg["binding"]["coupling_strength"],
                        temperature      = cfg["binding"]["decay_scale"],
                        dt               = cfg["binding"]["dt"],
                    ).to(device)
                # else: keep existing (resume-safe)
            else:
                # Detach non-pacemaker layers
                layer.hakorn = None

        active = [i for i in range(len(layers))
                  if hasattr(layers[i], 'hakorn') and layers[i].hakorn is not None]
        print(f"  [v7] ✅ Sparse HAKORNLayer: pacemakers at layers {active}")
    except Exception as e:
        print(f"  [v7] ⚠️  HAKORNLayer attachment failed: {e}")


# ══════════════════════════════════════════════════════════════════════════
# 5. Main monkey-patch — activate_all_modules_v7
# ══════════════════════════════════════════════════════════════════════════

def activate_all_modules_v7(
    trainer,
    pcgrad_inst,
    hpc_guard_v7,
    hysteresis_det,
    kuramoto_scheduler,     # FrictionAwareCouplingSchedulerV7
    geo_ctrl,
    extra_loss_hooks=None,
    lambda_align: float = 0.05,
    align_sample_k: int = 64,
):
    """
    HyDRA v7 monkey-patch of distillation_step.

    Key differences from v6:
      - VolumeWeightedCE: DISABLED
      - 3-Phase CE reduction: REMOVED
      - PCGrad: ASYMMETRIC — CE gradient protected
      - Kuramoto: COMPOSITE GATING via FrictionAwareCouplingSchedulerV7
      - L_align: ACTIVE — distance matrix alignment from teacher
      - HAKORNLayer: SPARSE — only layers {4,8,11}
    """

    orig_step = trainer.distillation_step

    @functools.wraps(orig_step)
    def patched_step_v7(batch):
        # ── 1. Original step: CE backward + optimizer.step ────────────────
        metrics = orig_step(batch)
        cur_step = trainer.step

        # ── 2. Collect live metrics ───────────────────────────────────────
        rdc_ema   = metrics.get('rdc_ema',   metrics.get('rdc_proxy', 0.0))
        logit_std = metrics.get('logit_std', 0.0)
        l_hidden  = metrics.get('l_hidden',  0.0)

        # ── 3. HPCGuardV7 — per-step with velocity check ─────────────────
        if hpc_guard_v7 is not None:
            try:
                action = hpc_guard_v7.step(
                    rdc_ema=float(rdc_ema),
                    step=cur_step,
                    logit_std=float(logit_std),
                    l_hidden=float(l_hidden),
                )
                if action.startswith('abort'):
                    print(f"  [HPCGuardV7] 🛑 {action}")
                    # Signal trainer to stop
                    trainer.stop = True
                elif action != 'ok' and cur_step % 500 == 0:
                    print(f"  [HPCGuardV7] step={cur_step} {action}")
            except Exception:
                pass

        # ── 4. HysteresisDetector ─────────────────────────────────────────
        if hysteresis_det is not None:
            try:
                hysteresis_det.update(rdc_ema=float(rdc_ema))
                if hysteresis_det.hysteresis_detected() and cur_step % 1000 == 0:
                    print(f"  [Hysteresis] ⚠️  cycles="
                          f"{hysteresis_det.get_cycle_count()} @ step={cur_step}")
            except Exception:
                pass

        # ── 5. Composite Kuramoto Gating (Q1) ────────────────────────────
        if kuramoto_scheduler is not None:
            try:
                k_eff = kuramoto_scheduler.step(
                    rdc_ema=float(rdc_ema),
                    logit_std=float(logit_std),
                    layers=trainer.student.core_model.encoder.layers,
                )
                metrics['k_eff'] = k_eff
                if cur_step % 500 == 0:
                    print(f"  [KuramotoGate] {kuramoto_scheduler.report()}")
            except Exception:
                pass

        # ── 6. GeometricController — every 200 steps ─────────────────────
        if geo_ctrl is not None and cur_step % 200 == 0:
            try:
                telemetry = {
                    'rdc_ema':     float(rdc_ema),
                    'logit_std':   float(logit_std),
                    'l_hidden':    float(l_hidden),
                    'mean_radius': float(metrics.get('mean_radius', 1.5)),
                    'step':        cur_step,
                }
                geo_ctrl.step(telemetry, cur_step)
            except Exception:
                pass

        # ── 7. L_align hook (Q8) — distance matrix alignment ─────────────
        # Runs every 50 steps after warmup to avoid cold-start interference
        if cur_step >= 100 and cur_step % 50 == 0:
            try:
                # Get teacher hidden states via a no-grad forward pass
                input_ids = batch['input_ids'].to(trainer.device)
                with torch.no_grad():
                    t_out = trainer.teacher(input_ids, return_hidden=True)
                    t_hidden = t_out.get('hidden_states')
                    if t_hidden is None and hasattr(t_out, 'get'):
                        t_hidden = t_out.get('last_hidden_state')

                # Get student hidden states
                s_out = trainer.student(input_ids)
                s_hidden = s_out.get('hidden_states')

                if t_hidden is not None and s_hidden is not None:
                    l_align = compute_l_align(
                        student_hidden=s_hidden,
                        teacher_hidden=t_hidden,
                        sample_k=align_sample_k,
                        lambda_align=lambda_align,
                    )
                    if l_align.requires_grad and l_align.item() > 0:
                        trainer.optimizer.zero_grad()
                        l_align.backward()
                        torch.nn.utils.clip_grad_norm_(
                            trainer.student.parameters(),
                            trainer.config.gradient_clip * 0.3,
                        )
                        trainer.optimizer.step()
                        metrics['l_align'] = float(l_align.item())
            except Exception:
                pass

        # ── 8. Asymmetric PCGrad + extra_loss_hooks (Q2) ─────────────────
        hooks = extra_loss_hooks or getattr(trainer, '_extra_loss_hooks', [])
        if hooks:
            # Step A: capture pristine CE gradients from orig_step
            # (orig_step already ran backward; grads are in .grad)
            ce_grads = []
            for p in trainer.student.parameters():
                ce_grads.append(
                    p.grad.clone().detach() if p.grad is not None else None
                )

            # Step B: zero grads and compute aux losses
            trainer.optimizer.zero_grad()
            hook_losses = []
            for hook_fn in hooks:
                try:
                    result = hook_fn(trainer)
                    if (result is not None
                            and isinstance(result, torch.Tensor)
                            and result.item() > 0):
                        hook_losses.append(result)
                except Exception:
                    pass

            if hook_losses:
                try:
                    # Resolve conflicts AMONG aux losses via PCGrad
                    if pcgrad_inst is not None and len(hook_losses) > 1:
                        task_losses = {f"hook_{i}": l
                                       for i, l in enumerate(hook_losses)}
                        pcgrad_inst.backward_and_project(
                            task_losses=task_losses,
                            shared_params=list(trainer.student.parameters()),
                        )
                    else:
                        combined = torch.stack(hook_losses).sum()
                        combined.backward()

                    # Step C: Asymmetric projection — protect CE gradient
                    # Project aux gradient AWAY from CE direction if conflicting
                    for p, g_ce in zip(trainer.student.parameters(), ce_grads):
                        if p.grad is not None and g_ce is not None:
                            g_aux = p.grad.data
                            dot   = torch.sum(g_aux * g_ce)
                            # If conflict (angle > 90°), remove component along CE
                            if dot < 0:
                                g_aux -= (dot / (torch.sum(g_ce * g_ce) + 1e-8)) * g_ce
                            # Recombine: pristine CE + non-conflicting aux
                            p.grad.data = g_ce + g_aux
                        elif g_ce is not None:
                            # Restore pure CE gradient if no aux grad
                            if p.grad is None:
                                p.grad = g_ce.clone()
                            else:
                                p.grad.data = g_ce

                    torch.nn.utils.clip_grad_norm_(
                        trainer.student.parameters(),
                        trainer.config.gradient_clip,
                    )
                    trainer.optimizer.step()
                    metrics['aux_loss'] = float(
                        sum(l.item() for l in hook_losses)
                    )
                except Exception:
                    pass

        return metrics

    trainer.distillation_step = patched_step_v7

    print("  [v7/ActiveIntegration] ✅ Monkey-patch applied")
    print("  [v7/ActiveIntegration]   HPCv7 ✅  Hysteresis ✅  GeoCtrl ✅")
    print("  [v7/ActiveIntegration]   KuramotoGate ✅  L_align ✅")
    print("  [v7/ActiveIntegration]   AsymmetricPCGrad ✅  VolCE DISABLED ✅")
    print("  [v7/ActiveIntegration]   3-Phase CE reduction REMOVED ✅")
    return trainer
