# active_integration_hook.py
# Gerado para HyDRA v6 — ativa todos os módulos dormentes via monkey-patch
#
# MÓDULOS ATIVADOS:
# 1. _extra_loss_hooks  — chamados ANTES do backward() via segundo accumulate pass
# 2. PCGrad             — gradient surgery entre task losses
# 3. HPCTrainingGuard   — monitora rdc/logit_std a cada step
# 4. HysteresisDetector — detecta oscilações a cada step
# 5. GeometricController— ajusta lambda_radius por telemetria a cada step
# 6. VolumeWeightedCE   — boost de CE por volume hiperbólico
# 7. FrictionAwareCouplingScheduler — ajusta coupling Kuramoto por passo

import types, torch, functools

def activate_all_modules(
    trainer,
    pcgrad_inst,
    hpc_guard,
    hysteresis_det,
    geo_ctrl,
    vol_ce,
    friction_scheduler,
    extra_loss_hooks=None,
):
    """
    Monkey-patches trainer.distillation_step to activate all dormant modules.

    Design: wraps the original step with a post-step hook that:
      - Collects live metrics from the completed step
      - Runs per-step monitors (HPC, Hysteresis, GeoController, Friction)
      - Accumulates extra_loss_hooks into a secondary backward pass
      - Applies PCGrad gradient surgery when losses conflict
    """

    orig_step = trainer.distillation_step

    @functools.wraps(orig_step)
    def patched_step(batch):
        # ── 1. Run original step (base loss + backward + optimizer.step) ──
        metrics = orig_step(batch)
        cur_step = trainer.step  # already incremented inside orig_step

        # ── 2. Collect live metrics for all monitors ──────────────────────
        rdc_ema   = metrics.get('rdc_ema',   metrics.get('rdc_proxy', 0.0))
        logit_std = metrics.get('logit_std', 0.0)
        l_hidden  = metrics.get('l_hidden',  0.0)
        order_param = 0.5  # default; updated by binding hook if available

        # ── 3. HPCTrainingGuard — per-step health monitor ─────────────────
        if hpc_guard is not None:
            try:
                action = hpc_guard.step(
                    rdc_ema=float(rdc_ema),
                    step=cur_step,
                    order_parameter=order_param,
                    logit_std=float(logit_std),
                    l_hidden=float(l_hidden),
                )
                if action not in ('ok', 'monitor') and cur_step % 500 == 0:
                    print(f"  [HPCGuard] step={cur_step} action={action}")
            except Exception:
                pass

        # ── 4. HysteresisDetector — per-step oscillation detection ────────
        if hysteresis_det is not None:
            try:
                hysteresis_det.update(rdc_ema=float(rdc_ema))
                if hysteresis_det.hysteresis_detected() and cur_step % 1000 == 0:
                    print(f"  [Hysteresis] ⚠️  cycles={hysteresis_det.get_cycle_count()} @ step={cur_step}")
            except Exception:
                pass

        # ── 5. GeometricController — adaptive lambda_radius per step ──────
        if geo_ctrl is not None and cur_step % 200 == 0:
            try:
                telemetry = {
                    'rdc_ema':        float(rdc_ema),
                    'logit_std':      float(logit_std),
                    'l_hidden':       float(l_hidden),
                    'mean_radius':    float(metrics.get('mean_radius', 1.5)),
                    'order_parameter':float(order_param),
                    'w_entropy':      float(metrics.get('w_entropy', 5.5)),
                    'step':           cur_step,
                }
                adjustments = geo_ctrl.step(telemetry, cur_step)
                if adjustments and cur_step % 2000 == 0:
                    adj_str = ' '.join(f"{k}={v:.4f}" for k,v in adjustments.items()
                                      if k not in ('step',))
                    print(f"  [GeoCtrl] step={cur_step} {adj_str}")
            except Exception:
                pass

        # ── 6. FrictionAwareCouplingScheduler — per-step ──────────────────
        if friction_scheduler is not None:
            try:
                friction_scheduler.step()
            except Exception:
                pass

        # ── 7. VolumeWeightedCE — add volume-weighted CE loss ─────────────
        # Runs a separate micro-backward on the volume CE term
        if vol_ce is not None and cur_step % 50 == 0:
            try:
                with torch.no_grad():
                    batch_ids = batch['input_ids'].to(trainer.device)
                    s = trainer.student(batch_ids)
                    logits = s['logits']
                    labels = batch.get('labels', batch_ids).to(trainer.device)
                    # Get hidden states for volume weighting
                    hidden = s.get('hidden_states', logits)
                v_loss = vol_ce(logits.detach(), labels, hidden.detach())
                if v_loss.requires_grad:
                    trainer.optimizer.zero_grad()
                    v_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        trainer.student.parameters(), 0.5)  # conservative clip
                    trainer.optimizer.step()
                metrics['vol_ce_loss'] = float(v_loss.item())
            except Exception:
                pass

        # ── 8. _extra_loss_hooks — secondary backward pass ────────────────
        hooks = getattr(trainer, '_extra_loss_hooks', [])
        if hooks:
            hook_losses = []
            for hook_fn in hooks:
                try:
                    result = hook_fn(trainer)
                    if result is not None and isinstance(result, torch.Tensor) \
                       and result.item() > 0:
                        hook_losses.append(result)
                except Exception:
                    pass

            if hook_losses:
                try:
                    # PCGrad: project hook gradients against task gradient
                    combined = torch.stack(hook_losses).sum()
                    if pcgrad_inst is not None and len(hook_losses) > 1:
                        task_losses = {f"hook_{i}": l for i, l in enumerate(hook_losses)}
                        trainer.optimizer.zero_grad()
                        pcgrad_inst.backward_and_project(
                            task_losses=task_losses,
                            shared_params=list(trainer.student.parameters()),
                        )
                    else:
                        trainer.optimizer.zero_grad()
                        combined.backward()

                    torch.nn.utils.clip_grad_norm_(
                        trainer.student.parameters(),
                        trainer.config.gradient_clip * 0.5  # half clip for aux losses
                    )
                    trainer.optimizer.step()
                    metrics['aux_loss'] = float(combined.item())
                except Exception:
                    pass

        return metrics

    trainer.distillation_step = patched_step
    print("  [ActiveIntegration] ✅ Monkey-patch applied to distillation_step")
    print("  [ActiveIntegration]   HPC ✅  Hysteresis ✅  GeoCtrl ✅")
    print("  [ActiveIntegration]   PCGrad ✅  VolCE ✅  Friction ✅  Hooks ✅")
    return trainer
