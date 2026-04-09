# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
geometric_distillation.py
=========================
Losses estruturais para prevenir Degenerate Equilibrium (DegEq).

Hierarquia teórica:
    ProjectiveKLLoss         — Direction 1: baseline ablation
    DecoupledRadialAngular   — Direction 3: solução estrutural primária
    DistanceCorrMetric       — F2: métrica de diagnóstico (não treinável)

Fundamento:
    DegEq ocorre quando o objetivo não respeita a decomposição do tensor
    métrico hiperbólico g_H = dr² ⊕ sinh²(r) g_{S^{n-1}}.
    A solução estrutural (D3) faz o objetivo respeitar essa geometria
    separando os gradientes nos sub-feixes radial e angular.

Paper claim central:
    "Instead of replacing KL, we factorize its geometric inconsistencies."
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Direction 1 — Projective KL  (baseline ablation)
# ─────────────────────────────────────────────────────────────────────────────

class ProjectiveKLLoss(nn.Module):
    """
    KL no subespaço angular puro com elastic radius anchor.

    Fix do Wandering Radius (seção 2.3 do documento de avaliação):
    ∂L_KL/∂r = 0 por construção → r entra no espaço nulo do Hessiano
    → random walk sob AdamW → instabilidade numérica em runs longas.

    O elastic anchor mantém r perto de r_fixed com tolerância delta,
    sem forçar raio uniforme (o que reintroduziria o problema da D2).

    Papel no paper: "naive baseline" no ablation study — mostra que
    controlar escala sem desacoplar estrutura é insuficiente.
    """

    def __init__(
        self,
        substrate,
        r_fixed:       float = 1.5,
        temperature:   float = 1.0,
        anchor_weight: float = 0.01,
        anchor_delta:  float = 0.2,   # tolerância elástica
    ):
        super().__init__()
        self.substrate      = substrate
        self.r_fixed        = r_fixed
        self.T              = temperature
        self.anchor_weight  = anchor_weight
        self.anchor_delta   = anchor_delta

    def _project_angular(self, h: torch.Tensor) -> torch.Tensor:
        """Re-embed na geodesic sphere S_{r_fixed} preservando só direção."""
        orig_shape = h.shape
        h_flat = h.reshape(-1, h.shape[-1])
        v      = self.substrate.log_map_zero(h_flat).float()
        v_dir  = F.normalize(v[:, 1:], dim=-1)
        r0     = torch.tensor(self.r_fixed, dtype=v.dtype, device=v.device)
        spatial = torch.sinh(r0) * v_dir
        time    = torch.cosh(r0).expand(len(v_dir), 1)
        return torch.cat([time, spatial], dim=-1).reshape(orig_shape)

    def forward(
        self,
        h_student: torch.Tensor,   # [B, L, n+1]
        W_vocab:   torch.Tensor,   # [V, n+1]
        p_teacher: torch.Tensor,   # [B, L, V] soft targets
    ) -> dict:
        h_proj = self._project_angular(h_student)
        W_proj = self._project_angular(W_vocab)

        B, L, _ = h_proj.shape
        # Minkowski inner product — puro angular
        logits  = -(h_proj.reshape(B*L, -1) @ W_proj.T).reshape(B, L, -1)
        log_p   = F.log_softmax(logits / self.T, dim=-1)
        l_kl    = F.kl_div(
            log_p.reshape(-1, log_p.shape[-1]),
            p_teacher.reshape(-1, p_teacher.shape[-1]),
            reduction='batchmean',
        ) * self.T ** 2

        # Elastic radius anchor — previne Wandering Radius
        # Deadzone [r_fixed - delta, r_fixed + delta]: sem gradiente
        # Fora da deadzone: MSE cresce quadraticamente
        r_h     = h_student[..., 1:].norm(dim=-1)              # [B, L]
        r_dev   = (r_h - self.r_fixed).abs() - self.anchor_delta
        l_anchor = F.relu(r_dev).pow(2).mean()

        l_total = l_kl + self.anchor_weight * l_anchor
        return {
            'total':    l_total,
            'l_kl':     l_kl,
            'l_anchor': l_anchor,
            'r_mean':   r_h.mean(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Direction 3 — Decoupled Radial-Angular  (solução estrutural)
# ─────────────────────────────────────────────────────────────────────────────

class DecoupledRadialAngularLoss(nn.Module):
    """
    Decompõe distilação em dois objetivos ortogonais na métrica H^n.

    g_H = dr² ⊕ sinh²(r) g_{S^{n-1}}

    L_angular: ∂L/∂r = 0 — KL no espaço de direções puras
    L_radial:  ∂L/∂û = 0 — calibração via entropia do teacher

    r_target = σ(-H_norm) × r_max, onde H_norm = H_teacher / log(V)
    Normalização por log(V) garante invariância ao tamanho do vocabulário
    e à temperatura — necessário para generalização cross-dataset.

    Ontologia (seção 4.2 do documento):
        H alta  → incerteza → r_target próximo da origem
        H baixa → especificidade → r_target próximo de r_max
    Isso é consequência da geometria hiperbólica, não design choice.
    """

    def __init__(
        self,
        substrate,
        vocab_size:    int   = 50257,
        r_max:         float = 3.0,
        lambda_radial: float = 0.1,
        temperature:   float = 1.0,
    ):
        super().__init__()
        self.substrate     = substrate
        self.log_V         = math.log(vocab_size)   # normalização entropia
        self.r_max         = r_max
        self.lambda_radial = lambda_radial
        self.T             = temperature

    def _decompose(self, h: torch.Tensor):
        """
        h: [..., n+1] Lorentz ambient
        → r: [...]    raio geodésico
        → û: [..., n] direção unitária no tangente
        """
        orig = h.shape[:-1]
        h_flat = h.reshape(-1, h.shape[-1])
        v      = self.substrate.log_map_zero(h_flat).float()
        r      = v[:, 1:].norm(dim=-1)              # [N]
        u      = F.normalize(v[:, 1:], dim=-1)      # [N, n]
        return r.reshape(orig), u.reshape(*orig, -1)

    def forward(
        self,
        h_student:       torch.Tensor,   # [B, L, n+1]
        W_vocab:         torch.Tensor,   # [V, n+1]
        p_teacher:       torch.Tensor,   # [B, L, V] soft targets
        teacher_entropy: torch.Tensor,   # [B, L] H(p_teacher) por posição
    ) -> dict:
        r_s, u_s = self._decompose(h_student)          # [B,L], [B,L,n]
        _,   u_w = self._decompose(W_vocab)             # [V, n]

        B, L, n = u_s.shape

        # ── L_angular: KL na direção pura, ∂L/∂r = 0 ─────────────────────
        cos_logits = (
            u_s.reshape(B*L, 1, n) * u_w.unsqueeze(0)
        ).sum(-1).reshape(B, L, -1)                    # [B, L, V]

        log_p_ang = F.log_softmax(cos_logits / self.T, dim=-1)
        l_angular = F.kl_div(
            log_p_ang.reshape(-1, cos_logits.shape[-1]),
            p_teacher.reshape(-1, p_teacher.shape[-1]),
            reduction='batchmean',
        ) * self.T ** 2

        # ── L_radial: calibração via entropia normalizada, ∂L/∂û = 0 ──────
        # Normaliza por log(V) — invariância ao vocab size e temperatura
        H_norm   = (teacher_entropy / self.log_V).clamp(0.0, 1.0)  # [B, L]
        r_target = torch.sigmoid(-H_norm) * self.r_max              # [B, L]
        l_radial = F.mse_loss(r_s, r_target.detach())

        l_total = l_angular + self.lambda_radial * l_radial
        return {
            'total':    l_total,
            'l_angular':   l_angular,
            'l_radial':    l_radial,
            'r_mean':      r_s.mean(),
            'r_target':    r_target.mean(),
            'H_norm_mean': H_norm.mean(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# F2 — Distance Correlation Metric  (diagnóstico, não treinável)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_f2(
    h_student:      torch.Tensor,   # [B, n+1] last-token pooled
    teacher_embeds: torch.Tensor,   # [B, D_teacher]
    substrate,
    eps: float = 1e-6,
) -> float:
    """
    F2: correlação de Pearson entre matrizes de distância do student e teacher.

    F2 → 1: estrutura geométrica do student alinha com a do teacher.
    F2 → 0: DegEq — student perdeu estrutura relacional.

    Métrica crítica para Paper 2:
    - PPL sozinha não prova que estrutura foi preservada
    - F2 prova que geometria aprendida é consistente com o teacher
    - Argumento contra claim "DegEq is harmless if PPL is low"
    """
    B = h_student.shape[0]
    if B < 4:
        return float('nan')

    # Distâncias geodésicas do student
    h64    = h_student.double()
    inner  = h64[:, 1:] @ h64[:, 1:].T - h64[:, 0:1] @ h64[:, 0:1].T
    inner  = inner.clamp(max=-1.0 - eps)
    D_s    = torch.acosh(-inner).float()                # [B, B]

    # Distâncias coseno do teacher
    e_n    = F.normalize(teacher_embeds.float(), dim=-1)
    D_t    = (1.0 - e_n @ e_n.T).clamp(min=0)          # [B, B]

    # Apenas triângulo superior (sem diagonal)
    mask   = torch.triu(torch.ones(B, B, dtype=torch.bool), diagonal=1)
    d_s    = D_s[mask]
    d_t    = D_t[mask]

    # Correlação de Pearson
    d_s_c  = d_s - d_s.mean()
    d_t_c  = d_t - d_t.mean()
    num    = (d_s_c * d_t_c).sum()
    den    = (d_s_c.pow(2).sum() * d_t_c.pow(2).sum()).sqrt() + eps
    return (num / den).item()
