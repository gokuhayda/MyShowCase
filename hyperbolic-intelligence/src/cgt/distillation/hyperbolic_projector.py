import torch
import torch.nn as nn


class HiddenProjector(nn.Module):
    """
    v1 — Projeção euclidiana pura.

    Linear(768→128) + LayerNorm.
    Aprende a melhor projeção linear para maximizar cosine similarity.
    Nunca vê a manifold hiperbólica.
    Cego ao raio — alimenta o DegEq.
    """

    def __init__(self, in_dim=768, out_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h_teacher):
        return self.norm(self.proj(h_teacher))


class HyperbolicProjectorV2(nn.Module):
    """
    v2 — Projeção euclidiana + elevação para a manifold.

    Conceito: Linear → exp_map_zero.
    Problema: o raio de saída é determinado pela norma do output
    da linear layer, que cresce livremente durante o treino.
    O max_tangent_norm controla o teto mas não o valor aprendido —
    o projector ainda pode saturar no clamp e perder gradiente.
    Não resolve o DegEq estruturalmente.
    """

    def __init__(self, in_dim=768, out_dim=128, substrate=None,
                 max_tangent_norm=1.5):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.substrate = substrate
        self.max_tangent_norm = max_tangent_norm

    def forward(self, h_teacher):
        v = self.linear(h_teacher)                        # [B, L, n]
        v0 = torch.zeros_like(v[..., :1])
        v_full = torch.cat([v0, v], dim=-1)               # [B, L, n+1]
        return self.substrate.exp_map_zero(
            v_full, max_tangent_norm=self.max_tangent_norm
        )                                                  # [B, L, n+1] on H^n


class HyperbolicProjectorV3(nn.Module):
    """
    v3 — Projeção com raio aprendível e geometricamente controlado.

    Por que esta versão:
        v1 é cega ao raio → alimenta DegEq.
        v2 eleva para H^n mas o raio é subproduto da norma da linear,
        que pode saturar no clamp max_tangent_norm e matar o gradiente.

    Solução:
        Separar explicitamente direção e raio em dois ramos independentes.
        - Ramo angular: linear → L2 normalize → direção unitária em R^n
        - Ramo radial:  linear → sigmoid → escala para [r_min, r_max]
        O vetor tangente final é v = r * direcao, depois exp_map_zero.

    Consequências:
        - O gradiente do raio nunca satura (sigmoid é suave em todo domínio)
        - O gradiente da direção é independente do raio (normalização L2)
        - A loss geodésica downstream penaliza erros de raio e ângulo
          separadamente, tornando o DegEq visível no gradiente
        - Compatível com substrate.dist() como loss (sem cosine similarity)

    Desvantagem conhecida:
        Dois ramos = ~1.5x parâmetros vs v1.
        O ramo radial precisa de supervisão — sem loss geodésica,
        ele aprende r=r_mid trivialmente (ponto fixo do sigmoid escalado).
    """

    def __init__(self, in_dim=768, out_dim=128, substrate=None,
                 r_min=0.5, r_max=3.0):
        super().__init__()
        self.substrate = substrate
        self.r_min = r_min
        self.r_max = r_max

        # ramo angular — aprende direção
        self.dir_proj = nn.Linear(in_dim, out_dim, bias=False)

        # ramo radial — aprende profundidade geodésica
        self.rad_proj = nn.Linear(in_dim, 1, bias=True)

    def forward(self, h_teacher):
        """
        h_teacher: [B, L, in_dim]
        returns:   [B, L, n+1] on H^n
        """
        # direção unitária no espaço tangente
        direction = self.dir_proj(h_teacher)               # [B, L, n]
        direction = nn.functional.normalize(direction,
                                            p=2, dim=-1)   # unitário

        # raio geodésico aprendível em [r_min, r_max]
        r = torch.sigmoid(self.rad_proj(h_teacher))        # [B, L, 1]
        r = self.r_min + r * (self.r_max - self.r_min)    # escala

        # vetor tangente: r * direção, com v0=0 para exp_map_zero
        v_spatial = r * direction                          # [B, L, n]
        v0 = torch.zeros_like(v_spatial[..., :1])
        v_full = torch.cat([v0, v_spatial], dim=-1)        # [B, L, n+1]

        return self.substrate.exp_map_zero(v_full)         # [B, L, n+1]
