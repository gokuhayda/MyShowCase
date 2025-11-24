import numpy as np
from statsmodels.stats.power import zt_ind_solve_power
from typing import Final

class ExperimentDesign:
    """
    Planejamento rigoroso para A/B Testing.
    Evita erros como:
    - P-Hacking (parar o experimento quando parece "bom")
    - Peeking (olhar resultados antes da hora)
    - Testes inviáveis por falta de tráfego
    
    Esta classe encapsula a matemática avançada e entrega 
    uma API simples voltada para negócios.
    """

    DEFAULT_ALPHA: Final[float] = 0.05   # 5%
    DEFAULT_POWER: Final[float] = 0.80   # 80%

    def calculate_sample_size(self,
                              baseline_conversion: float,
                              minimum_detectable_effect: float) -> int:
        """
        Calcula o tamanho de amostra NECESSÁRIO POR GRUPO.
        
        Args:
            baseline_conversion: taxa de conversão atual (0.10 = 10%)
            minimum_detectable_effect: diferença mínima detectável (0.02 = +2pp)
        """

        # Efeito padronizado (aprox). Em produção, use Cohen’s h.
        effect_size = minimum_detectable_effect / np.sqrt(
            baseline_conversion * (1 - baseline_conversion)
        )

        n_obs = zt_ind_solve_power(
            effect_size=effect_size,
            alpha=self.DEFAULT_ALPHA,
            power=self.DEFAULT_POWER,
            ratio=1.0,
            alternative="two-sided"
        )

        return int(np.ceil(n_obs))


if __name__ == "__main__":
    designer = ExperimentDesign()

    needed = designer.calculate_sample_size(
        baseline_conversion=0.10,       # 10%
        minimum_detectable_effect=0.02  # +2pp
    )

    print("--- Planejamento de Teste A/B ---")
    print("Taxa Atual: 10%")
    print("Alvo: 12% (+2pp)")
    print(f"Usuários necessários por grupo: {needed}")
    print(f"Total necessário: {needed * 2}")
    print()
    print("Lição Sênior:")
    print("Se você recebe apenas ~100 usuários por dia, "
          "o experimento levaria ~", (needed * 2) / 100, "dias.")
