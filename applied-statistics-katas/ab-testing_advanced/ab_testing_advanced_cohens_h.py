import numpy as np
from statsmodels.stats.power import zt_ind_solve_power
from abc import ABC, abstractmethod
from typing import Final, List, Tuple

# 1. ABSTRAÇÃO (Strategy Pattern)
class ExperimentStrategy(ABC):
    """Interface para estratégias de cálculo de tamanho de amostra."""
    
    @abstractmethod
    def calculate_sample_size(self, baseline: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
        pass

# 2. IMPLEMENTAÇÃO FREQUENTISTA (Com Cohen's h)
class FrequentistStrategy(ExperimentStrategy):
    """
    Cálculo clássico usando Teste Z e Cohen's h para padronização do efeito.
    """
    
    def _calculate_cohens_h(self, p1: float, p2: float) -> float:
        """
        Calcula a distância padronizada entre duas proporções.
        h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
        """
        return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    
    def calculate_sample_size(self, baseline: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
        target_rate = baseline + mde
        
        # Cálculo rigoroso do Effect Size
        effect_size = self._calculate_cohens_h(target_rate, baseline)
        
        # Resolve para N
        n_obs = zt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative='two-sided'
        )
        return int(np.ceil(n_obs))

# 3. MÓDULO DE VIABILIDADE TEMPORAL
class FeasibilityAnalyzer:
    """Calcula se o experimento é viável no tempo."""
    
    @staticmethod
    def estimate_duration(sample_size_per_variant: int, daily_traffic: int, n_variants: int = 2) -> float:
        total_sample = sample_size_per_variant * n_variants
        days = total_sample / daily_traffic
        return days


if __name__ == "__main__":
    # Teste rápido manual
    strategy = FrequentistStrategy()
    n = strategy.calculate_sample_size(0.10, 0.02) # 10% -> 12%
    
    analyzer = FeasibilityAnalyzer()
    days = analyzer.estimate_duration(n, daily_traffic=500)
    
    print(f"Frequentista (Cohen's h): {n} usuários/grupo")
    print(f"Duração estimada (500 visits/dia): {days:.1f} dias")
