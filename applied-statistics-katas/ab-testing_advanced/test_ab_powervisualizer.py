import pytest
import matplotlib.pyplot as plt
from ab_testing_advanced_cohens_h import ExperimentStrategy, FrequentistStrategy, FeasibilityAnalyzer


# --- VISUALIZAÇÃO ---
class PowerVisualizer:
    @staticmethod
    def plot_power_curve(strategy: ExperimentStrategy, baseline: float):
        """Gera gráfico: Tamanho da Amostra vs MDE (Minimum Detectable Effect)"""
        mde_values = [0.005, 0.01, 0.02, 0.05, 0.10] # 0.5% a 10%
        sample_sizes = []
        
        for mde in mde_values:
            n = strategy.calculate_sample_size(baseline, mde)
            sample_sizes.append(n)
            
        plt.figure(figsize=(10, 6))
        plt.plot(mde_values, sample_sizes, marker='o', linestyle='-', color='b')
        plt.title(f'Power Curve (Baseline: {baseline:.0%})')
        plt.xlabel('Minimum Detectable Effect (MDE)')
        plt.ylabel('Sample Size per Variant')
        plt.grid(True)
        plt.show()

# --- TESTES PARAMÉTRICOS (Pytest) ---
# Isso substitui escrever 10 funções de teste separadas!

@pytest.mark.parametrize("baseline, mde, expected_range", [
    (0.10, 0.02, (3500, 4000)),   # 10% -> 12% requer ~3.8k
    (0.50, 0.05, (1500, 1600)),   # 50% -> 55% requer ~1.5k
    (0.01, 0.005, (30000, 35000)) # 1% -> 1.5% requer MUITO tráfego
])
def test_frequentist_calculation(baseline, mde, expected_range):
    strategy = FrequentistStrategy()
    n = strategy.calculate_sample_size(baseline, mde)
    
    # Verifica se está dentro do range esperado (matemática é exata, mas ranges são seguros)
    assert expected_range[0] <= n <= expected_range[1]

@pytest.mark.parametrize("n, traffic, expected_days", [
    (1000, 100, 20.0), # (1000 * 2) / 100 = 20
    (5000, 1000, 10.0) # (5000 * 2) / 1000 = 10
])
def test_feasibility(n, traffic, expected_days):
    days = FeasibilityAnalyzer.estimate_duration(n, traffic)
    assert days == expected_days
