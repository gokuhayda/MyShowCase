import numpy as np
from scipy.stats import beta
from ab_testing_advanced_cohens_h import ExperimentStrategy

class BayesianSimulationStrategy(ExperimentStrategy):
    """
    Estratégia Bayesiana para estimar tamanho de amostra em Testes A/B.
    
    A pergunta aqui é:
        "Se a diferença for real, quantas observações preciso para ter X% de certeza 
        de que a Variante B é melhor do que A?"
    
    - Usamos priors Beta(1,1) (não informativos)
    - Simulação Monte Carlo dos posteriores
    - Cálculo empírico de P(B > A)
    """

    def calculate_sample_size(
        self, 
        baseline: float, 
        mde: float, 
        alpha: float = 0.05, 
        power: float = 0.8
    ) -> int:

        # Probabilidade-alvo: P(B > A)
        required_certainty = 1 - alpha  # ex: 95%
        target = baseline + mde         # taxa da variante B

        # Loop incremental de busca (simples e didático)
        for n in range(100, 10000, 100):

            # Conversões esperadas para cada variante
            conv_a = int(n * baseline)
            conv_b = int(n * target)

            # Posteriores Beta
            post_a = beta(1 + conv_a, 1 + n - conv_a)
            post_b = beta(1 + conv_b, 1 + n - conv_b)

            # Amostragem Monte Carlo
            samples = 2000
            samp_a = post_a.rvs(samples)
            samp_b = post_b.rvs(samples)

            # Estimativa empírica de P(B > A)
            prob_b_better = np.mean(samp_b > samp_a)

            # Critério de parada
            if prob_b_better >= required_certainty:
                return n

        # Caso extremo (não convergiu)
        return 10000
