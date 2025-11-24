from scipy.stats import beta
from ab_testing_advanced_cohens_h import ExperimentStrategy

class BayesianSimulationStrategy(ExperimentStrategy):
    """
    Estima tamanho de amostra via Simulação de Monte Carlo.
    Pergunta: 'Quantas amostras preciso para que, se a diferença for real,
    eu tenha X% de certeza que B > A?'
    """
    
    def calculate_sample_size(self, baseline: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
        # Simplificação para o Kata: Bayesiano geralmente precisa de menos dados,
        # mas computacionalmente é mais pesado simular.
        # Aqui usamos uma heurística ou simulação rápida.
        
        target = baseline + mde
        required_certainty = 1 - alpha # ex: 95%
        
        # Simulação iterativa (simplificada para didática)
        for n in range(100, 10000, 100):
            # Simulamos conversões esperadas
            conv_a = int(n * baseline)
            conv_b = int(n * target)
            
            # Posterior Beta(alpha+conv, beta+n-conv)
            # Priors fracos (1,1)
            posterior_a = beta(1 + conv_a, 1 + n - conv_a)
            posterior_b = beta(1 + conv_b, 1 + n - conv_b)
            
            # P(B > A) aproximado via amostragem
            samples = 2000
            samp_a = posterior_a.rvs(samples)
            samp_b = posterior_b.rvs(samples)
            prob_b_better = np.mean(samp_b > samp_a)
            
            if prob_b_better >= required_certainty:
                return n
                
        return 10000 # Max cap
