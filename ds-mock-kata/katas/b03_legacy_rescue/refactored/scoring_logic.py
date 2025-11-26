class ScoringLogic:
    """
    Lógica PURA. Zero I/O. Zero Mocks necessários para testar isto.
    Apenas matemática e regras de negócio.
    """
    
    @staticmethod
    def calculate_base_score(age: int, income: float, history: int) -> float:
        # Regra de negócio isolada
        return (age * 0.1) + (income / 1000) + (history * 5)
    
    @staticmethod
    def calculate_final_score(base_score: float, ml_probability: float) -> float:
        return base_score * ml_probability
