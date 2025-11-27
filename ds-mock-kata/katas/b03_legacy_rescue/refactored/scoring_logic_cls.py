from typing import Final

class ScoringLogic:
    """
    Lógica PURA. Zero I/O. Zero Mocks necessários para testar isto.
    Apenas matemática e regras de negócio.
    """
    
    # Definição das constantes (Business Rules)
    AGE_WEIGHT: Final[float] = 0.10
    INCOME_WEIGHT: Final[float] = 0.001  # Multiplicar por 0.001 é o mesmo que dividir por 1000
    HISTORY_WEIGHT: Final[int] = 5

    @classmethod
    def calculate_base_score(cls, age: int, income: float, history: int) -> float:
        """
        Calcula o score base usando os pesos definidos na classe.
        """
        # Nota: income * 0.001 é matematicamente igual a income / 1000
        score = (age * cls.AGE_WEIGHT) + \
                (income * cls.INCOME_WEIGHT) + \
                (history * cls.HISTORY_WEIGHT)
        
        return score

    @staticmethod
    def calculate_final_score(base_score: float, ml_probability: float) -> float:
        return base_score * ml_probability
