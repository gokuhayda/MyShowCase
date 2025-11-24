from typing import Final

# 2. IMPLEMENTAÇÕES CONCRETAS
class CorreiosStrategy:
    """Correios: Mais barato, taxa fixa base."""
    RATE: Final[float] = 2.5
    FIXED_FEE: Final[float] = 10.0
    
    def calculate(self, weight: float) -> float:
        return (weight * self.RATE) + self.FIXED_FEE

class FedExStrategy:
    """FedEx: Mais rápido, sem taxa fixa, multiplicador maior."""
    RATE: Final[float] = 3.0
    
    def calculate(self, weight: float) -> float:
        return weight * self.RATE

class DHLStrategy:
    """DHL: Premium, taxa fixa alta."""
    RATE: Final[float] = 4.0
    FIXED_FEE: Final[float] = 20.0
    
    def calculate(self, weight: float) -> float:
        return (weight * self.RATE) + self.FIXED_FEE
