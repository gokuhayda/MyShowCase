from typing import Protocol

# 1. DEFINIÇÃO DE CONTRATO (Protocol é o ABC moderno do Python)
class ShippingStrategy(Protocol):
    def calculate(self, weight: float) -> float:
        pass
