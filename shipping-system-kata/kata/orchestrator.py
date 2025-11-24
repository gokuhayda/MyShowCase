from dataclasses import dataclass
from contract import ShippingStrategy


@dataclass
class ShippingService:
    """
    O Serviço não sabe qual transportadora está usando.
    Ele apenas recebe uma 'strategy' no construtor.
    """
    strategy: ShippingStrategy
    
    def get_shipping_cost(self, weight: float) -> float:
        # Aqui poderia ter logs, validações, regras de negócio extras
        print(f"Calculando frete para {weight}kg...")
        return self.strategy.calculate(weight)
