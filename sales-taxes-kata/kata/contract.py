"""
Contract Layer
==============
Define a abstração usada pelas estratégias de imposto.

Este módulo aplica DIP (Dependency Inversion Principle):
--------------------------------------------------------
• O sistema depende da ABSTRAÇÃO, não das implementações concretas.
• Novas taxas podem ser criadas sem modificar código existente.
"""

from decimal import Decimal
from abc import ABC, abstractmethod
from domain import Product


class TaxStrategy(ABC):
    """
    Interface (contrato) para cálculo de impostos.

    Princípio aplicado:
    -------------------
    • ISP — Interface Segregation Principle:
      Uma interface pequena, coesa e que não força métodos desnecessários.

    • LSP — Liskov Substitution Principle:
      Toda subclasse deve implementar `calculate_tax` com a mesma assinatura.
    """

    @abstractmethod
    def calculate_tax(self, product: Product) -> Decimal:
        """
        Calcula o valor do imposto para um produto.

        Deve ser:
        • Determinístico
        • Livre de side effects
        • Compatível com o contrato da classe base (LSP)

        Parameters
        ----------
        product : Product
            Produto alvo do cálculo.

        Returns
        -------
        Decimal
            Valor do imposto correspondente.
        """
        pass
