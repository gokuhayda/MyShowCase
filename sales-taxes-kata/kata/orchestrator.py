"""
Tax Calculator (Orquestrador)
==============================
Aplica uma lista de estratégias (Strategy Pattern) para calcular o
imposto total de forma extensível e polimórfica.

Princípios aplicados:
---------------------
• OCP: novas taxas entram sem alterar esta classe.
• DIP: depende da abstração TaxStrategy.
• LSP: todas as estratégias podem ser substituídas entre si.
• Functional Core: cálculo puro.
• Imperative Shell: arredondamento e composição externa.
"""

from typing import List
from decimal import Decimal, ROUND_UP
from domain import Product
from contract import TaxStrategy


class TaxCalculator:
    """
    Orquestrador responsável pelo cálculo total de imposto.

    Não conhece detalhes das estratégias. Apenas itera e aplica.
    """

    def __init__(self, strategies: List[TaxStrategy]):
        """
        Injeta as estratégias ativas de imposto.

        Dependency Injection + DIP.
        """
        self.strategies = strategies

    def get_total_tax(self, product: Product) -> Decimal:
        """
        Soma todos os impostos aplicáveis ao produto.

        Polimorfismo:
        -------------
        Cada elemento da lista pode ser qualquer subclasse de TaxStrategy.
        Isso é LSP em ação: múltiplas formas, uma mesma chamada.
        """
        total = Decimal('0.00')

        for strategy in self.strategies:
            total += strategy.calculate_tax(product)

        return self._round_tax(total)

    @staticmethod
    def _round_tax(amount: Decimal) -> Decimal:
        """
        Arredondamento estilo Thoughtworks:
        • Sempre para cima
        • Para múltiplos de 0.05

        Ex:
        • 41.71 → 41.75
        • 41.76 → 41.80
        """
        if amount == 0:
            return Decimal('0.00')

        return (
            (amount / Decimal('0.05'))
            .quantize(Decimal('1'), rounding=ROUND_UP)
            * Decimal('0.05')
        )
