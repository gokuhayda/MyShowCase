"""
Concrete Strategies
===================
Cada classe aqui representa uma regra de imposto independente.

Padrão utilizado:
-----------------
• Strategy Pattern: comportamento variável encapsulado em classes distintas.
• OCP — Open/Closed Principle: adicione novas taxas sem modificar código existente.

Todas são funções puras: dado um produto, retornam o imposto.
"""

from contract import TaxStrategy
from domain import Product
from decimal import Decimal
from typing import Final


class BasicSalexTax(TaxStrategy):
    """
    Estratégia para o imposto básico (10%).

    Notas:
    ------
    • Não se aplica a produtos isentos.
    • RATE é uma constante imutável (Final).
    """

    RATE: Final[Decimal] = Decimal('0.10')

    def calculate_tax(self, product: Product) -> Decimal:
        if product.is_exempt:
            return Decimal('0.00')
        return product.price * self.RATE


class ImportDutyTax(TaxStrategy):
    """
    Estratégia para taxa de importação (5%).

    • Aplica-se apenas a produtos importados.
    """

    RATE: Final[Decimal] = Decimal('0.05')

    def calculate_tax(self, product: Product) -> Decimal:
        if not product.is_imported:
            return Decimal('0.00')
        return product.price * self.RATE
