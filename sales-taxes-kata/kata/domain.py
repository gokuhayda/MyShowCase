"""
Domain Layer
============
Neste módulo ficam apenas entidades puras e imutáveis do domínio.
Aqui não existe comportamento com efeitos colaterais, garantindo
testes simples, determinismo e aderência ao conceito de Functional Core.
"""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Product:
    """
    Entidade imutável que representa um produto.

    Por que imutável?
    -----------------
    • Facilita testes (TDD)
    • Evita comportamento inesperado (side effects)
    • Segue boas práticas de Domain Modeling

    Attributes
    ----------
    name : str
        Nome do produto.
    price : Decimal
        Preço bruto do produto.
    is_imported : bool
        Indica se o item é importado (relevante para taxas).
    is_exempt : bool
        Indica se é isento do imposto básico.
    """
    name: str
    price: Decimal
    is_imported: bool
    is_exempt: bool
