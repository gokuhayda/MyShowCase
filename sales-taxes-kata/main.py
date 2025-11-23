"""
Main Module
===========
Ponto de entrada do sistema.

Objetivo deste arquivo:
-----------------------
• Demonstrar composição explícita do sistema (Imperative Shell)
• Mostrar como o Factory Pattern ativa estratégias de imposto
• Exibir utilização do TaxCalculator (Functional Core)
• Servir como exemplo pedagógico de integração entre camadas

Este arquivo NÃO contém lógica de negócio.
A lógica está isolada no Functional Core (domain, concrete, orchestrator).
Aqui nós apenas:

1. Orquestramos dependências
2. Instanciamos o produto
3. Chamamos o orquestrador
4. Exibimos resultados

Esse padrão é altamente valorizado em entrevistas TW:
• o domínio é puro
• as regras são extensíveis
• o main é apenas um cliente do sistema
"""

from decimal import Decimal
from kata.domain import Product
from kata.factory import TaxConfigurationFactory
from kata.orchestrator import TaxCalculator


if __name__ == '__main__':

    # 1. Obtemos as estratégias ativas via Factory Pattern
    #    A factory sabe quais impostos estão habilitados para a região.
    #    Através dela, seguimos OCP: trocar políticas sem reescrever o core.
    strategies = TaxConfigurationFactory.get_active_strategies('DEFAULT')

    # 2. O TaxCalculator recebe as estratégias (Dependency Injection)
    calculator = TaxCalculator(strategies)

    # 3. Criamos um produto do domínio (imutável = Functional Core)
    product = Product(
        name='book',
        price=Decimal('12.49'),
        is_imported=True,
        is_exempt=False
    )

    # 4. Cálculo do imposto total (polimorfismo + Strategy Pattern)
    tax = calculator.get_total_tax(product)

    # 5. Saída para observação
    print(f"--- Factory Pattern ---")
    print(f"Estratégias Ativas: {[type(s).__name__ for s in calculator.strategies]}")
    print(f"Imposto Calculado: {tax}")
