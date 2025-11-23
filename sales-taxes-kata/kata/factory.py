"""
Factory Layer
=============
Objetivo:
---------
Centralizar a criação e ativação das estratégias de imposto.

Por quê?
--------
• Isola a "main" da lógica de quais impostos estão ativos.
• Facilita troca por região, país, configuração, feature flags, etc.
• Segue SRP e OCP simultaneamente.
"""

from typing import List
from concrete import BasicSalexTax, ImportDutyTax
from contract import TaxStrategy


class TaxConfigurationFactory:
    """
    Factory que monta a composição correta de estratégias.

    Novas regiões podem ser adicionadas sem quebrar nada:
    basta retornar uma nova lista de estratégias.
    """

    @staticmethod
    def get_active_strategies(region: str = 'DEFAULT') -> List[TaxStrategy]:
        """
        Retorna a lista de estratégias de imposto para a região.

        Poderia evoluir para ler:
        • .env
        • Banco de dados
        • Flags de configuração
        • Service Discovery
        """
        if region == 'DEFAULT':
            return [BasicSalexTax(), ImportDutyTax()]

        elif region == 'BR':
            # Exemplo: adicionar ICMS, ISS, etc.
            return []

        return []
