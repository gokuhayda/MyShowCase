
from factory import StrategyFactory
# ==========================================
# 5. ORCHESTRATOR (O Orquestrador)
# Orchestrator (Orquestrador): É o chefe. Ele não põe a 
# mão na massa. Ele pega a lista de itens, pede 
# para os operários para a Fábrica e manda executarem o trabalho.
# ==========================================


class GildedRose:
    """
    Orquestrador principal do sistema.
    
    Responsabilidade:
        - Receber a lista de itens.
        - Obter a estratégia correta via StrategyFactory.
        - Delegar a atualização de cada item conforme sua regra específica.

    Observação:
        Esta classe não contém lógica de negócio. Toda a regra está nas
        estratégias concretas, garantindo aderência ao OCP e DIP.
    """

    def __init__(self, items):
        self.items = items

    def update_quality(self) -> None:
        """Atualiza qualidade e sell_in de todos os itens usando polimorfismo."""
        for item in self.items:
            strategy = StrategyFactory.create_strategy(item)
            strategy.update(item)
