from contract import UpdateStrategy
from domain import Item
from concrete import (
    BackStagePassStrategy,
    AgedBrieStrategy,
    NormalItemStrategy,
    SulfurasStrategy,
    ConjuredItemStrategy,
)


class StrategyFactory:
    """
    Factory responsável por selecionar a estratégia correta para atualização
    de um item conforme sua categoria.

    Essa classe centraliza a decisão de qual `UpdateStrategy` deve ser usada
    para cada tipo de item, evitando condicionais espalhadas no código principal.
    """

    @staticmethod
    def create_strategy(item: Item) -> UpdateStrategy:
        """
        Retorna a estratégia apropriada para o tipo do item informado.

        Parâmetros
        ----------
        item : Item
            Instância contendo `name`, `sell_in` e `quality`.

        Retorna
        -------
        UpdateStrategy
            Implementação concreta da estratégia correspondente ao item.
        """
        match item.name:
            case "Aged Brie":
                return AgedBrieStrategy()
            case "Backstage passes to a TAFKAL80ETC concert":
                return BackStagePassStrategy()
            case "Sulfuras, Hand of Ragnaros":
                return SulfurasStrategy()
            case "Conjured Mana Cake":
                return ConjuredItemStrategy()
            case _:
                return NormalItemStrategy()
