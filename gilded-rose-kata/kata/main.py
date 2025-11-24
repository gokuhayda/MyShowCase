from orchestrator import GildedRose
from domain import Item


def _self_test() -> None:
    """
    Executa validações rápidas para assegurar que a refatoração
    mantém o comportamento esperado dos itens principais.

    Este módulo funciona como um smoke test local, não substituindo
    a suíte de testes formal.
    """

    # Normal Item
    item = Item("Normal", 10, 20)
    GildedRose([item]).update_quality()
    assert item.quality == 19, "Normal item: quality deveria reduzir para 19"

    # Aged Brie
    item = Item("Aged Brie", 2, 0)
    GildedRose([item]).update_quality()
    assert item.quality == 1, "Aged Brie: quality deveria aumentar para 1"

    # Conjured
    item = Item("Conjured Mana Cake", 3, 6)
    GildedRose([item]).update_quality()
    assert item.quality == 4, "Conjured: quality deveria reduzir para 4"

    print("✔ Todas as validações rápidas passaram.")


if __name__ == "__main__":
    print("--- Execução de validação rápida ---")
    _self_test()
