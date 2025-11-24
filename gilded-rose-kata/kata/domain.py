from dataclasses import dataclass

# ============================================================
# 1. DOMAIN (Dado Puro)
# Representa a entidade de negócio manipulada pelo sistema.
# O Goblin (dono da loja) não permite lógica aqui — apenas estado.
# ============================================================

@dataclass
class Item:
    """
    Entidade de domínio que representa um item da loja.

    Atributos:
        name (str): Nome do item.
        sell_in (int): Dias restantes para venda. Reduz 1 por dia.
        quality (int): Qualidade/valor do item. Geralmente entre 0 e 50.
    """
    name: str
    sell_in: int
    quality: int

    def __repr__(self):
        return f"{self.name}, {self.sell_in}, {self.quality}"
