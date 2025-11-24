from contract import UpdateStrategy
from domain import Item

# ==========================================
# 3. CONCRETE (As Implementações / Regras)
# Concrete (Concreto): São os operários especializados. 
# Cada um sabe fazer o trabalho de um jeito (um cuida do Queijo,
# outro do Ingresso).
# ==========================================

class NormalItemStrategy(UpdateStrategy):
    """
    Estratégia padrão para itens comuns.
    Regras:
    - Perde 1 de qualidade por dia.
    - Após o vencimento (sell_in < 0), perde 2 por dia.
    - Qualidade nunca é negativa.
    """
    def update(self, item: Item):
        self._decrease_quality(item)
        item.sell_in -= 1

        if item.sell_in < 0:
            self._decrease_quality(item)

    def _decrease_quality(self, item: Item):
        if item.quality > 0:
            item.quality -= 1


class AgedBrieStrategy(UpdateStrategy):
    """
    Estratégia para 'Aged Brie'.
    Regras:
    - Ganha 1 de qualidade por dia.
    - Após o vencimento, ganha 2 por dia.
    - Qualidade máxima é 50.
    """
    def update(self, item: Item):
        self._increase_quality(item)
        item.sell_in -= 1

        if item.sell_in < 0:
            self._increase_quality(item)

    def _increase_quality(self, item: Item):
        if item.quality < 50:
            item.quality += 1


class BackstagePassStrategy(UpdateStrategy):
    """
    Estratégia para 'Backstage Passes'.
    Regras:
    - +1 de qualidade normalmente.
    - +2 se faltam < 10 dias.
    - +3 se faltam < 5 dias.
    - Após o show (sell_in < 0), qualidade zera.
    - Qualidade máxima é 50.
    """
    def update(self, item: Item):
        if item.quality < 50:
            item.quality += 1

            if item.sell_in < 10 and item.quality < 50:
                item.quality += 1

            if item.sell_in < 5 and item.quality < 50:
                item.quality += 1

        item.sell_in -= 1

        if item.sell_in < 0:
            item.quality = 0


class ConjuredItemStrategy(UpdateStrategy):
    """
    Estratégia para itens 'Conjured'.
    Regras:
    - Degradam 2x mais rápido que itens normais.
    - Após o vencimento, degradam 4x mais rápido.
    """
    def update(self, item: Item):
        self._decrease_quality(item)  # 1x
        self._decrease_quality(item)  # 2x
        item.sell_in -= 1

        if item.sell_in < 0:
            self._decrease_quality(item)  # 3x
            self._decrease_quality(item)  # 4x

    def _decrease_quality(self, item: Item):
        if item.quality > 0:
            item.quality -= 1


class SulfurasStrategy(UpdateStrategy):
    """
    Estratégia para itens lendários ('Sulfuras').
    Regras:
    - Qualidade não muda.
    - sell_in não muda.
    - Item é imutável.
    """
    def update(self, item: Item):
        pass  # Item lendário → não sofre alterações
