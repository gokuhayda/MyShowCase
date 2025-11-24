import pytest
from domain import Item
from orchestrator import GildedRose


# ============================================================
# Helper para rodar a atualização em 1 dia
# ============================================================
def update_once(item):
    app = GildedRose([item])
    app.update_quality()


# ============================================================
# TESTES PARA ITENS NORMAIS
# ============================================================
def test_normal_item_quality_decreases():
    item = Item("Normal Item", 10, 20)
    update_once(item)
    assert item.quality == 19
    assert item.sell_in == 9


def test_normal_item_quality_degrades_twice_after_expiration():
    item = Item("Normal Item", 0, 20)
    update_once(item)
    assert item.quality == 18


def test_normal_item_quality_never_negative():
    item = Item("Normal Item", 5, 0)
    update_once(item)
    assert item.quality == 0


# ============================================================
# TESTES PARA AGED BRIE
# ============================================================
def test_aged_brie_increases_quality():
    item = Item("Aged Brie", 2, 0)
    update_once(item)
    assert item.quality == 1


def test_aged_brie_increases_twice_after_expiration():
    item = Item("Aged Brie", 0, 10)
    update_once(item)
    assert item.quality == 12


def test_aged_brie_max_quality_50():
    item = Item("Aged Brie", 5, 50)
    update_once(item)
    assert item.quality == 50


# ============================================================
# TESTES PARA BACKSTAGE PASS
# ============================================================
def test_backstage_increases_normal():
    item = Item("Backstage passes to a TAFKAL80ETC concert", 15, 20)
    update_once(item)
    assert item.quality == 21


def test_backstage_increases_by_2_when_10_days_left():
    item = Item("Backstage passes to a TAFKAL80ETC concert", 10, 20)
    update_once(item)
    assert item.quality == 22


def test_backstage_increases_by_3_when_5_days_left():
    item = Item("Backstage passes to a TAFKAL80ETC concert", 5, 20)
    update_once(item)
    assert item.quality == 23


def test_backstage_quality_drops_to_zero_after_concert():
    item = Item("Backstage passes to a TAFKAL80ETC concert", 0, 40)
    update_once(item)
    assert item.quality == 0


# ============================================================
# TESTES PARA SULFURAS
# ============================================================
def test_sulfuras_never_changes():
    item = Item("Sulfuras, Hand of Ragnaros", 0, 80)
    update_once(item)
    assert item.sell_in == 0
    assert item.quality == 80


# ============================================================
# TESTES PARA CONJURED
# ============================================================
def test_conjured_degrades_twice_as_fast_before_expiration():
    item = Item("Conjured Mana Cake", 3, 6)
    update_once(item)
    assert item.quality == 4


def test_conjured_degrades_four_after_expiration():
    item = Item("Conjured Mana Cake", 0, 10)
    update_once(item)
    assert item.quality == 6


# ============================================================
# TESTE DE INTEGRAÇÃO DA ORQUESTRAÇÃO
# ============================================================
def test_gilded_rose_processes_multiple_items():
    items = [
        Item("Normal Item", 5, 10),
        Item("Aged Brie", 2, 0),
        Item("Conjured Mana Cake", 3, 6),
    ]

    app = GildedRose(items)
    app.update_quality()

    assert items[0].quality == 9
    assert items[1].quality == 1
    assert items[2].quality == 4
