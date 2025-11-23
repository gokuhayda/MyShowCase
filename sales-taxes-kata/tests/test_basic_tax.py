from decimal import Decimal
from domain import Product
from concrete import BasicSalexTax

def test_basic_tax_applied_when_not_exempt():
    tax = BasicSalexTax().calculate_tax(
        Product("Perfume", Decimal("10.00"), False, False)
    )
    assert tax == Decimal("1.00")


def test_basic_tax_zero_when_exempt():
    tax = BasicSalexTax().calculate_tax(
        Product("Book", Decimal("10.00"), False, True)
    )
    assert tax == Decimal("0.00")
