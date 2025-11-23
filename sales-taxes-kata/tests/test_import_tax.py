from decimal import Decimal
from domain import Product
from concrete import ImportDutyTax

def test_import_tax_applied_when_imported():
    tax = ImportDutyTax().calculate_tax(
        Product("Chocolate", Decimal("10.00"), True, False)
    )
    assert tax == Decimal("0.50")


def test_import_tax_zero_when_not_imported():
    tax = ImportDutyTax().calculate_tax(
        Product("Chocolate Nacional", Decimal("10.00"), False, False)
    )
    assert tax == Decimal("0.00")
