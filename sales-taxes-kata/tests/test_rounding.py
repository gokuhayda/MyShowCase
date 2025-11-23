from decimal import Decimal
from orchestrator import TaxCalculator

def test_rounding_up_to_nearest_005():
    assert TaxCalculator._round_tax(Decimal("41.71")) == Decimal("41.75")
    assert TaxCalculator._round_tax(Decimal("41.76")) == Decimal("41.80")
    assert TaxCalculator._round_tax(Decimal("0.01")) == Decimal("0.05")
