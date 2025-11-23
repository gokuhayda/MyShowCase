from decimal import Decimal
from factory import TaxConfigurationFactory
from orchestrator import TaxCalculator
from domain import Product

def test_full_integration_default_region():
    product = Product("Perfume Importado", Decimal("47.50"), True, False)
    strategies = TaxConfigurationFactory.get_active_strategies("DEFAULT")

    tax = TaxCalculator(strategies).get_total_tax(product)

    # Basic (10% = 4.75), Import (5% = 2.375 → 2.38)
    # Total bruto ≈ 7.125 → arredonda para 7.15
    assert tax == Decimal("7.15")
