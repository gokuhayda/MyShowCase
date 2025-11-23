from factory import TaxConfigurationFactory
from concrete import BasicSalexTax, ImportDutyTax

def test_default_region_strategies():
    strategies = TaxConfigurationFactory.get_active_strategies("DEFAULT")
    assert any(isinstance(s, BasicSalexTax) for s in strategies)
    assert any(isinstance(s, ImportDutyTax) for s in strategies)
