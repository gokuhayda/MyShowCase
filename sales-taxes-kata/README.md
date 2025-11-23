
# 🧠 Sales Taxes Kata — A Engineering Exercise

Este repositório apresenta uma implementação profissional do *Sales Taxes Problem*, estruturada segundo práticas amplamente utilizadas em consultorias de elite.  
O objetivo é demonstrar engenharia de software sênior, arquitetura clara e comunicação técnica de alto nível.

---

# 🏷️ Badges

![Python Version](https://img.shields.io/badge/python-3.11+-blue)
![Tests](https://img.shields.io/badge/tests-passing-green)
![Design](https://img.shields.io/badge/architecture-FunctionalCore%2FImperativeShell-purple)
![Pattern](https://img.shields.io/badge/pattern-Strategy-orange)
![SOLID](https://img.shields.io/badge/SOLID-Compliant-brightgreen)

---

# 🌐 Arquitetura — Functional Core, Imperative Shell & Strategy-Driven Design

A arquitetura segue um padrão muito valorizado pela Thoughtworks:

## **Functional Core (puro)**
- Regras de negócio 100% determinísticas  
- Zero side effects  
- Fácil de testar e refatorar  
- Modelos e cálculos puros

## **Imperative Shell (impuro)**
- Entrada/saída  
- Arredondamentos  
- Configuração (Factory Pattern)  
- Composição do sistema  

Essa separação reduz acoplamento, aumenta previsibilidade e facilita pairing.

---

# 📁 Estrutura do Projeto

```
📁 sales-taxes-kata/
 ┣ 🐍 main.py                    → Entry point (Imperative Shell)
 ┣ 📦 kata/                      → Functional Core + business logic
 ┃   ┣ domain.py                 → Product entity
 ┃   ┣ contract.py               → TaxStrategy (ISP + DIP)
 ┃   ┣ concrete.py               → Concrete strategies (Strategy Pattern)
 ┃   ┣ orchestrator.py           → TaxCalculator (LSP + OCP)
 ┃   ┗ factory.py                → Composition root / wiring
 ┗ 🧪 tests/                     → Unit tests (TDD)
 ┗ 🧪 docs/                      → Diagramas UML
```

---

# 🧩 Padrões e Princípios Demonstrados

## ✔ Strategy Pattern
Cada imposto é isolado como uma estratégia independente.

- O orquestrador **não conhece** as classes concretas  
- Extensões não quebram código existente  
- Polimorfismo puro (LSP)

## ✔ SOLID aplicado

### **S — SRP**  
Cada classe tem uma única razão para mudar.

### **O — OCP**  
Novos impostos?  
Basta criar uma nova estratégia — sem tocar no `TaxCalculator`.

### **L — LSP**  
Todas as estratégias podem ser substituídas sem quebrar o orquestrador.

### **I — ISP**  
Interface pequena, clara e específica.

### **D — DIP**  
O orquestrador depende de abstrações, não implementações.

---

# 🔎 Exemplo de Uso

```python
from kata.factory import TaxConfigurationFactory
from kata.orchestrator import TaxCalculator
from kata.domain import Product
from decimal import Decimal

strategies = TaxConfigurationFactory.get_active_strategies()

calculator = TaxCalculator(strategies)

product = Product(
    name="Perfume Importado",
    price=Decimal("47.50"),
    is_imported=True,
    is_exempt=False
)

tax = calculator.get_total_tax(product)

print(f"Total tax: {tax}")
```

### Saída esperada:
```
Total tax: 7.15
```

---

# 🎯 Regra de Arredondamento (Estilo Thoughtworks)

Sempre arredonde **para cima** até o múltiplo de 0.05 mais próximo.

| Valor | Arredondado |
|-------|-------------|
| 41.71 | 41.75 |
| 41.76 | 41.80 |
| 0.01  | 0.05  |

Implementado em `TaxCalculator._round_tax`.

---

# 🧪 Testes (TDD)

Exemplo:

```python
def test_basic_tax_non_exempt():
    p = Product("Book", Decimal("10.00"), False, False)
    strategies = [BasicSalexTax()]
    tax = TaxCalculator(strategies).get_total_tax(p)
    assert tax == Decimal("1.00")
```

Testes são:

- pequenos  
- determinísticos  
- independentes  
- fáceis de ler  
- guiados por comportamento  

---

# 🏆 Por que esta solução combina com a cultura da elite DS?

Este kata demonstra:

- separação intencional de responsabilidades  
- código orientado a princípios, não a atalhos  
- pureza do domínio + orquestração explícita  
- uso forte de abstrações  
- testabilidade e clareza arquitetural  
- decisões explicáveis em pairing  

Esse é exatamente o tipo de raciocínio que a TW avalia em entrevistas.

---

# 📜 Licença

MIT License.
