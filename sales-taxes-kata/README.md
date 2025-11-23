# 🧠 Sales Taxes Kata — A Thoughtworks‑Style Engineering Exercise

Este repositório apresenta uma implementação profissional do *Sales Taxes Problem*, estruturada segundo práticas amplamente utilizadas em consultorias de elite como a **Thoughtworks**.  
O foco aqui não é apenas “fazer funcionar”, mas demonstrar:

- Engenharia de software rigorosa  
- Arquitetura pensada  
- Isolamento de efeitos colaterais  
- Abordagem funcional + orientação a objetos  
- Extensibilidade real (OCP)  
- Polimorfismo seguro (LSP)  
- Clareza de comunicação e verbalização técnica  
- Testabilidade (TDD friendly)

Se você está se preparando para entrevistas sênior — especialmente TW — este repositório demonstra exatamente o tipo de raciocínio, design e clareza técnica esperado.

---

# 🌐 Arquitetura Geral — Functional Core / Imperative Shell

A arquitetura adota o padrão defendido historicamente pela Thoughtworks:

**Functional Core (puro):**  
- Regras de negócio determinísticas  
- Zero side-effects  
- Testes simples e estáveis  
- Facilidade para refatoração  

**Imperative Shell (impuro):**  
- Entrada/saída  
- Arredondamentos  
- Configuração (factory)  
- Composição do sistema  

### Motivação arquitetural
A TW valoriza muito *separação de responsabilidades, testabilidade e clareza cognitiva*.  
A divisão clara entre *pureza* e *efeitos colaterais* ajuda a criar sistemas mais previsíveis, fáceis de evoluir e resilientes a mudanças — exatamente o perfil de design avaliado em pair programming.

---

# 🧱 Estrutura do Projeto

```
📁 project/
 ┣ domain.py            → Entidade Product (imutável, funcional)
 ┣ contract.py          → Abstração TaxStrategy (DIP + ISP)
 ┣ concrete.py          → Estratégias concretas (Strategy Pattern)
 ┣ orchestrator.py      → TaxCalculator (polimorfismo + LSP)
 ┣ factory.py           → Composição e ativação das estratégias
 ┗ tests/               → Testes unitários (TDD)
```

---

# 🧩 Padrões e Princípios Demonstrados

## ✔ Strategy Pattern
Cada regra de imposto é encapsulada em uma “estratégia”.  
O orquestrador **não sabe** que tipo de taxa está sendo aplicada.

### Por que Thoughtworks gosta disso?
- Remove condicionais (`if`, `elif`) difíceis de manter  
- Permite evolução independente  
- Reduz acoplamento entre política e mecanismo  

---

## ✔ SOLID aplicado de forma explícita

### **S — SRP**
Cada módulo tem uma única razão para mudar.

### **O — OCP**
Novas taxas?  
Crie uma classe.  
Não toque no orquestrador.

### **L — LSP**
O `TaxCalculator` confia que todas as estratégias respeitam o contrato.

### **I — ISP**
A interface é pequena, intencional e limpa.

### **D — DIP**
Orquestrador depende da *abstração*, não das implementações.

Este kata é praticamente um showcase perfeito de SOLID aplicado em código real.

---

# 🔎 Código de Exemplo — Uso Completo

```python
from factory import TaxConfigurationFactory
from orchestrator import TaxCalculator
from domain import Product
from decimal import Decimal

# Estratégias ativas de imposto (DEFAULT = SalesTax + ImportDuty)
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

# 🎯 Regras de Arredondamento — Estilo Thoughtworks

```
Sempre arredonde PARA CIMA para o múltiplo de 0.05 mais próximo.
```

Exemplos:

| Valor | Arredondado |
|-------|-------------|
| 41.71 | 41.75 |
| 41.76 | 41.80 |
| 0.01  | 0.05  |

Regra implementada em `orchestrator.py`.

---

# 🧪 Testes (TDD)

Exemplo de teste para imposto básico:

```python
def test_basic_tax_non_exempt():
    p = Product("Book", Decimal("10.00"), is_imported=False, is_exempt=False)
    strategies = [BasicSalexTax()]
    tax = TaxCalculator(strategies).get_total_tax(p)
    assert tax == Decimal("1.00")
```

Exemplo para importados:

```python
def test_import_duty():
    p = Product("Chocolate Importado", Decimal("10.00"), True, True)
    strategies = [ImportDutyTax()]
    tax = TaxCalculator(strategies).get_total_tax(p)
    assert tax == Decimal("0.50")
```

### Por que TDD combina perfeitamente aqui?
- Classes puras → baixa complexidade cognitiva  
- Funções determinísticas → testes confiáveis  
- Princípios SOLID → testes independentes  

---

# 🧠 Senioridade: O Que Este Kata Demonstra

✔ entendimento profundo de abstrações  
✔ uso intencional de padrões  
✔ testabilidade pensada desde o início  
✔ domínio de princípios de design  
✔ clareza arquitetural (core vs shell)  
✔ comunicação e nomeação profissional  
✔ código extensível e sustentável  
✔ orientação à prática TW (verbalização, trade-offs, decisões arquiteturais)  

Este repositório mostra não apenas como você codifica —  
mas **como você pensa software**.

---

# 📜 Licença
MIT License.

---

Se quiser integrar:

- 🌐 versão em inglês  
- 📈 Github Actions (CI)  
- 📊 cobertura de testes  
- 🎨 banner visual “Sales Taxes Kata — Thoughtworks Edition”  

Posso gerar tudo automaticamente.  
Só pedir!

