# 📘 Glossário 

Este glossário ajuda a traduzir conceitos de Python moderno e Design Patterns para stakeholders, explicando termos técnicos e mostrando exemplos práticos.

---

## 1. Protocol (Interface)

| Termo Técnico | Explicação  | Exemplo |
|---------------|----------------------------|---------|
| **Protocol (Duck Typing Estático)** | Um contrato baseado em comportamento. Se a classe **tem o método esperado**, ela é aceita. Não precisa herdar explicitamente de uma classe base. | ```python\nfrom typing import Protocol\nclass ShippingStrategy(Protocol):\n    def calculate(self, weight: float) -> float: ...\n``` |
| **Valor de Negócio** | Flexibilidade máxima: qualquer transportadora que tenha `calculate` pode ser usada. Facilita testes, mocks e injeção de dependência. | Simplesmente criar uma nova classe com `calculate` já funciona no sistema, sem alterar o serviço. |

**Analogia:**  
> “Se anda como um pato e grasna como um pato, então é um pato.”  

---

## 2. ABC (Abstract Base Class)

| Termo Técnico | Explicação  | Exemplo |
|---------------|----------------------------|---------|
| **ABC (Abstract Base Class)** | Contrato formal que força herança. Classes concretas devem implementar todos os métodos abstratos. | ```python\nfrom abc import ABC, abstractmethod\nclass TaxStrategy(ABC):\n    @abstractmethod\n    def calculate(self, price: float) -> float: ...\n``` |
| **Valor de Negócio** | Garantia de conformidade em tempo de execução, mas menos flexível para mudanças rápidas ou testes. | Mais verboso, exige herança explícita. |

**Analogia:**  
> Contrato formal: “Se você quer ser uma TaxStrategy, precisa assinar e cumprir todas as cláusulas.”

---

## 3. Strategies (Implementações Concretas)

| Termo Técnico | Explicação  | Exemplo |
|---------------|----------------------------|---------|
| **Shipping Strategies** | Classes que implementam o cálculo de frete específico para cada transportadora. | ```python\nclass CorreiosStrategy:\n    RATE = 2.5\n    FIXED_FEE = 10.0\n    def calculate(self, weight: float) -> float:\n        return (weight * self.RATE) + self.FIXED_FEE\n``` |
| **Valor de Negócio** | Cada transportadora tem regras próprias, mas o serviço que calcula frete não precisa saber detalhes. Facilita troca de fornecedores ou regras. | Podemos mudar de Correios para FedEx sem alterar o ShippingService. |

---

## 4. Service Layer

| Termo Técnico | Explicação  | Exemplo |
|---------------|----------------------------|---------|
| **ShippingService** | Camada que recebe a Strategy e aplica regras comuns: logs, validações, métricas. Não sabe detalhes de cálculo. | ```python\n@dataclass\nclass ShippingService:\n    strategy: ShippingStrategy\n    def get_shipping_cost(self, weight: float) -> float:\n        print(f"Calculando frete para {weight}kg...")\n        return self.strategy.calculate(weight)\n``` |
| **Valor de Negócio** | Centraliza regras globais de negócio, deixando a lógica específica encapsulada na Strategy. | Pode adicionar validação de peso negativo ou métricas de performance sem alterar cada transportadora. |

---

## 5. Testabilidade

| Termo Técnico | Explicação  | Exemplo |
|---------------|----------------------------|---------|
| **Injeção de Dependência** | Permite trocar a Strategy por uma **FakeStrategy** em testes, sem depender de APIs reais. | ```python\nclass FakeStrategy:\n    def calculate(self, weight: float) -> float:\n        return 42.0\nservice = ShippingService(strategy=FakeStrategy())\ncost = service.get_shipping_cost(10.0)\nprint(cost)  # Sempre 42.0\n``` |
| **Valor de Negócio** | Garante testes confiáveis e rápidos. Sem risco de falha de transportadora real ou instabilidade de API. | Permite simular cenários complexos de negócio sem custo real. |

---

### ✅ Resumo de boas práticas
- Use **Protocol** para contratos flexíveis e Pythonic.  
- Use **Strategies** para encapsular regras específicas.  
- Centralize lógica comum no **Service Layer**.  
- Teste com **FakeStrategies** para isolar dependências externas.  
- Este padrão garante **flexibilidade, testabilidade e clareza de responsabilidades**.
