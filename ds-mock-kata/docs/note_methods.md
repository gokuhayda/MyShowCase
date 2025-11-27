# Estudo: @staticmethod vs @classmethod em Python

## TL;DR (Resumo Executivo)

**Use `@classmethod` quando:** O método precisa acessar atributos da classe (constantes, configurações) ou você planeja usar herança para variar comportamentos.

**Use `@staticmethod` quando:** O método é uma função utilitária independente que não precisa de nada da classe - está ali só por organização/namespace.

**Regra de ouro:** Se você escrever o nome da classe hardcoded dentro do método (ex: `ScoringLogic.AGE_WEIGHT`), provavelmente deveria ser `@classmethod`.

---

## 1. Resumo Conceitual

| Tipo | Comportamento | Descrição |
| :--- | :--- | :--- |
| **`@staticmethod`** | **Cego** | Não sabe qual classe o chamou. Funciona como uma função solta que apenas "mora" dentro da classe por organização. |
| **`@classmethod`** | **Consciente** | Recebe a própria classe como primeiro argumento (`cls`). Ele consegue ler atributos da classe (como pesos `AGE_WEIGHT`) dinamicamente. |

---

## 2. Comparação Prática

### A. Usando `@staticmethod` (O jeito "Rígido")

Se você mantiver como estático, é obrigado a escrever o nome da classe explicitamente para acessar as constantes.

```python
class ScoringLogic:
    AGE_WEIGHT = 0.10

    @staticmethod
    def calculate_base_score(age: int):
        # RUIM: Acopla o método ao nome exato da classe "ScoringLogic".
        # Se renomear a classe ou usar herança, a referência permanece fixa.
        return age * ScoringLogic.AGE_WEIGHT 
```

### B. Usando `@classmethod` (O jeito "Flexível")

Ao usar método de classe, ganha-se o parâmetro `cls`.

```python
class ScoringLogic:
    AGE_WEIGHT = 0.10

    @classmethod
    def calculate_base_score(cls, age: int):
        # BOM: "cls" refere-se a QUEM chamou o método (esta classe ou uma filha).
        # Permite polimorfismo de constantes.
        return age * cls.AGE_WEIGHT
```

### C. Quando `@staticmethod` é a Escolha CERTA

Nem tudo precisa ser `@classmethod`! Use estático para funções utilitárias puras que não dependem de nada da classe:

```python
class ScoringLogic:
    AGE_WEIGHT = 0.10

    @staticmethod
    def validate_cpf(cpf: str) -> bool:
        """
        Validação pura de CPF - não precisa de cls nem de atributos da classe.
        Está aqui apenas por organização (agrupa lógicas relacionadas a scoring).
        """
        cpf = ''.join(filter(str.isdigit, cpf))
        if len(cpf) != 11:
            return False
        # ... lógica de validação dos dígitos verificadores ...
        return True
    
    @staticmethod
    def format_currency(value: float) -> str:
        """Formatação pura - função matemática que não olha para a classe."""
        return f"R$ {value:,.2f}".replace(',', '_').replace('.', ',').replace('_', '.')
```

**Por que estático aqui?** Essas funções não leem `AGE_WEIGHT`, `INCOME_WEIGHT` nem nada da classe. São ferramentas independentes que poderiam viver fora da classe, mas ficam dentro para manter a coesão do código.

---

## 3. Por que isso importa? (O "Pulo do Gato")

A grande vantagem surge ao criar variações da lógica (Herança/Polimorfismo) sem reescrever o cálculo. Se precisar de uma lógica para Clientes VIP, onde o peso da idade é menor:

### Exemplo em Ação: Herança de Constantes

```python
class ScoringLogic:
    """Regra Padrão"""
    AGE_WEIGHT: float = 0.10
    INCOME_WEIGHT: float = 0.001
    HISTORY_WEIGHT: int = 5

    @classmethod
    def calculate_base_score(cls, age: int, income: float, history: int) -> float:
        # O segredo: 'cls' será substituído pela classe que chamou o método
        print(f"Calculando usando a classe: {cls.__name__}")
        print(f"Peso usado para idade: {cls.AGE_WEIGHT}")
        
        return (age * cls.AGE_WEIGHT) + \
               (income * cls.INCOME_WEIGHT) + \
               (history * cls.HISTORY_WEIGHT)

class VIPScoringLogic(ScoringLogic):
    """
    Regra VIP: Herda tudo, mas altera os pesos.
    Não precisamos copiar e colar a lógica de cálculo!
    """
    AGE_WEIGHT: float = 0.05   # Idade pesa metade
    HISTORY_WEIGHT: int = 10   # Histórico pesa o dobro
```

#### Como `cls` funciona na prática (Fluxo de Resolução):

```
Chamada: VIPScoringLogic.calculate_base_score(30, 5000, 2)
          │
          ├──> Python procura o método em VIPScoringLogic
          │    (não encontra, porque foi herdado)
          │
          ├──> Sobe para a classe pai: ScoringLogic
          │    (encontra calculate_base_score)
          │
          └──> IMPORTANTE: Passa VIPScoringLogic como 'cls'
               (não ScoringLogic!)
               
Dentro do método:
  cls.AGE_WEIGHT  →  VIPScoringLogic.AGE_WEIGHT  →  0.05 ✓
  cls.__name__    →  "VIPScoringLogic"
```

**Comparação visual:**

```
┌─────────────────────────────────────────────────────────┐
│ ScoringLogic (Classe Pai)                              │
│ ─────────────────────────────────────────────────────── │
│ AGE_WEIGHT = 0.10                                       │
│ INCOME_WEIGHT = 0.001                                   │
│ HISTORY_WEIGHT = 5                                      │
│                                                         │
│ @classmethod                                            │
│ calculate_base_score(cls, ...):                         │
│     return age * cls.AGE_WEIGHT + ...                   │
│              ↑                                          │
│              └── "cls" aponta para quem chamou!         │
└─────────────────────────────────────────────────────────┘
                          │
                          │ herda o método
                          ↓
┌─────────────────────────────────────────────────────────┐
│ VIPScoringLogic (Classe Filha)                         │
│ ─────────────────────────────────────────────────────── │
│ AGE_WEIGHT = 0.05      ← Sobrescreve o peso!            │
│ HISTORY_WEIGHT = 10    ← Sobrescreve o peso!            │
│                                                         │
│ (herda calculate_base_score, mas quando executado,     │
│  cls.AGE_WEIGHT vai ler 0.05, não 0.10!)               │
└─────────────────────────────────────────────────────────┘
```

### Executando o Teste

```python
dados_cliente = {"age": 30, "income": 5000.0, "history": 2}

print("--- Cenário 1: Cliente Comum ---")
ScoringLogic.calculate_base_score(**dados_cliente)
# Resultado: 18.0 
# (30*0.1) + (5000*0.001) + (2*5) = 3 + 5 + 10

print("\n--- Cenário 2: Cliente VIP ---")
VIPScoringLogic.calculate_base_score(**dados_cliente)
# Resultado: 26.5 
# (30*0.05) + (5000*0.001) + (2*10) = 1.5 + 5 + 20
```

### O Perigo do `@staticmethod` na Herança

Se `calculate_base_score` fosse estático, ele apontaria fixamente para `ScoringLogic.AGE_WEIGHT` (0.10). Ao chamar através da classe VIP, ele ignoraria o `AGE_WEIGHT = 0.05` e calcularia o valor errado (18.0 em vez de 26.5).

---

## 4. Casos Edge e Comportamentos Especiais

### A. Chamando `@classmethod` via Instância

```python
# Funciona, mas é incomum
cliente_vip = VIPScoringLogic()
score = cliente_vip.calculate_base_score(30, 5000, 2)
# 'cls' ainda aponta para VIPScoringLogic (a classe da instância)
```

**Boa prática:** Chame métodos de classe diretamente pela classe (`VIPScoringLogic.calculate_base_score(...)`), não por instâncias. Isso deixa claro que o método não usa estado da instância (`self`).

### B. Testando com Mocks (unittest)

```python
import unittest
from unittest.mock import patch

class TestScoring(unittest.TestCase):
    
    def test_classmethod_respeita_heranca(self):
        """@classmethod lê dinamicamente os atributos da classe"""
        # Não precisa mockar - basta criar a subclasse
        class TestScoring(ScoringLogic):
            AGE_WEIGHT = 1.0  # Peso exagerado para teste
        
        result = TestScoring.calculate_base_score(10, 0, 0)
        self.assertEqual(result, 10.0)  # 10 * 1.0 = 10
    
    def test_staticmethod_ignora_heranca(self):
        """@staticmethod sempre usa referência hardcoded"""
        # Se calculate_base_score fosse estático e usasse ScoringLogic.AGE_WEIGHT:
        class TestScoring(ScoringLogic):
            AGE_WEIGHT = 1.0
        
        # O resultado seria ERRADO porque o método olha para ScoringLogic, não TestScoring
        # (Este teste falharia se o método fosse estático!)
    
    @patch.object(ScoringLogic, 'AGE_WEIGHT', 0.5)
    def test_mock_atributo_classe(self, mock_weight):
        """Mockando atributos de classe funciona com @classmethod"""
        result = ScoringLogic.calculate_base_score(10, 0, 0)
        self.assertEqual(result, 5.0)  # 10 * 0.5 = 5
```

### C. Interação com `super()`

```python
class AuditedScoringLogic(VIPScoringLogic):
    """Adiciona auditoria antes do cálculo"""
    
    @classmethod
    def calculate_base_score(cls, age: int, income: float, history: int) -> float:
        print(f"[AUDIT] Cálculo iniciado para classe: {cls.__name__}")
        
        # super() funciona normalmente com @classmethod
        result = super().calculate_base_score(age, income, history)
        
        print(f"[AUDIT] Resultado: {result}")
        return result

# A cadeia de herança funciona perfeitamente:
# AuditedScoringLogic → VIPScoringLogic → ScoringLogic
# E 'cls' sempre aponta para AuditedScoringLogic em toda a cadeia!
```

---

## 5. Regra de Decisão (DRY)

### Tabela de Decisão Rápida

| Pergunta | Resposta | Use |
|:---------|:---------|:----|
| O método acessa atributos da classe? | **Sim** | `@classmethod` |
| Você planeja usar herança para variar comportamento? | **Sim** | `@classmethod` |
| É um construtor alternativo (factory method)? | **Sim** | `@classmethod` |
| É uma função utilitária pura (tipo `math.sqrt`)? | **Sim** | `@staticmethod` |
| Não usa `self` nem `cls`? | **Sim** | `@staticmethod` |

### Exemplos do Mundo Real

#### Use `@classmethod` para:

```python
class DatabaseConnection:
    DEFAULT_TIMEOUT = 30
    
    @classmethod
    def from_env(cls):
        """Factory method - alternativa ao __init__"""
        import os
        return cls(
            host=os.getenv('DB_HOST'),
            timeout=cls.DEFAULT_TIMEOUT  # ← Lê da classe
        )

class Configuration:
    API_VERSION = "v2"
    
    @classmethod
    def get_endpoint(cls, resource: str) -> str:
        """Monta URL usando versão da API"""
        return f"https://api.example.com/{cls.API_VERSION}/{resource}"
```

#### Use `@staticmethod` para:

```python
class DateUtils:
    @staticmethod
    def is_weekend(date) -> bool:
        """Pura lógica de data - não precisa de nada da classe"""
        return date.weekday() >= 5
    
    @staticmethod
    def format_br_date(date) -> str:
        """Formatação pura"""
        return date.strftime("%d/%m/%Y")

class Validators:
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validação pura - poderia ser função global"""
        import re
        return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))
```

### Checklist Final

Antes de escolher, pergunte-se:

1. ✅ **Escrevi o nome da classe dentro do método?** (ex: `ScoringLogic.AGE_WEIGHT`)  
   → Mude para `@classmethod` e use `cls.AGE_WEIGHT`

2. ✅ **Vou criar subclasses com comportamentos diferentes?**  
   → Use `@classmethod` para permitir polimorfismo

3. ✅ **É um factory method / construtor alternativo?**  
   → Use `@classmethod` (padrão Python: `dict.fromkeys()`, `datetime.now()`)

4. ✅ **O método poderia viver fora da classe sem perder sentido?**  
   → Talvez seja `@staticmethod` (ou até função global)

5. ✅ **Estou apenas agrupando funções utilitárias por organização?**  
   → `@staticmethod` está OK aqui

---

## Referências e Leitura Adicional

- [PEP 3115](https://peps.python.org/pep-3115/) - Métodos de classe e estáticos
- [Python Data Model - Special Methods](https://docs.python.org/3/reference/datamodel.html#special-method-names)
- Design Pattern: Factory Method com `@classmethod`
