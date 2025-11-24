# Gilded Rose Refactoring Kata — Python (OCP + Strategy + Factory)

Este repositório contém uma implementação totalmente refatorada do famoso
**Gilded Rose Kata**, aplicando princípios sólidos de engenharia de software:

- **OCP (Open/Closed Principle)**
- **DIP (Dependency Inversion Principle)**
- **Strategy Pattern**
- **Factory Pattern**
- **Orquestrador isolado da regra de negócio**
- Código limpo, modular e extensível

O resultado é um sistema onde novas regras (novos tipos de itens) podem ser
adicionadas **sem modificar código existente**, apenas criando novas estratégias.

---

## 📂 Estrutura do Projeto

gilded-rose-kata/
│
├── kata/              # 📦 Pacote principal (código-fonte)
│   ├── __init__.py
│   ├── domain.py                  # DOMAIN → Classe Item
│   ├── contract.py                # CONTRACT → Interface UpdateStrategy
│   ├── concrete.py                # CONCRETE → Estratégias (Brie, Normal, etc.)
│   ├── factory.py                 # FACTORY → Decide qual estratégia usar
│   └── orchestrator.py            # ORCHESTRATOR → Classe GildedRose
│
├── tests/                         # 🧪 Testes automatizados
│   ├── __init__.py
│   └── test_gilded_rose.py
├── docs/                          # 📚 Documentação
│   ├── GLOSSARY.md
│
├── main.py                        # 🔥 Executável principal (validação rápida)
│
└── README.md                      # 📘 Documentação geral do projeto

---

## 🧠 Arquitetura (Explicação Rápida)

### **1. Domain (Item)**
- Dado puro.
- Não pode ser modificado (regra original do kata).
- Define: `name`, `sell_in`, `quality`.

### **2. Contract — UpdateStrategy**
Define **o contrato** que todas as estratégias devem seguir:

```python
def update(self, item: Item):
    pass

## 🧩 Garantias Arquiteturais

### ✔ Polimorfismo
Cada item usa sua própria estratégia (`strategy.update(item)`), sem IFs espalhados.

### ✔ DIP — Dependency Inversion Principle
O nível alto (GildedRose) **depende apenas da abstração** (`UpdateStrategy`), nunca das classes concretas.

---

## 🔧 3. Concrete (Estratégias)

Implementações da interface `UpdateStrategy`.  
Cada classe encapsula sua própria regra de atualização:

- `NormalItemStrategy`
- `AgedBrieStrategy`
- `BackStagePassStrategy`
- `SulfurasStrategy`
- `ConjuredItemStrategy`

---

## 🏭 4. Factory

Recebe um `Item` e retorna **a estratégia correta**.

Benefícios:

- Remove IFs de dentro do GildedRose  
- Centraliza decisões  
- Facilita extensões (OCP)

---

## 🎻 5. Orchestrator — `GildedRose`

Responsabilidades:

- Percorrer todos os itens
- Pedir à Factory a estratégia correta
- Chamar `strategy.update(item)`

A classe **não contém lógica de negócio**, apenas coordena.

