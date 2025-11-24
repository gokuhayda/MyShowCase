# Gilded Rose Refactoring Kata — Python (OCP + Strategy + Factory)

Este repositório contém uma implementação totalmente refatorada do famoso **Gilded Rose Kata**, aplicando princípios sólidos de engenharia de software:

- **OCP (Open/Closed Principle)**
- **DIP (Dependency Inversion Principle)**
- **Strategy Pattern**
- **Factory Pattern**
- **Orquestrador isolado da regra de negócio**
- Código limpo, modular e extensível

O resultado é um sistema onde novas regras (novos tipos de itens) podem ser adicionadas **sem modificar código existente**, apenas criando novas estratégias.

---

## 🎯 O Contexto: O Kata de Refatoração Mais Famoso do Mundo

O **Gilded Rose** é o kata de refatoração mais famoso do mundo. Ele simula um sistema de inventário de RPG onde a lógica de atualização de itens é um ninho de ratos de `if/else` aninhados.

### 🌱 Greenfield vs 🏚️ Brownfield

Até agora, você pode ter criado código **novo** (Greenfield). Mas **80% do trabalho de um Sênior é lidar com Código Legado** (Brownfield).

O Gilded Rose Kata ensina exatamente isso:

- Como lidar com código sem testes
- Como refatorar sem quebrar funcionalidades
- Como transformar spaghetti code em arquitetura limpa
- Como aplicar SOLID em código legado

---

## 💀 O Problema (Código Júnior - Apenas leia e chore)

```python
# ❌ CÓDIGO LEGADO (Spaghetti Code)
def update_quality(items):
    for item in items:
        if item.name != "Aged Brie" and item.name != "Backstage passes":
            if item.quality > 0:
                if item.name != "Sulfuras":
                    item.quality = item.quality - 1
        else:
            if item.quality < 50:
                item.quality = item.quality + 1
                if item.name == "Backstage passes":
                    if item.sell_in < 11:
                        if item.quality < 50:
                            item.quality = item.quality + 1
        # ... continua por mais 50 linhas assim ...
```

### 🚨 Problemas Deste Código

1. **Lógica aninhada**: Impossível entender sem debugar
2. **Violação de OCP**: Cada novo item = mais IFs
3. **Sem testes**: Medo de mudar qualquer coisa
4. **Duplicação**: Mesma lógica repetida em vários lugares
5. **Baixa coesão**: Uma função faz tudo
6. **Alto acoplamento**: Tudo depende de tudo

---

## ⚡ A Regra de Ouro da Refatoração

> **"Primeiro garanta os testes, depois mude o código"**

Não vamos refatorar isso "na raça". A abordagem correta é:

1. ✅ Criar testes de caracterização (preservar comportamento atual)
2. ✅ Garantir cobertura de 100% dos casos
3. ✅ Refatorar com segurança (testes passando)
4. ✅ Aplicar padrões (Strategy, Factory, etc.)
5. ✅ Verificar que todos os testes continuam passando

---

## 📂 Estrutura do Projeto

```
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
│
├── docs/                          # 📚 Documentação
│   ├── GLOSSARY.md
│
├── main.py                        # 🔥 Executável principal (validação rápida)
│
└── README.md                      # 📘 Documentação geral do projeto
```

---

## 🧠 Arquitetura (Explicação Rápida)

### **1. Domain (Item)**

- Dado puro
- Não pode ser modificado (regra original do kata)
- Define: `name`, `sell_in`, `quality`

```python
@dataclass
class Item:
    name: str
    sell_in: int
    quality: int
```

### **2. Contract — UpdateStrategy**

Define **o contrato** que todas as estratégias devem seguir:

```python
class UpdateStrategy(Protocol):
    def update(self, item: Item) -> None:
        """Atualiza o item conforme suas regras específicas."""
        ...
```

### **3. Concrete (Estratégias)**

Implementações da interface `UpdateStrategy`.

Cada classe encapsula sua própria regra de atualização:

- `NormalItemStrategy` — Itens comuns
- `AgedBrieStrategy` — Queijo que melhora com o tempo
- `BackStagePassStrategy` — Ingressos que valorizam perto do evento
- `SulfurasStrategy` — Item lendário que nunca muda
- `ConjuredItemStrategy` — Itens que degradam 2x mais rápido

```python
class AgedBrieStrategy:
    def update(self, item: Item) -> None:
        item.sell_in -= 1
        self._increase_quality(item)
        if item.sell_in < 0:
            self._increase_quality(item)
    
    def _increase_quality(self, item: Item) -> None:
        if item.quality < 50:
            item.quality += 1
```

### **4. Factory**

Recebe um `Item` e retorna **a estratégia correta**.

```python
class StrategyFactory:
    @staticmethod
    def create_strategy(item: Item) -> UpdateStrategy:
        if item.name == "Aged Brie":
            return AgedBrieStrategy()
        elif item.name == "Backstage passes":
            return BackStagePassStrategy()
        # ... demais estratégias
```

**Benefícios:**

- Remove IFs de dentro do GildedRose
- Centraliza decisões
- Facilita extensões (OCP)

### **5. Orchestrator — `GildedRose`**

**Responsabilidades:**

- Percorrer todos os itens
- Pedir à Factory a estratégia correta
- Chamar `strategy.update(item)`

```python
class GildedRose:
    def __init__(self, items: list[Item]):
        self.items = items
    
    def update_quality(self) -> None:
        for item in self.items:
            strategy = StrategyFactory.create_strategy(item)
            strategy.update(item)
```

A classe **não contém lógica de negócio**, apenas coordena.

---

## 🧩 Garantias Arquiteturais

### ✔ Polimorfismo

Cada item usa sua própria estratégia `strategy.update(item)`, sem IFs espalhados.

### ✔ OCP — Open/Closed Principle

Para adicionar um novo tipo de item:

1. Criar nova Strategy
2. Registrar na Factory
3. **Pronto!** Nenhum código existente foi modificado

### ✔ DIP — Dependency Inversion Principle

O nível alto (`GildedRose`) **depende apenas da abstração** (`UpdateStrategy`), nunca das classes concretas.

### ✔ SRP — Single Responsibility Principle

Cada classe tem **uma única razão para mudar**:

- `AgedBrieStrategy` → regras do queijo
- `StrategyFactory` → decisão de qual estratégia usar
- `GildedRose` → orquestração

### ✔ Testabilidade

Cada Strategy pode ser testada isoladamente:

```python
def test_aged_brie_increases_quality():
    item = Item("Aged Brie", 10, 20)
    strategy = AgedBrieStrategy()
    strategy.update(item)
    assert item.quality == 21
```

---

## 🧪 Testes

O projeto inclui testes abrangentes com `pytest`:

```bash
# Rodar todos os testes
pytest tests/

# Rodar com cobertura
pytest --cov=kata tests/

# Rodar com verbose
pytest -v tests/
```

**Cobertura de testes:**

- ✅ Testes por estratégia individual
- ✅ Testes de integração do GildedRose
- ✅ Testes de limites (quality 0-50, sell_in negativos)
- ✅ Testes de regressão (comportamento original preservado)

---

## 🚀 Como Executar

```bash
# 1. Clone o repositório
git clone <url-do-repo>
cd gilded-rose-kata

# 2. Crie um ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Instale dependências
pip install -r requirements.txt

# 4. Execute o exemplo
python main.py

# 5. Rode os testes
pytest tests/
```

---

## 📚 Regras de Negócio

### Normal Items

- Perde 1 de qualidade por dia
- Após vencimento (`sell_in < 0`): perde 2 por dia
- Quality nunca pode ser negativa

### Aged Brie

- **Ganha** qualidade com o tempo (+1 por dia)
- Após vencimento: ganha 2 por dia
- Quality máxima: 50

### Backstage Passes

- Ganha qualidade conforme se aproxima do show:
  - Mais de 10 dias: +1
  - 10 dias ou menos: +2
  - 5 dias ou menos: +3
- Após o show (`sell_in < 0`): quality = 0

### Sulfuras (Item Lendário)

- **Nunca perde qualidade**
- **Nunca altera sell_in**
- Quality fixa em 80

### Conjured Items

- Degrada **2x mais rápido** que itens normais
- Antes do vencimento: -2 por dia
- Após vencimento: -4 por dia

---


## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaStrategy`)
3. Commit suas mudanças (`git commit -m 'Add: NovaStrategy para item X'`)
4. Push para a branch (`git push origin feature/NovaStrategy`)
5. Abra um Pull Request

---

## 📝 Licença

Este projeto é baseado no Gilded Rose Kata original e está sob licença MIT.

---

## 🔗 Referências

- [Gilded Rose Kata Original](https://github.com/emilybache/GildedRose-Refactoring-Kata)
- [Refactoring: Improving the Design of Existing Code (Martin Fowler)](https://refactoring.com/)
- [Working Effectively with Legacy Code (Michael Feathers)](https://www.oreilly.com/library/view/working-effectively-with/0131177052/)
- [Clean Code (Robert C. Martin)](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)

---

**Desenvolvido com 💙 aplicando princípios de Clean Code e SOLID**
