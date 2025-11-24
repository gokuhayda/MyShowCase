# 📘 GLOSSÁRIO — Gilded Rose Kata  
### Arquitetura • SOLID • Design Patterns • Metodologia • Termos Técnicos

Este glossário reúne **todos os conceitos teóricos, padrões de projeto, princípios SOLID, metodologia de kata e terminologia técnica** usados na implementação profissional do **Gilded Rose Kata**.

---

# 🏛️ 1. Conceitos Fundamentais de Arquitetura

## **1.1 Domain Layer (Domínio)**
Camada que representa o **mundo real** e seus dados.

No Gilded Rose, o domínio é a entidade:

- `Item` → representa apenas dados (*não contém lógica*).

> Regra oficial do Kata: a classe `Item` não pode ser alterada (“Goblin Rule”).

---

## **1.2 Contract (Interface / Abstração)**
Define **o que** deve ser feito, mas não **como**.

Exemplo:

```python
class UpdateStrategy(ABC):
    @abstractmethod
    def update(self, item):
        ...
## Benefícios

- Permite polimorfismo
- Reduz acoplamento
- Viabiliza o DIP (Dependency Inversion Principle)

## 1. Arquitetura e Padrões

### 1.3 Concrete Implementations (Concreto)

São as classes que implementam as regras reais.

**Exemplos:**

- NormalItemStrategy
- AgedBrieStrategy
- BackStagePassStrategy
- SulfurasStrategy
- ConjuredItemStrategy

Cada uma encapsula uma regra de negócio.

### 1.4 Factory Pattern

Objeto que decide qual estratégia aplicar com base no item.

```python
strategy = StrategyFactory.create(item)
```

**Vantagens:**

- Elimina if/elif espalhados
- Centraliza a decisão
- Facilita OCP (Open/Closed Principle)

### 1.5 Orchestrator Pattern

Classe de alto nível que não implementa regras, apenas coordena:

```python
strategy = StrategyFactory.create_strategy(item)
strategy.update(item)
```

**Ideal para:**

- Manter separação clara de responsabilidades
- Reduzir acoplamento
- Respeitar DIP

## 2. Princípios SOLID Aplicados

### 2.1 SRP — Single Responsibility Principle

Cada classe tem apenas uma responsabilidade:

- `AgedBrieStrategy` → regras do queijo
- `BackStagePassStrategy` → regras do ingresso
- `StrategyFactory` → decide a estratégia
- `GildedRose` → orquestra tudo

### 2.2 OCP — Open/Closed Principle

Código aberto para extensão, fechado para modificação.

Adicionar um novo item:
- Não altera nenhuma classe existente

**Basta:**
1. Criar uma nova Strategy
2. Registrá-la na Factory

### 2.3 LSP — Liskov Substitution Principle

Qualquer Strategy substitui qualquer outra:

```python
strategy: UpdateStrategy
```

Nada quebra, pois todas seguem o mesmo contrato.

### 2.4 ISP — Interface Segregation Principle

A interface contém apenas o necessário:
- 1 único método: `update(item)`

### 2.5 DIP — Dependency Inversion Principle

GildedRose (nível alto) depende da abstração, não do concreto:

```python
strategy: UpdateStrategy
```

**Benefícios:**

- Baixo acoplamento
- Maior extensibilidade
- Testabilidade superior

## 3. Padrões de Projeto Utilizados

### 3.1 Strategy Pattern

Cada regra é uma estratégia isolada.

**Permite:**

- Troca de comportamento em runtime
- Polimorfismo
- Isolamento de regras de negócio

### 3.2 Factory Pattern

Define "qual estratégia usar" sem expor ifs.

### 3.3 Orchestrator Pattern

Separa coordenação da execução.

- `GildedRose` → coordena
- `Strategies` → executam

### 3.4 Encapsulamento

Regras internas em métodos privados:

```python
self._decrease_quality(item)
```

Protege consistência interna.

## 4. Regras de Negócio

### 4.1 Normal Items

- Perde 1 qualidade por dia
- Após vencer: perde 2
- `quality >= 0`

### 4.2 Aged Brie

- Ganha qualidade
- Após vencer: ganha 2
- Máximo = 50

### 4.3 Backstage Pass

- +1 (>10 dias)
- +2 (≤10)
- +3 (≤5)
- Qualidade = 0 após o show

### 4.4 Sulfuras

- Item lendário
- Nunca perde qualidade
- Nunca altera `sell_in`

### 4.5 Conjured

- Degrada 2x mais rápido
- Após vencer → 4 por dia

## 5. Metodologia do Kata

O Gilded Rose Kata é um exercício clássico de:

- Refatoração
- Design orientado a objetos
- Aplicação dos princípios SOLID
- TDD
- Limpeza de código legado

**Objetivo:**

- Preservar comportamento
- Isolar regras
- Remover condicionais
- Permitir extensões limpas

## 6. Testes Utilizados

- `pytest`
- Testes por estratégia
- Testes de integração leve
- Testes extremos dos limites (quality, sell_in)

**Foco:**

- Comportamento determinístico
- Segurança para refatorar

## 7. Pilares de OOP Usados

### Polimorfismo

Uma chamada → muitas implementações:

```python
strategy.update(item)
```

### Encapsulamento

Cada regra isolada em sua própria Strategy.

### Herança

Todas Strategies estendem `UpdateStrategy`.

### Abstração

Define o "o que fazer", não o "como".

## 8. Terminologia Técnica

| Termo | Explicação |
|-------|------------|
| Strategy | Algoritmo intercambiável usado em runtime. |
| Factory | Seleciona e devolve a Strategy certa. |
| Orchestrator | Coordena operações sem implementar regras. |
| Domain Object | Representa dados puros (ex: Item). |
| Refactoring | Melhorar código sem alterar comportamento. |
| Legacy Code | Código sem testes ou estrutura ruim. |
| Business Rule | Regra do domínio. |
| Clean Code | Código simples, direto e legível. |
| Cohesion | Foco de uma classe em uma única tarefa. |
| Coupling | Dependência entre módulos. Quanto menos, melhor. |
| DIP | Alto nível depende de abstrações, não concretos. |
