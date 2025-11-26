# Guia de Arquitetura Limpa - Design Patterns & Princípios

## Índice

1. [API Client - Boundary Layer](#api-client---boundary-layer)
2. [Lógica Pura - Core do Negócio](#lógica-pura---core-do-negócio)
3. [Orchestrator - Dependency Injection](#orchestrator---dependency-injection)

---

## API Client - Boundary Layer (I/O Isolado)

### 📋 Visão Geral

Este arquivo representa a camada de **FRONTEIRA** entre nosso sistema e o mundo externo.
Aqui ficam as chamadas HTTP, leitura de arquivos, banco de dados, etc.

### 🎯 Princípios Aplicados

#### 1. Single Responsibility Principle
- Esta classe tem **UMA** responsabilidade: comunicar com a API externa
- Não faz cálculos, não orquestra fluxo, apenas I/O

#### 2. Boundary Pattern (Clean Architecture)
- Isola o mundo externo do nosso Core
- Facilita trocar implementações (ex: mudar de REST para GraphQL)
- Permite mockar facilmente em testes

#### 3. Fail Fast & Explicit
- Erros de I/O são propagados claramente
- Não esconde problemas de rede/API

### 📚 Conceito: Hexagonal Architecture (Ports & Adapters)

```
         ┌─────────────────────┐
         │   APLICAÇÃO (Core)  │
         │   • ScoringLogic    │
         │   • Orchestrator    │
         └──────────┬──────────┘
                    │ usa (interface/port)
         ┌──────────▼──────────┐
         │   ADAPTER/BOUNDARY  │  ← VOCÊ ESTÁ AQUI!
         │   • CustomerApiClient│
         └──────────┬──────────┘
                    │ fala com
         ┌──────────▼──────────┐
         │   MUNDO EXTERNO     │
         │   • API REST        │
         │   • Database        │
         │   • S3              │
         └─────────────────────┘
```

### 💡 Benefícios desta Separação

#### 1. Testabilidade
- Em testes, mockamos ESTA classe
- Não precisamos de servidor HTTP rodando

#### 2. Flexibilidade
- Mudar de requests para httpx? Só muda aqui!
- Mudar de REST para GraphQL? Só muda aqui!
- Adicionar cache? Só muda aqui!

#### 3. Manutenibilidade
- Todos os problemas de rede estão concentrados aqui
- Fácil adicionar retry logic, circuit breaker, etc.

### 📝 Responsabilidades

Cliente para comunicação com API externa de dados de clientes.

**Responsabilidade Única: APENAS falar com a API.**

#### Esta classe NÃO faz:
- ❌ Cálculos de score (isso é ScoringLogic)
- ❌ Orquestração de fluxo (isso é Orchestrator)
- ❌ Validação de regras de negócio (isso é Core)

#### Esta classe FAZ:
- ✅ Chamadas HTTP para API externa
- ✅ Serialização/Deserialização de JSON
- ✅ (Futuro) Retry logic, timeout, error handling

### 🎓 Design Pattern: Adapter Pattern

Esta classe é um "Adapter" que adapta a interface da API externa para o formato que nossa aplicação espera.

**Exemplo de mudança isolada:**

Se amanhã a API mudar de:
```
GET /customers/{id}
```

Para:
```
POST /api/v2/customer-data
```

**Só precisamos mudar ESTA classe!**

O resto do código (Orchestrator, Logic) permanece intacto.

---

## Lógica Pura - Core do Negócio (SEM I/O)

### 📋 Visão Geral

Este arquivo contém **APENAS** lógica de negócio pura.
Nenhuma dependência externa. Nenhum I/O. Apenas matemática e regras.

### 🎯 Princípios Aplicados

#### 1. Pure Functions (Funções Puras)
- Mesma entrada → Mesma saída (determinístico)
- Sem efeitos colaterais (não muda estado externo)
- Sem I/O (não lê arquivos, banco, rede)

#### 2. Testabilidade
- Não precisa de mocks!
- Testes são RÁPIDOS (milissegundos)
- Testes são CONFIÁVEIS (não dependem de rede/disco)

#### 3. Single Responsibility
- Esta classe só cuida das REGRAS DE NEGÓCIO
- Nada de HTTP, banco de dados, ou modelos ML

### 📚 Conceito: Separation of Concerns

Em Clean Architecture, dividimos o código em camadas:

```
┌─────────────────────────────────────┐
│         CORE (Lógica Pura)          │  ← VOCÊ ESTÁ AQUI!
│  • Regras de negócio                │
│  • Sem I/O                          │
│  • Testável sem mocks               │
└─────────────────────────────────────┘
           ↑
           │ Usa (sem dependência de infraestrutura)
           │
┌─────────────────────────────────────┐
│     BOUNDARY (I/O / Infraestrutura) │
│  • HTTP, Banco, Arquivos            │
│  • Modelos ML (carregamento)        │
│  • Testável COM mocks               │
└─────────────────────────────────────┘
```

### 💡 Regra de Ouro

**O CORE nunca depende do BOUNDARY.**
**O BOUNDARY pode usar o CORE.**

### 📝 Características da Lógica Pura

#### Esta classe não tem:
- ❌ Dependências externas (sem imports de libs de I/O)
- ❌ Estado mutável (stateless)
- ❌ Efeitos colaterais (não muda nada fora dela)

#### Esta classe tem:
- ✅ Regras de negócio explícitas
- ✅ Funções puras (determinísticas)
- ✅ Facilidade de teste (sem mocks!)

### 🎓 Design Pattern: Strategy Pattern (implícito)

Ao isolar a lógica aqui, facilitamos trocar a estratégia de cálculo.

**Amanhã, se a regra de negócio mudar:**
- Não preciso mexer em I/O
- Não preciso mexer em orquestração
- Só mudo esta classe!

**Exemplos de mudanças futuras fáceis:**
- "Idade agora vale 0.2 pontos" → Muda só aqui
- "Renda deve ser dividida por 1500" → Muda só aqui
- "Histórico tem peso exponencial" → Muda só aqui

### 📊 Regra de Negócio: Cálculo de Score Base

#### Fórmula:

```
Score Base = (Idade × 0.1) + (Renda ÷ 1.000) + (Histórico × 5)
```

**Onde:**
- **Idade:** Anos de vida (ex: 30 anos → 3.0 pontos)
- **Renda:** Renda mensal em R$ (ex: R$ 50.000 → 50.0 pontos)
- **Histórico:** Anos como cliente (ex: 2 anos → 10.0 pontos)

#### Assinatura da Função:

```python
def calculate_base_score(age: int, income: float, history: int) -> float:
    """
    Args:
        age: Idade do cliente em anos (ex: 30)
        income: Renda mensal em reais (ex: 50000.0)
        history: Anos como cliente (ex: 2)
        
    Returns:
        Score base calculado
    """
```

#### Exemplos:

**Exemplo 1:**
```python
>>> ScoringLogic.calculate_base_score(30, 50000, 2)
63.0

# Explicação: (30 * 0.1) + (50000 / 1000) + (2 * 5)
#           = 3.0 + 50.0 + 10.0
#           = 63.0
```

**Exemplo 2:**
```python
>>> ScoringLogic.calculate_base_score(25, 30000, 1)
37.5

# Explicação: (25 * 0.1) + (30000 / 1000) + (1 * 5)
#           = 2.5 + 30.0 + 5.0
#           = 37.5
```

### 🎯 Por que esta Função é Testável

1. **Determinística:** Mesma entrada → Mesma saída
2. **Sem I/O:** Não precisa de rede, disco, ou serviços
3. **Sem estado:** Não depende de variáveis globais
4. **Rápida:** Executa em microssegundos

### 🎯 Design Decision: Multiplicação vs Soma

**Por que multiplicar e não somar?**

#### Multiplicação:
- ✅ Probabilidade baixa "veta" scores altos
- ✅ Escala naturalmente (0-100% de confiança)
- ✅ Comportamento intuitivo

#### Soma seria problemática:
- ❌ Probabilidade 0 não zeraria o score
- ❌ Escalas incompatíveis (score pode ser 100+, prob é 0-1)

---

## Orchestrator - Dependency Injection

### 📋 Visão Geral

Este arquivo representa a **ORQUESTRAÇÃO** do fluxo de negócio.
Ele coordena diferentes componentes, mas NÃO faz I/O nem lógica complexa.

### 🎯 Princípios Aplicados

#### 1. Dependency Injection (DI)
- As dependências (API client, ML model) são INJETADAS via construtor
- Não criamos instâncias dentro da classe
- Facilita testes: posso injetar mocks!

#### 2. Orchestration (Coordenação)
- Esta classe sabe QUANDO chamar cada componente
- Mas não sabe COMO cada componente funciona internamente
- Delega responsabilidades

#### 3. Inversion of Control (IoC)
- Quem cria a instância controla as dependências
- O orchestrator não decide qual API ou modelo usar
- Isso é decidido "de fora" (no main ou no teste)

### 📚 Conceito: Dependency Injection Pattern

#### ❌ SEM Dependency Injection (Acoplado)

```python
class Orchestrator:
    def __init__(self):
        self.api = CustomerApiClient()  # ← Criado aqui! Acoplado!
        self.model = load_model()        # ← Não consigo trocar!

    def generate_score(self, id):
        data = self.api.get_customer_data(id)  # ← Sempre usa essa API
        # Como testar sem fazer HTTP real?
```

#### ✅ COM Dependency Injection (Desacoplado)

```python
class Orchestrator:
    def __init__(self, api_client, ml_model):  # ← Recebe pronto!
        self.api_client = api_client  # ← Posso injetar mock!
        self.ml_model = ml_model      # ← Posso injetar mock!

    def generate_score(self, id):
        data = self.api_client.get_customer_data(id)
        # Em testes: api_client = Mock()
        # Em produção: api_client = CustomerApiClient()
```

### 💡 Benefícios da Dependency Injection

#### 1. Testabilidade

```python
# Teste com mocks
mock_api = Mock()
mock_model = Mock()
orchestrator = CustomerScoreOrchestrator(mock_api, mock_model)
# ↑ Posso testar sem I/O real!
```

#### 2. Flexibilidade

```python
# Em produção
prod_api = CustomerApiClient()
prod_model = load_ml_model('prod_model.pkl')
orchestrator = CustomerScoreOrchestrator(prod_api, prod_model)

# Em desenvolvimento (com modelo diferente)
dev_model = DummyModel()  # Modelo fake para dev
orchestrator = CustomerScoreOrchestrator(prod_api, dev_model)
```

#### 3. Configurabilidade

```python
# Posso trocar implementações sem mudar código
cached_api = CachedApiClient(CustomerApiClient())
orchestrator = CustomerScoreOrchestrator(cached_api, prod_model)
```

### 🎓 Design Pattern: Facade Pattern (Fachada)

O Orchestrator atua como uma "fachada" simplificando o uso de múltiplos subsistemas:

```
          ┌─────────────────────┐
          │   ORCHESTRATOR      │  ← Interface simples
          │   generate_score()  │
          └──────────┬──────────┘
                     │ delega para
          ┌──────────┴──────────┐
          │                     │
    ┌─────▼─────┐       ┌──────▼──────┐       ┌──────────┐
    │ API Client│       │  ML Model   │       │  Logic   │
    └───────────┘       └─────────────┘       └──────────┘
         I/O                  I/O              Pure Logic
```

**Cliente não precisa saber dos detalhes de cada subsistema!**

### 📝 Responsabilidades do Orchestrator

Orquestra o fluxo de cálculo de score do cliente.

#### Responsabilidades:
1. Coordenar chamadas entre componentes
2. Transformar dados entre formatos
3. Definir a ORDEM de execução

#### NÃO é responsável por:
- ❌ Fazer I/O (delegado para api_client)
- ❌ Calcular lógica de negócio (delegado para ScoringLogic)
- ❌ Executar modelo ML (delegado para ml_model)

### 🎓 Padrão: Strategy Pattern (implícito)

Ao injetar diferentes `api_clients` ou `ml_models`, estamos permitindo diferentes "estratégias" sem mudar código.

**Exemplos:**
- `api_client` pode ser: HTTP, gRPC, Cache, Mock
- `ml_model` pode ser: LogisticRegression, RandomForest, DummyModel

---

## 📚 Resumo dos Padrões

| Padrão | Localização | Benefício Principal |
|--------|-------------|---------------------|
| **Adapter** | API Client | Isola dependências externas |
| **Pure Functions** | Scoring Logic | Testabilidade sem mocks |
| **Dependency Injection** | Orchestrator | Flexibilidade e testabilidade |
| **Facade** | Orchestrator | Simplifica interface complexa |
| **Strategy** | Todo o sistema | Permite trocar implementações |

---

## 🎯 Princípios SOLID Aplicados

- **S**ingle Responsibility: Cada classe tem uma responsabilidade
- **O**pen/Closed: Extensível via injeção de dependência
- **L**iskov Substitution: Mocks substituem implementações reais
- **I**nterface Segregation: Interfaces mínimas e focadas
- **D**ependency Inversion: Core não depende de infraestrutura
