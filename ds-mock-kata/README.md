
# 🥋 Data Science Mock Katas

> **Pratique testes unitários com mocking em Python através de exercícios progressivos focados em Data Science**

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Por que Mocking?](#por-que-mocking)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Os Katas](#os-katas)
  - [Kata 01: Boundary S3](#kata-01-boundary-s3)
  - [Kata 02: ML Pipeline](#kata-02-ml-pipeline)
  - [Kata 03: Legacy Rescue](#kata-03-legacy-rescue)
- [Conceitos Fundamentais](#conceitos-fundamentais)
- [Recursos Adicionais](#recursos-adicionais)
- [Contribuindo](#contribuindo)

---

## 🎯 Sobre o Projeto

Este repositório contém uma série de **katas** (exercícios de código) focados em ensinar técnicas de **mocking** e **testes unitários** para cientistas de dados. Cada kata aborda um cenário comum em projetos de Data Science onde o mocking é essencial para criar testes rápidos, confiáveis e isolados.

### Objetivos de Aprendizado

- ✅ Entender **quando** e **por que** usar mocks
- ✅ Dominar `unittest.mock` e `@patch`
- ✅ Isolar dependências externas (I/O, APIs, modelos ML)
- ✅ Aplicar princípios de **Clean Architecture** em projetos de DS
- ✅ Refatorar código legado para torná-lo testável

---

## 🤔 Por que Mocking?

Em Data Science, frequentemente trabalhamos com:

- 🌐 **APIs externas** (lento, pode falhar)
- 🗄️ **Bancos de dados** (requer infraestrutura)
- ☁️ **Cloud Storage** (S3, GCS - custo e latência)
- 🤖 **Modelos ML pesados** (5GB+, treino demorado)

**Sem mocking:**
```python
def test_pipeline():
    df = pd.read_csv("s3://bucket/data.csv")  # ❌ Chamada real ao S3
    model.fit(X_train, y_train)                # ❌ Treino real (minutos)
    # Teste lento, instável, caro
```

**Com mocking:**
```python
@patch('module.pd.read_csv')
def test_pipeline(mock_read_csv):
    mock_read_csv.return_value = fake_df     # ✅ Instantâneo
    # Teste rápido, confiável, sem custo
```

### Benefícios do Mocking

| Benefício | Sem Mock | Com Mock |
|-----------|----------|----------|
| **Velocidade** | Segundos/Minutos | Milissegundos |
| **Confiabilidade** | Depende de rede/infra | 100% determinístico |
| **Custo** | Pode gerar custos (API, cloud) | Zero custo |
| **Isolamento** | Testa múltiplas camadas | Testa apenas lógica |

---

## 📁 Estrutura do Projeto

```
ds-mock-kata/
│
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências do projeto
├── main.py                           # Runner interativo para os katas
│
├── docs/                             # Documentação adicional
│   ├── mindset.md                    # Filosofia de testes e mocking
│   └── mock_cheat_sheet.md           # Guia rápido de referência
│
├── katas/                            # Exercícios práticos
│   ├── b01_boundary_s3/              # Kata 01: Mockar I/O com S3
│   │   ├── __init__.py
│   │   └── data_loader.py
│   │
│   ├── b02_ml_pipeline/              # Kata 02: Mockar modelos ML
│   │   ├── __init__.py
│   │   └── model_trainer.py
│   │
│   └── b03_legacy_rescue/            # Kata 03: Refatoração com DI
│       ├── ADDITIONAL.md
│       ├── original.py               # Código legado (antes)
│       └── refactored/               # Código refatorado (depois)
│           ├── api_client.py
│           ├── scoring_logic.py
│           └── orchestrator.py
│
└── tests/                            # Testes unitários
    ├── __init__.py
    ├── test_01_boundary_s3.py
    ├── test_02_ml_pipeline.py
    └── test_03_legacy_rescue.py
```

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.10+
- pip

### Passos

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/ds-mock-kata.git
cd ds-mock-kata

# 2. (Opcional) Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## 💻 Como Usar

### Modo Interativo (Recomendado)

Execute o runner interativo que permite escolher qual kata rodar:

```bash
python main.py
```

Você verá um menu:

```
Escolha o kata para rodar:
1 = Boundary S3
2 = ML Pipeline
3 = Legacy Rescue
4 = Todos
Digite a opção: _
```

### Modo Direto com pytest

Execute todos os testes:

```bash
pytest tests/ -v
```

Execute um kata específico:

```bash
# Kata 01
pytest tests/test_01_boundary_s3.py -v

# Kata 02
pytest tests/test_02_ml_pipeline.py -v

# Kata 03
pytest tests/test_03_legacy_rescue.py -v
```

### Modo Direto com unittest

```bash
# Rodar todos os testes
python -m unittest discover tests

# Rodar kata específico
python -m unittest tests.test_01_boundary_s3
```

---

## 🥋 Os Katas

### Kata 01: Boundary S3

**📚 Conceito:** Isolar I/O externo (Cloud Storage)

#### O Desafio

Você tem uma classe que carrega dados do S3:

```python
class S3DataLoader:
    def load_csv(self, s3_path: str) -> pd.DataFrame:
        return pd.read_csv(s3_path)  # ← Chamada real ao S3!
```

#### O Problema

- ❌ Requer credenciais AWS
- ❌ Requer conexão de rede
- ❌ Lento (latência + download)
- ❌ Pode custar dinheiro

#### A Solução: Mock

```python
@patch("katas.b01_boundary_s3.data_loader.pd.read_csv")
def test_load_csv(self, mock_read_csv):
    # Arrange: Preparar dados falsos
    fake_df = pd.DataFrame({'id': [1, 2, 3]})
    mock_read_csv.return_value = fake_df
    
    # Act: Executar método
    loader = S3DataLoader()
    result = loader.load_csv("s3://bucket/data.csv")
    
    # Assert: Verificar resultado e comportamento
    pd.testing.assert_frame_equal(result, fake_df)
    mock_read_csv.assert_called_once_with("s3://bucket/data.csv")
```

#### 🎯 Lições Aprendidas

1. **Regra de Ouro:** Mock onde a função é **USADA**, não onde é **DEFINIDA**
   - ❌ Errado: `@patch('pandas.read_csv')`
   - ✅ Certo: `@patch('katas.b01_boundary_s3.data_loader.pd.read_csv')`

2. **Verificação Dupla:**
   - **Estado:** O resultado está correto?
   - **Comportamento:** O método foi chamado corretamente?

---

### Kata 02: ML Pipeline

**📚 Conceito:** Isolar modelos ML pesados

#### O Desafio

Você tem uma classe que treina modelos:

```python
class ModelTrainer:
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)  # ← Treino real (lento!)
        predictions = model.predict(X_test)
        return accuracy_score(y_test, predictions)
```

#### O Problema

- ❌ Treinar modelo é lento (segundos a horas)
- ❌ Requer dados reais ou sintéticos grandes
- ❌ Comportamento não-determinístico (random_state pode variar)
- ❌ Testes ficam lentos e flaky

#### A Solução: Mock

```python
@patch('katas.b02_ml_pipeline.model_trainer.RandomForestClassifier')
def test_train_and_evaluate_flow(self, mock_rf_class):
    # Arrange: Configurar mock da classe e instância
    mock_instance = mock_rf_class.return_value
    mock_instance.predict.return_value = np.array([1, 0])
    
    # Act
    trainer = ModelTrainer()
    accuracy = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # Assert
    mock_instance.fit.assert_called_once_with(X_train, y_train)
    mock_instance.predict.assert_called_once_with(X_test)
    self.assertEqual(accuracy, 1.0)
```

#### 🎯 Lições Aprendidas

1. **Mock de Classe vs Instância:**
   - Mockamos a **classe** com `@patch`
   - Configuramos a **instância** com `.return_value`

2. **Verificação de Fluxo:**
   - Testamos se `fit()` foi chamado
   - Testamos se `predict()` foi chamado
   - Testamos se a lógica de acurácia funciona

---

### Kata 03: Legacy Rescue

**📚 Conceito:** Refatorar código legado para torná-lo testável

#### O Desafio

Você herda este código:

```python
# CÓDIGO LEGADO (original.py)
MODEL = load_model()  # ❌ Global!

def generate_customer_score(customer_id: int) -> float:
    # ❌ I/O misturado com lógica
    response = requests.get(f"https://api.fake.com/customers/{customer_id}")
    data = response.json()
    
    # ❌ Lógica enterrada no meio
    base_score = (data["age"] * 0.1) + (data["income"] / 1000)
    
    # ❌ Dependência global
    ml_prob = MODEL.predict_proba([[...]])[0][1]
    
    return base_score * ml_prob
```

#### Os Problemas

1. **Acoplamento a I/O:** Chamada HTTP direta
2. **Dependência Global:** Modelo não pode ser mockado
3. **Lógica Misturada:** Regras de negócio enterradas
4. **Violação SRP:** Faz múltiplas coisas
5. **Não testável:** Impossível testar sem rede/modelo real

#### A Solução: Refatoração com DI

**Passo 1: Separar em camadas**

```python
# api_client.py - BOUNDARY (I/O)
class CustomerApiClient:
    def get_customer_data(self, customer_id: int) -> dict:
        response = requests.get(f"https://api.fake.com/customers/{customer_id}")
        return response.json()

# scoring_logic.py - CORE (Lógica Pura)
class ScoringLogic:
    @staticmethod
    def calculate_base_score(age: int, income: float, history: int) -> float:
        return (age * 0.1) + (income / 1000) + (history * 5)
    
    @staticmethod
    def calculate_final_score(base_score: float, ml_probability: float) -> float:
        return base_score * ml_probability

# orchestrator.py - ORCHESTRATION (Coordenação)
class CustomerScoreOrchestrator:
    def __init__(self, api_client, ml_model):  # ← DEPENDENCY INJECTION
        self.api_client = api_client
        self.ml_model = ml_model
        self.logic = ScoringLogic()
    
    def generate_score(self, customer_id: int) -> float:
        data = self.api_client.get_customer_data(customer_id)
        base = self.logic.calculate_base_score(data['age'], data['income'], data['history'])
        prob = self.ml_model.predict_proba([[...]])[0][1]
        return self.logic.calculate_final_score(base, prob)
```

**Passo 2: Testar cada camada isoladamente**

```python
# Teste 1: Lógica Pura (SEM mocks!)
def test_scoring_logic_math(self):
    base = ScoringLogic.calculate_base_score(30, 1000, 2)
    self.assertEqual(base, 14.0)  # (30*0.1) + (1000/1000) + (2*5)

# Teste 2: Orquestração (COM mocks)
def test_orchestrator_flow(self):
    mock_api = Mock()
    mock_model = Mock()
    mock_api.get_customer_data.return_value = {"age": 30, "income": 1000, "history": 2}
    mock_model.predict_proba.return_value = [[0.99, 0.5]]
    
    orchestrator = CustomerScoreOrchestrator(mock_api, mock_model)
    score = orchestrator.generate_score(999)
    
    mock_api.get_customer_data.assert_called_once_with(999)
    self.assertEqual(score, 7.0)
```

#### 🎯 Lições Aprendidas

1. **Separation of Concerns:**
   - **Boundary:** I/O e dependências externas
   - **Core:** Lógica pura (sem I/O)
   - **Orchestration:** Coordenação com DI

2. **Dependency Injection:**
   - Dependências são **injetadas** (não criadas internamente)
   - Facilita testes com mocks

3. **Testabilidade:**
   - Lógica pura: testa sem mocks (rápido!)
   - Orquestração: testa com mocks (isolado!)

---

## 🧠 Conceitos Fundamentais

### 1. O que é um Mock?

Um **mock** é um objeto falso que simula o comportamento de um objeto real. Usado para:

- Isolar código em teste
- Evitar dependências lentas/custosas
- Controlar comportamento de forma determinística

### 2. Quando Usar Mocks?

✅ **Use mocks quando:**
- Operações de I/O (rede, disco, banco)
- APIs externas
- Modelos ML pesados
- Operações lentas ou custosas
- Comportamento não-determinístico

❌ **Não use mocks quando:**
- Lógica pura (matemática, transformações simples)
- Funções rápidas e sem efeitos colaterais
- Quando o mock seria mais complexo que o código real

### 3. Anatomia de um Mock

```python
from unittest.mock import Mock, patch

# 1. Criar mock manual
mock_api = Mock()
mock_api.get_data.return_value = {"result": "ok"}

# 2. Usar @patch (recomendado)
@patch('module.path.function')
def test_something(self, mock_function):
    mock_function.return_value = "fake_value"
    # seu teste aqui
```

### 4. Padrão AAA (Arrange-Act-Assert)

```python
def test_example(self):
    # ARRANGE: Preparar
    mock = Mock()
    mock.method.return_value = 42
    
    # ACT: Executar
    result = some_function(mock)
    
    # ASSERT: Verificar
    self.assertEqual(result, 42)
    mock.method.assert_called_once()
```

### 5. Regra de Ouro do @patch

**Mock onde a função é USADA, não onde é DEFINIDA!**

```python
# modulo_a.py
def funcao_original():
    return "real"

# modulo_b.py
from modulo_a import funcao_original

def usa_funcao():
    return funcao_original()

# test.py
# ❌ ERRADO
@patch('modulo_a.funcao_original')

# ✅ CERTO
@patch('modulo_b.funcao_original')
```

### 6. Princípios de Clean Architecture

```
┌─────────────────────────┐
│   CORE (Lógica Pura)    │  ← Sem I/O, sem dependências
│   • Regras de negócio   │
│   • Testável sem mocks  │
└────────────┬────────────┘
             │
             │ usa
             │
┌────────────▼────────────┐
│  BOUNDARY (I/O)         │  ← APIs, DB, S3, ML
│  • Isolado              │
│  • Mockável             │
└─────────────────────────┘
```

**Benefícios:**
- Core é rápido de testar (sem mocks)
- Boundaries são isolados (com mocks)
- Mudanças em I/O não afetam lógica

---

## 📚 Recursos Adicionais

### Documentação Oficial

- [unittest.mock - Python Docs](https://docs.python.org/3/library/unittest.mock.html)
- [pytest-mock](https://pytest-mock.readthedocs.io/)

### Artigos Recomendados

- [Stop Mocking, Start Testing](https://nedbatchelder.com/blog/201206/tldw_stop_mocking_start_testing.html)
- [Mocks Aren't Stubs](https://martinfowler.com/articles/mocksArentStubs.html)

### Livros

- **"Clean Architecture"** - Robert C. Martin
- **"Working Effectively with Legacy Code"** - Michael Feathers
- **"Test Driven Development"** - Kent Beck

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Se você tem ideias para novos katas ou melhorias:

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/novo-kata`)
3. Commit suas mudanças (`git commit -m 'Adiciona novo kata'`)
4. Push para a branch (`git push origin feature/novo-kata`)
5. Abra um Pull Request

### Ideias para Novos Katas

- Kata 04: Mockar conexões com banco de dados
- Kata 05: Mockar requisições HTTP com `requests`
- Kata 06: Mockar operações de arquivo (CSV, JSON)
- Kata 07: Mockar bibliotecas de deep learning (TensorFlow, PyTorch)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

## 🙏 Agradecimentos

Este projeto foi criado para ajudar cientistas de dados a dominar testes unitários e mocking, habilidades essenciais para escrever código de produção robusto e manutenível.

**Happy Mocking! 🎭**

---

## 📊 Status do Projeto

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 💬 Contato

Dúvidas? Sugestões? Abra uma [issue](https://github.com/seu-usuario/ds-mock-kata/issues) ou entre em contato!
