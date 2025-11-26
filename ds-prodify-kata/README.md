# 🏭 Production ML Pipeline

Pipeline de Machine Learning de nível sênior aplicando princípios SOLID, injeção de dependências e arquitetura modular.

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Arquitetura](#-arquitetura)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Conceitos Aplicados](#-conceitos-aplicados)
- [Como Usar](#-como-usar)
- [Componentes](#-componentes)
- [Testes](#-testes)
- [Extensibilidade](#-extensibilidade)

---

## 🎯 Visão Geral

Este projeto demonstra como refatorar um notebook jupyteriano em uma pipeline de produção testável e expansível. A arquitetura separa responsabilidades em componentes independentes que podem ser testados isoladamente.

### Problema Original

```python
# ❌ Código Spaghetti (Notebook)
df = pd.read_csv("vendas.csv")
df = df.dropna()
df['total'] = df['qtd'] * df['preco']
model = LinearRegression()
model.fit(df[['qtd', 'preco']], df['total'])
```

### Solução Sênior

```python
# ✅ Código Modular (Produção)
pipeline = TrainingPipeline(
    loader=CsvLoader(),
    cleaner=SalesDataCleaner(),
    trainer=ModelTrainer()
)
model = pipeline.run("vendas.csv")
```

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────┐
│         TrainingPipeline (Orchestrator)      │
│                                              │
│  ┌────────┐  ┌─────────┐  ┌──────────────┐ │
│  │ Loader │→ │ Cleaner │→ │ ModelTrainer │ │
│  └────────┘  └─────────┘  └──────────────┘ │
└─────────────────────────────────────────────┘
        ↓             ↓              ↓
   LoaderStrategy CleanerStrategy  Trainer
   (Protocol)     (Protocol)       (Class)
```

### Princípios Aplicados

- **Single Responsibility Principle (SRP)**: Cada classe tem uma única razão para mudar
- **Dependency Injection**: Componentes são injetados no construtor
- **Protocol-based Design**: Contratos sem herança
- **Open/Closed Principle**: Extensível sem modificação

---

## 📁 Estrutura do Projeto

```
ds-prodify-kata
/kata/
    ├── loaders.py          # Interface e implementação de carregamento
    ├── cleaners.py         # Interface e implementação de limpeza
    ├── trainers.py         # Lógica de treinamento
    ├── orchestrator.py     # Orquestração da pipeline
└── tests/
    ├── test_loaders.py
    ├── test_cleaners.py
    ├── test_trainers.py
    └── test_pipeline.py
```

---

## 💡 Conceitos Aplicados

### 1. Protocols (Duck Typing Explícito)

```python
# Interface - define o CONTRATO
class LoaderStrategy(Protocol):
    def load(self, path: str) -> pd.DataFrame: ...

# Implementação - cumpre o CONTRATO
class CsvLoader:
    def load(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)
```

**Por que isso importa?**

- Sem herança obrigatória
- Type hints funcionam corretamente
- Facilita mocking em testes
- Permite múltiplas implementações

### 2. Dependency Injection

```python
@dataclass
class TrainingPipeline:
    loader: LoaderStrategy      # ← Injetado
    cleaner: CleanerStrategy    # ← Injetado
    trainer: ModelTrainer       # ← Injetado
```

**Benefícios:**

- Pipeline não conhece implementações concretas
- Fácil trocar componentes
- Testabilidade máxima

### 3. Single Responsibility Principle

Cada classe tem **uma única responsabilidade**:

| Classe | Responsabilidade |
|--------|------------------|
| `CsvLoader` | Carregar dados de CSV |
| `SalesDataCleaner` | Limpar e transformar dados |
| `ModelTrainer` | Treinar modelo |
| `TrainingPipeline` | Orquestrar o fluxo |

---

## 🚀 Como Usar

### Uso Básico

```python
from orchestrator import TrainingPipeline
from loaders import CsvLoader
from cleaners import SalesDataCleaner
from trainers import ModelTrainer

# Monta a pipeline
pipeline = TrainingPipeline(
    loader=CsvLoader(),
    cleaner=SalesDataCleaner(),
    trainer=ModelTrainer()
)

# Executa
model = pipeline.run("vendas.csv")
```

### Saída Esperada

```
🚀 Iniciando Pipeline de Produção...
📂 Lendo CSV: vendas.csv
🧹 Limpando dados...
🤖 Treinando modelo com 3 linhas...
✅ Pipeline finalizada com sucesso!
```

---

## 🔧 Componentes

### 1. LoaderStrategy (loaders.py)

**Interface:**
```python
class LoaderStrategy(Protocol):
    def load(self, path: str) -> pd.DataFrame: ...
```

**Implementação:**
```python
class CsvLoader:
    def load(self, path: str) -> pd.DataFrame:
        print(f"📂 Lendo CSV: {path}")
        return pd.DataFrame({
            'qtd': [1, 2, None, 4], 
            'preco': [10.0, 20.0, 30.0, 40.0]
        })
```

**Responsabilidade:** Carregar dados da fonte

### 2. CleanerStrategy (cleaners.py)

**Interface:**
```python
class CleanerStrategy(Protocol):
    def clean(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

**Implementação:**
```python
class SalesDataCleaner:
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🧹 Limpando dados...")
        df = df.dropna().copy()
        df['total'] = df['qtd'] * df['preco']
        return df
```

**Responsabilidade:** Limpar e transformar dados

### 3. ModelTrainer (trainers.py)

```python
class ModelTrainer:
    def train(self, df: pd.DataFrame) -> Any:
        X = df[['qtd', 'preco']]
        y = df['total']
        
        model = LinearRegression()
        model.fit(X, y)
        return model
```

**Responsabilidade:** Treinar modelo com dados limpos

### 4. TrainingPipeline (orchestrator.py)

```python
@dataclass
class TrainingPipeline:
    loader: LoaderStrategy
    cleaner: CleanerStrategy
    trainer: ModelTrainer
    
    def run(self, input_path: str) -> Any:
        raw_data = self.loader.load(input_path)
        clean_data = self.cleaner.clean(raw_data)
        model = self.trainer.train(clean_data)
        return model
```

**Responsabilidade:** Orquestrar o fluxo de execução

---

## 🧪 Testes

### Por que essa arquitetura facilita testes?

Cada componente pode ser testado **isoladamente**, sem dependências externas.

### Exemplo: Testando SalesDataCleaner

```python
import pandas as pd
from cleaners import SalesDataCleaner

def test_cleaner_removes_nulls_and_creates_total():
    # Arrange
    df = pd.DataFrame({
        "qtd": [1, None, 3],
        "preco": [10, 20, 30]
    })
    cleaner = SalesDataCleaner()
    
    # Act
    result = cleaner.clean(df)
    
    # Assert
    assert len(result) == 2  # Linha com None foi removida
    assert list(result["total"]) == [10, 90]  # 1*10, 3*30
```

### Exemplo: Testando com Mock

```python
from unittest.mock import Mock
from orchestrator import TrainingPipeline

def test_pipeline_calls_components_in_order():
    # Arrange
    mock_loader = Mock()
    mock_loader.load.return_value = pd.DataFrame({"qtd": [1], "preco": [10]})
    
    mock_cleaner = Mock()
    mock_cleaner.clean.return_value = pd.DataFrame({"qtd": [1], "preco": [10], "total": [10]})
    
    mock_trainer = Mock()
    mock_trainer.train.return_value = "trained_model"
    
    pipeline = TrainingPipeline(
        loader=mock_loader,
        cleaner=mock_cleaner,
        trainer=mock_trainer
    )
    
    # Act
    result = pipeline.run("fake_path.csv")
    
    # Assert
    mock_loader.load.assert_called_once_with("fake_path.csv")
    mock_cleaner.clean.assert_called_once()
    mock_trainer.train.assert_called_once()
    assert result == "trained_model"
```

### Cobertura de Testes

| Componente | Tipo de Teste | O que Testar |
|------------|---------------|--------------|
| `CsvLoader` | Unitário | Carregamento correto |
| `SalesDataCleaner` | Unitário | Remoção de nulls, cálculo de total |
| `ModelTrainer` | Unitário | Preparação de X e y, chamada do fit |
| `TrainingPipeline` | Integração | Fluxo completo com mocks |

---

## 🔄 Extensibilidade

### Adicionando Nova Fonte de Dados

```python
class BigQueryLoader:
    def load(self, path: str) -> pd.DataFrame:
        # path seria uma query SQL
        from google.cloud import bigquery
        client = bigquery.Client()
        return client.query(path).to_dataframe()

# Usar na pipeline SEM MUDAR NADA
pipeline = TrainingPipeline(
    loader=BigQueryLoader(),  # ← Nova implementação
    cleaner=SalesDataCleaner(),
    trainer=ModelTrainer()
)
```

### Adicionando Nova Estratégia de Limpeza

```python
class AdvancedDataCleaner:
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Lógica mais sofisticada
        df = df.dropna()
        df = self.remove_outliers(df)
        df = self.feature_engineering(df)
        df['total'] = df['qtd'] * df['preco']
        return df
    
    def remove_outliers(self, df): ...
    def feature_engineering(self, df): ...

# Usar na pipeline SEM MUDAR NADA
pipeline = TrainingPipeline(
    loader=CsvLoader(),
    cleaner=AdvancedDataCleaner(),  # ← Nova implementação
    trainer=ModelTrainer()
)
```

### Adicionando Novo Modelo

```python
class XGBoostTrainer:
    def train(self, df: pd.DataFrame) -> Any:
        import xgboost as xgb
        X = df[['qtd', 'preco']]
        y = df['total']
        
        model = xgb.XGBRegressor()
        model.fit(X, y)
        return model

# Usar na pipeline SEM MUDAR NADA
pipeline = TrainingPipeline(
    loader=CsvLoader(),
    cleaner=SalesDataCleaner(),
    trainer=XGBoostTrainer()  # ← Nova implementação
)
```

---

## 🎓 Cenário de Entrevista

### Pergunta: "O modelo está treinando com dados sujos. Onde você investiga?"

**Opções:**
- A) `TrainingPipeline`
- B) `ModelTrainer`
- C) `SalesDataCleaner`

**Resposta Correta: C**

**Justificativa:**

1. **Pipeline (A)** apenas orquestra - não transforma dados
2. **Trainer (B)** apenas treina com o que recebe - não valida
3. **Cleaner (C)** é responsável pela qualidade dos dados

**Teste para verificar:**

```python
def test_cleaner_handles_edge_cases():
    df = pd.DataFrame({
        "qtd": [1, None, -5, 0],  # casos extremos
        "preco": [10, 20, 30, 0]
    })
    
    cleaner = SalesDataCleaner()
    result = cleaner.clean(df)
    
    # Verificar se dados sujos foram tratados
    assert result['qtd'].isnull().sum() == 0
    assert (result['qtd'] > 0).all()
```

---

## 📊 Comparação: Antes vs Depois

| Aspecto | Notebook (Antes) | Pipeline (Depois) |
|---------|------------------|-------------------|
| **Testabilidade** | ❌ Impossível testar | ✅ 100% testável |
| **Manutenção** | ❌ Código acoplado | ✅ Componentes isolados |
| **Extensibilidade** | ❌ Requer reescrever | ✅ Adicionar nova classe |
| **Reusabilidade** | ❌ Copiar/colar | ✅ Importar módulo |
| **Debugging** | ❌ Global scope | ✅ Isolamento claro |
| **Colaboração** | ❌ Um único arquivo | ✅ Múltiplos módulos |

---

## 🏆 Benefícios da Arquitetura

### 1. Testabilidade Total
- Testes unitários sem arquivos reais
- Mocks simples e eficazes
- Isolamento de componentes

### 2. Manutenção Facilitada
- Bug em limpeza? → Olhe `cleaners.py`
- Bug em carregamento? → Olhe `loaders.py`
- Mudança de escopo? → Substitua um componente

### 3. Onboarding Rápido
- Estrutura clara
- Responsabilidades óbvias
- Documentação via tipo

### 4. Produção-Ready
- Logging centralizado (pode adicionar)
- Error handling modular
- Configuração externa (pode adicionar)
- CI/CD friendly

---

## 🚦 Próximos Passos

1. **Adicionar logging estruturado**
   ```python
   import logging
   logger = logging.getLogger(__name__)
   ```

2. **Adicionar configuração externa**
   ```python
   from dataclasses import dataclass
   
   @dataclass
   class PipelineConfig:
       input_path: str
       model_type: str
       cleaning_strategy: str
   ```

3. **Adicionar validação de dados**
   ```python
   from pydantic import BaseModel, validator
   
   class SalesData(BaseModel):
       qtd: int
       preco: float
       
       @validator('qtd')
       def qtd_must_be_positive(cls, v):
           if v <= 0:
               raise ValueError('qtd must be positive')
           return v
   ```

4. **Adicionar métricas e monitoramento**
   ```python
   from dataclasses import dataclass
   from datetime import datetime
   
   @dataclass
   class PipelineMetrics:
       start_time: datetime
       end_time: datetime
       rows_processed: int
       rows_cleaned: int
       model_accuracy: float
   ```

---

## 📚 Referências

- [Python Protocols - PEP 544](https://peps.python.org/pep-0544/)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Dependency Injection Pattern](https://en.wikipedia.org/wiki/Dependency_injection)
- [Martin Fowler - Refactoring](https://refactoring.com/)

---

## 🤝 Contribuindo

Este projeto é um exemplo didático. Contribuições são bem-vindas:

1. Adicione novos Loaders (Parquet, BigQuery, S3)
2. Adicione novos Cleaners (outlier removal, feature engineering)
3. Adicione novos Trainers (XGBoost, LightGBM, Neural Networks)
4. Melhore a cobertura de testes

---

## 📝 Licença

Este projeto é livre para uso educacional e demonstração de conceitos.

---

## 👨‍💻 Autor

Desenvolvido como material de estudo para pair programming sênior e entrevistas técnicas.

**Conceitos-chave:** SOLID, Dependency Injection, Protocol-based Design, Test-Driven Development

---

## 🎯 Conclusão

Esta arquitetura transforma código experimental em **código de produção**:

- ✅ Testável
- ✅ Manutenível
- ✅ Extensível
- ✅ Documentado
- ✅ Type-safe

**Perfeito para demonstrar em pair programming em empresas como ThoughtWorks, onde design evolutivo e qualidade de código são fundamentais.**
