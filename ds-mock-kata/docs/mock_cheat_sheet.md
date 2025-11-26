
# 🃏 Cheat Sheet: Python Mocks & Isolation

**Foco:** `unittest.mock` | **Meta:** Testes Rápidos, Determinísticos e Isolados

---

## 👑 A Regra de Ouro (The Golden Rule)

> **"Mock where it is USED, not where it is DEFINED."**

Faça o patch no **namespace onde a dependência é importada/usada**, não na biblioteca original.

### Exemplo Visual

```
┌─────────────────────────┐
│ Biblioteca Original     │
│ (pandas)                │
└───────────┬─────────────┘
            │ import
            │
┌───────────▼─────────────┐
│ Seu Arquivo             │  ← ✅ PATCH AQUI!
│ (data_loader.py)        │
└───────────┬─────────────┘
            │ usa
            │
┌───────────▼─────────────┐
│ Seu Teste               │
│ (test_data_loader.py)   │
└─────────────────────────┘
```

### Regra em Código

Se `data_loader.py` faz:
```python
from pandas import read_csv
```

O Patch deve ser:
```python
@patch('data_loader.read_csv')  # ✅ CORRETO
# NÃO: @patch('pandas.read_csv')  # ❌ ERRADO
```

**Por quê?** Porque `data_loader.py` tem sua própria referência a `read_csv` no seu namespace.

---

## 💰 Mock() vs patch() - A Pergunta de 1 Milhão

> **Esta é a pergunta de "1 milhão de dólares" que define a arquitetura dos seus testes.**

Para um Sênior, a resposta não é apenas "sintaxe", é sobre **Design de Código**.

### 🧠 O Modelo Mental (Analogia do Cinema)

- **`Mock()`** é o **Dublê** 🎭
  - É o ator falso que sabe cair da escada sem se machucar
  - Ele **É** o objeto substituto

- **`patch()`** é o **Diretor** 🎬
  - É quem troca o ator principal pelo dublê numa cena específica
  - Ele **É** o mecanismo de substituição

### 1️⃣ Quando usar `Mock()` (ou `MagicMock`) puro?

**Cenário:** Você usa **Injeção de Dependência (DI)**.

O seu código é limpo e pede as dependências no construtor. Você não precisa de "magia negra" para substituir nada, você apenas entrega o mock na mão da classe.

**Mentalidade:** ✅ **PREFERIDO**. Mostra baixo acoplamento.

```python
# ✅ Código Bem Desenhado (Com DI)
class AnaliseService:
    # Eu peço o cliente. Não crio ele escondido.
    def __init__(self, api_client):
        self.api = api_client
    
    def processar(self):
        return self.api.get_data()

# ✅ Teste (Sem patch!)
def test_analise():
    # 1. Crio o Dublê na mão
    fake_client = MagicMock()
    fake_client.get_data.return_value = {"status": "ok"}
    
    # 2. Entrego o Dublê (Injeção Manual)
    service = AnaliseService(api_client=fake_client)
    
    # 3. Testo naturalmente
    assert service.processar() == {"status": "ok"}
    fake_client.get_data.assert_called_once()
```

**Por que é melhor?**
- ✅ Mais explícito (vejo a dependência no construtor)
- ✅ Mais rápido (não precisa do overhead de `patch`)
- ✅ Refatoração segura (mudanças no código quebram o teste)
- ✅ Mostra que o código está bem arquitetado

### 2️⃣ Quando usar `patch()`?

**Cenário:** O código tem **Dependências Ocultas (Hardcoded)**.

A classe cria o objeto sozinha ou importa uma função globalmente. Você não tem como "passar" o mock, então precisa usar `patch` para invadir o módulo e trocar a referência à força.

**Mentalidade:** ⚠️ **NECESSÁRIO**, mas é sinal de acoplamento forte. Usado em código **Legado** ou **fronteiras de bibliotecas** (pandas, requests).

```python
# ❌ Código Acoplado (Sem DI)
from requests import get  # Dependência "Soldada" no código

class AnaliseService:
    def processar(self):
        # Não tem como injetar nada aqui! O 'get' é global.
        return get("http://api.com").json()

# ⚠️ Teste (Precisa de Cirurgia com Patch)
from unittest.mock import patch

# Tenho que dizer ONDE trocar o 'get'
@patch('meu_modulo.get')
def test_analise(mock_get):
    # Configurar o mock injetado pelo patch
    mock_get.return_value.json.return_value = {"status": "ok"}
    
    # Criar serviço (não recebe nada - dependência oculta)
    service = AnaliseService()
    
    # Testar
    assert service.processar() == {"status": "ok"}
    mock_get.assert_called_once_with("http://api.com")
```

**Quando patch é necessário?**
- ⚠️ Bibliotecas globais que você não controla (`pandas`, `requests`, `boto3`)
- ⚠️ Código legado sem DI
- ⚠️ Métodos estáticos, funções de módulo (`time.sleep`, `random.randint`)
- ⚠️ Imports no topo do arquivo

### 🏆 Tabela de Decisão (O Gráfico Sênior)

| Situação | Ferramenta | Por quê? | Exemplo |
|----------|------------|----------|---------|
| **Código Novo / Clean Arch** | `Mock()` (via DI) | Mais explícito, mais rápido, refatoração segura | `service = Service(api=mock_api)` |
| **Bibliotecas Globais** | `patch()` | Você não controla a biblioteca (pandas, requests, boto3) | `@patch('module.pd.read_csv')` |
| **Código Legado (Spaghetti)** | `patch()` | Difícil refatorar para DI agora, patch "estanca a sangria" | `@patch('legacy.hardcoded_db')` |
| **Métodos Estáticos / Time / Random** | `patch()` | São globais por natureza | `@patch('module.time.sleep')` |
| **Testes de Integração de Camadas** | `Mock()` (via DI) | Testo orquestração entre componentes | `orchestrator = Orch(api=mock, db=mock)` |

### 🎯 A Regra de Ouro da Entrevista

> **"Eu prefiro usar `Mock()` injetado via construtor porque facilita a arquitetura e deixa as dependências explícitas. Mas uso `patch()` quando preciso isolar bibliotecas de terceiros ou código legado que não usa injeção de dependência."**

### 🔄 Comparação Lado a Lado

#### Cenário: Carregar dados de uma API

**Estilo 1: Com DI (Preferido)**
```python
# Código
class DataLoader:
    def __init__(self, api_client):  # ← DI explícita
        self.api = api_client
    
    def load(self):
        return self.api.fetch_data()

# Teste
def test_data_loader():
    mock_api = Mock()
    mock_api.fetch_data.return_value = [1, 2, 3]
    
    loader = DataLoader(api_client=mock_api)  # ← Injeto mock
    assert loader.load() == [1, 2, 3]
```

**Estilo 2: Sem DI (Legado)**
```python
# Código
import requests  # ← Import global

class DataLoader:
    def load(self):
        # Dependência hardcoded!
        return requests.get("http://api.com").json()

# Teste
@patch('my_module.requests.get')  # ← Preciso de patch
def test_data_loader(mock_get):
    mock_get.return_value.json.return_value = [1, 2, 3]
    
    loader = DataLoader()  # Sem argumentos
    assert loader.load() == [1, 2, 3]
```

### 💡 Dica Prática: Quando Refatorar?

Se você se pega usando `patch()` para suas **próprias classes** (não bibliotecas externas), é hora de refatorar para DI:

```python
# ANTES (Ruim - precisa de patch)
class Pipeline:
    def run(self):
        db = Database()  # ← Cria internamente
        return db.query("SELECT *")

@patch('module.Database')  # ← Forçado a usar patch
def test_pipeline(mock_db_class):
    # ...

# DEPOIS (Bom - usa Mock direto)
class Pipeline:
    def __init__(self, database):  # ← DI
        self.db = database
    
    def run(self):
        return self.db.query("SELECT *")

def test_pipeline():
    mock_db = Mock()  # ← Mock direto, sem patch!
    pipeline = Pipeline(database=mock_db)
    # ...
```

### 🎓 Para a Entrevista 

**Se perguntarem:** "Por que você usa `Mock()` em vez de `patch()`?"

**Resposta Sênior:**
> "Quando tenho controle sobre o código, prefiro usar Dependency Injection e passar `Mock()` diretamente. Isso torna as dependências explícitas, facilita testes e melhora o design. Só uso `patch()` quando lido com bibliotecas de terceiros (como pandas ou requests) ou quando estou trabalhando com código legado que ainda não foi refatorado para DI. O `patch()` é uma ferramenta poderosa, mas também é um indicador de que há acoplamento forte no código."

---

## 🛠️ 1. Configurando o Comportamento (O que o Mock faz?)

| Comando | O que faz | Exemplo |
|---------|-----------|---------|
| `return_value` | Retorna um valor fixo sempre que chamado | `mock_api.get.return_value = {'status': 200}` |
| `side_effect` | Lança erro OU retorna valores diferentes em sequência | `mock_db.save.side_effect = TimeoutError`<br>`mock_rand.side_effect = [1, 5, 10]` |
| Atributos | Mocka propriedades/variáveis de instância | `mock_user.name = "Alice"`<br>`mock_user.is_admin = True` |
| `spec=True` | Limita o mock à API real (evita inventar métodos) | `@patch('...', autospec=True)` |

### Exemplos Detalhados

#### return_value - Valor Fixo

```python
mock_api = Mock()
mock_api.get_data.return_value = {"user": "Alice", "age": 30}

# Toda chamada retorna o mesmo
result1 = mock_api.get_data()  # {"user": "Alice", "age": 30}
result2 = mock_api.get_data()  # {"user": "Alice", "age": 30}
```

#### side_effect - Sequência de Valores

```python
mock_random = Mock()
mock_random.randint.side_effect = [1, 5, 10]

# Cada chamada retorna o próximo valor
result1 = mock_random.randint()  # 1
result2 = mock_random.randint()  # 5
result3 = mock_random.randint()  # 10
```

#### side_effect - Exceções

```python
mock_db = Mock()
mock_db.connect.side_effect = ConnectionError("DB offline")

# Lança exceção quando chamado
mock_db.connect()  # Raises: ConnectionError
```

#### autospec - Validação de API

```python
# Sem autospec: permite chamar métodos inexistentes
@patch('my_module.Calculator')
def test_bad(mock_calc):
    mock_calc.invented_method()  # ✅ Não reclama (perigoso!)

# Com autospec: só permite métodos reais
@patch('my_module.Calculator', autospec=True)
def test_good(mock_calc):
    mock_calc.invented_method()  # ❌ AttributeError (seguro!)
```

---

## 🎯 2. Padrões de Patching (Como injetar?)

### A. Decorator (O mais comum)

**Uso:** Ideal para testar a função inteira com o mock ativo.

```python
@patch('my_service.Database')  # ⚠️ Onde é USADO, não definido!
def test_get_user(self, mock_db_class):
    # 1. ARRANGE - Configurar
    mock_instance = mock_db_class.return_value  # A instância criada
    mock_instance.find.return_value = "Alice"
    
    # 2. ACT - Executar
    result = my_service.get_user_name(1)
    
    # 3. ASSERT - Verificar
    assert result == "Alice"
    mock_instance.find.assert_called_once_with(1)
```

**Ordem dos argumentos com múltiplos patches:**
```python
@patch('module.third')   # ← Último argumento
@patch('module.second')  # ← Segundo argumento
@patch('module.first')   # ← Primeiro argumento
def test_multiple(self, mock_first, mock_second, mock_third):
    # Ordem é INVERSA: de baixo para cima!
    pass
```

### B. Context Manager (`with`)

**Uso:** Ideal para mockar apenas um bloco pequeno do teste.

```python
def test_specific_block(self):
    # Código antes: mock não existe
    
    # O mock só existe dentro do 'with'
    with patch('my_service.requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        my_service.check_status()  # Usa o mock
    
    # Código depois: mock não existe mais
```

**Múltiplos context managers:**
```python
def test_multiple_contexts(self):
    with patch('module.api') as mock_api, \
         patch('module.db') as mock_db:
        
        mock_api.fetch.return_value = {"data": "test"}
        mock_db.save.return_value = True
        
        # Seu teste aqui
```

### C. Manual (Menos comum)

**Uso:** Quando precisa de controle fino sobre start/stop.

```python
def test_manual_patch(self):
    patcher = patch('my_module.function')
    mock_func = patcher.start()
    
    try:
        mock_func.return_value = 42
        # Seu teste
    finally:
        patcher.stop()  # IMPORTANTE: sempre parar!
```

---

## 🕵️ 3. Verificando Comportamento (Behavior Verification)

### Principais Asserções

| Asserção | O que verifica | Exemplo |
|----------|---------------|---------|
| `assert_called()` | Foi chamado pelo menos uma vez? | `mock.method.assert_called()` |
| `assert_called_once()` | Foi chamado exatamente 1 vez? | `mock.method.assert_called_once()` |
| `assert_called_once_with(args)` | Chamado 1 vez com estes argumentos exatos? | `mock.get.assert_called_once_with(1, 'a')` |
| `assert_called_with(args)` | A última chamada usou estes argumentos? | `mock.save.assert_called_with(data=x)` |
| `assert_not_called()` | Garante que não foi tocado | `mock.cache_miss.assert_not_called()` |
| `assert_any_call(args)` | Foi chamado com estes args em algum momento? | `mock.log.assert_any_call('error')` |
| `call_count` | Quantas vezes foi chamado? | `assert mock.fetch.call_count == 3` |
| `call_args` | Inspeciona argumentos da última chamada | `args, kwargs = mock.method.call_args` |
| `call_args_list` | Lista de todas as chamadas | `all_calls = mock.log.call_args_list` |

### Exemplos Práticos

#### Verificar quantidade de chamadas

```python
@patch('module.api_call')
def test_retry_logic(self, mock_api):
    mock_api.side_effect = [Timeout, Timeout, {"status": "ok"}]
    
    result = my_function_with_retry()
    
    # Verificar que tentou 3 vezes
    assert mock_api.call_count == 3
```

#### Verificar argumentos específicos

```python
@patch('module.logger')
def test_logging(self, mock_logger):
    process_data(user_id=123, action="login")
    
    # Verificar que foi logado corretamente
    mock_logger.info.assert_called_once_with(
        "User 123 performed action: login"
    )
```

#### Verificar múltiplas chamadas

```python
@patch('module.db.save')
def test_batch_save(self, mock_save):
    save_users([{"id": 1}, {"id": 2}, {"id": 3}])
    
    # Verificar que save foi chamado 3 vezes
    assert mock_save.call_count == 3
    
    # Verificar argumentos de cada chamada
    assert mock_save.call_args_list == [
        call({"id": 1}),
        call({"id": 2}),
        call({"id": 3}),
    ]
```

#### Verificar que NÃO foi chamado (cache hit)

```python
@patch('module.expensive_api_call')
def test_cache_works(self, mock_api):
    # Primeira chamada: deve chamar API
    get_data_cached(key="test")
    mock_api.assert_called_once()
    
    # Segunda chamada: NÃO deve chamar API (cache hit)
    get_data_cached(key="test")
    mock_api.assert_called_once()  # Ainda 1 só!
```

---

## 🚨 4. Receitas Prontas (Data Science)

### 📁 S3 / Leitura de Arquivos

```python
@patch('my_module.pd.read_csv')
def test_load_data(self, mock_read):
    # Arrange: Preparar DataFrame fake
    fake_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    mock_read.return_value = fake_df
    
    # Act: Executar função que lê CSV
    result = load_data_from_s3('s3://bucket/data.csv')
    
    # Assert: Verificar resultado e comportamento
    pd.testing.assert_frame_equal(result, fake_df)
    mock_read.assert_called_once_with('s3://bucket/data.csv')
```

**Variação: Mockar boto3 (S3 direto)**
```python
@patch('my_module.boto3.client')
def test_s3_download(self, mock_boto_client):
    mock_s3 = mock_boto_client.return_value
    mock_s3.download_file.return_value = None
    
    download_from_s3('bucket', 'key.csv', '/tmp/file.csv')
    
    mock_s3.download_file.assert_called_once_with(
        'bucket', 'key.csv', '/tmp/file.csv'
    )
```

### 🤖 Modelo de ML (Fit/Predict)

```python
@patch('my_module.RandomForestClassifier')
def test_training(self, MockModelClass):
    # Arrange: Configurar mock do modelo
    model_instance = MockModelClass.return_value
    model_instance.predict.return_value = np.array([1, 0, 1])
    model_instance.score.return_value = 0.95
    
    # Act: Executar pipeline de treino
    trainer = ModelTrainer()
    accuracy = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # Assert: Verificar que treinou e previu
    model_instance.fit.assert_called_once_with(X_train, y_train)
    model_instance.predict.assert_called_once_with(X_test)
    assert accuracy == 0.95
```

**Variação: Mockar modelo já carregado (joblib)**
```python
@patch('my_module.joblib.load')
def test_load_model(self, mock_joblib_load):
    mock_model = Mock()
    mock_model.predict.return_value = [1, 0]
    mock_joblib_load.return_value = mock_model
    
    predictor = ModelPredictor('model.pkl')
    result = predictor.predict([[1, 2, 3]])
    
    assert result == [1, 0]
    mock_joblib_load.assert_called_once_with('model.pkl')
```

### 🌐 API com Retry (Side Effect Lista)

```python
@patch('requests.get')
def test_retry_logic(self, mock_get):
    # Arrange: 1ª e 2ª chamadas falham, 3ª funciona
    mock_get.side_effect = [
        Timeout("Network error"),
        Timeout("Network error"),
        Mock(status_code=200, json=lambda: {"data": "success"})
    ]
    
    # Act: Função com retry
    result = fetch_with_retry('https://api.example.com/data', max_retries=3)
    
    # Assert: Verificar que tentou 3 vezes
    assert mock_get.call_count == 3
    assert result == {"data": "success"}
```

### 📊 Banco de Dados (SQLAlchemy)

```python
@patch('my_module.Session')
def test_db_query(self, MockSession):
    # Arrange: Configurar mock da sessão
    mock_session = MockSession.return_value
    mock_query = mock_session.query.return_value
    mock_query.filter.return_value.first.return_value = User(id=1, name="Alice")
    
    # Act: Executar query
    user = get_user_by_id(1)
    
    # Assert: Verificar resultado e chamadas
    assert user.name == "Alice"
    mock_session.query.assert_called_once_with(User)
    mock_query.filter.assert_called_once()
```

### 📈 Matplotlib/Plotly (Evitar renderização)

```python
@patch('my_module.plt.savefig')
@patch('my_module.plt.show')
def test_plot_generation(self, mock_show, mock_savefig):
    # Act: Gerar gráfico
    create_plot(data=[1, 2, 3])
    
    # Assert: Verificar que tentou salvar (sem realmente criar arquivo)
    mock_savefig.assert_called_once()
    mock_show.assert_not_called()  # Em testes, não queremos abrir janela
```

### ⏰ Datetime/Time (Congelar tempo)

```python
from unittest.mock import patch
from datetime import datetime

@patch('my_module.datetime')
def test_time_based_logic(self, mock_datetime):
    # Arrange: Fixar tempo em 2024-01-01 12:00:00
    fake_now = datetime(2024, 1, 1, 12, 0, 0)
    mock_datetime.now.return_value = fake_now
    
    # Act: Função que depende de tempo
    result = get_current_hour()
    
    # Assert: Sempre retorna 12 (tempo congelado)
    assert result == 12
```

**Alternativa melhor: use `freezegun`**
```python
from freezegun import freeze_time

@freeze_time("2024-01-01 12:00:00")
def test_time_with_freezegun(self):
    result = get_current_hour()
    assert result == 12
```

---

## 🎓 Padrões Avançados

### Mock de Classe com Instância

Quando você mocka uma **classe**, precisa configurar tanto a classe quanto suas instâncias:

```python
@patch('my_module.Database')
def test_database_operations(self, MockDatabaseClass):
    # Configurar a INSTÂNCIA (o que é criado com Database())
    mock_db_instance = MockDatabaseClass.return_value
    mock_db_instance.connect.return_value = True
    mock_db_instance.query.return_value = [{"id": 1}]
    
    # Usar no código
    db = Database()  # MockDatabaseClass() é chamado
    db.connect()     # mock_db_instance.connect() é chamado
    results = db.query("SELECT * FROM users")
    
    # Verificar
    mock_db_instance.connect.assert_called_once()
    assert results == [{"id": 1}]
```

### PropertyMock (Propriedades)

Para mockar `@property` ou atributos calculados:

```python
from unittest.mock import PropertyMock

@patch('my_module.Model')
def test_model_property(self, MockModel):
    mock_instance = MockModel.return_value
    
    # Configurar property
    type(mock_instance).is_trained = PropertyMock(return_value=True)
    
    # Usar
    model = Model()
    assert model.is_trained == True
```

### MagicMock (Operadores especiais)

Para mockar métodos mágicos (`__len__`, `__getitem__`, etc):

```python
from unittest.mock import MagicMock

mock_list = MagicMock()
mock_list.__len__.return_value = 3
mock_list.__getitem__.return_value = "item"

assert len(mock_list) == 3
assert mock_list[0] == "item"
```

---

## 💡 Dica Final de Entrevista

### Pergunta Comum: "Qual a diferença entre Mock e Stub?"

**Resposta Rápida:**

- **Stub:** "É um objeto burro que só retorna dados prontos para o teste rodar. Não verifico se foi chamado."
  ```python
  stub_api = Mock()
  stub_api.get_data.return_value = {"user": "Alice"}
  # Uso só para ter dados, não verifico comportamento
  ```

- **Mock:** "É um objeto inteligente usado para verificar comportamento. Eu pergunto pra ele: 'Você foi chamado? Com quais argumentos?'"
  ```python
  mock_api = Mock()
  mock_api.get_data.return_value = {"user": "Alice"}
  # ...
  mock_api.get_data.assert_called_once_with(user_id=123)  # ← Verificação!
  ```

**Resposta Completa (Se tiver tempo):**

| Aspecto | Stub | Mock |
|---------|------|------|
| **Propósito** | Fornecer dados para o teste | Verificar interações |
| **Verificação** | Não verifica se foi chamado | Verifica chamadas, argumentos, ordem |
| **Complexidade** | Simples, só retorna valores | Mais complexo, rastreia comportamento |
| **Foco** | State verification (resultado final) | Behavior verification (como chegou lá) |

---

## 🎯 Checklist para Testes com Mocks

Antes de considerar seu teste "completo", verifique:

- [ ] Mockei onde a função é **USADA**, não onde é **DEFINIDA**?
- [ ] Configurei `return_value` ou `side_effect` apropriadamente?
- [ ] Testei o **estado** (resultado final está correto)?
- [ ] Testei o **comportamento** (métodos foram chamados corretamente)?
- [ ] Usei `assert_called_once_with()` para verificar argumentos?
- [ ] Considerei usar `autospec=True` para validação de API?
- [ ] Meu teste é rápido (< 1 segundo)?
- [ ] Meu teste é determinístico (sempre passa ou sempre falha)?

---

## 📚 Referências Rápidas

### Imports Essenciais

```python
from unittest.mock import Mock, MagicMock, patch, PropertyMock, call
import pytest
```

### Comandos Úteis

```bash
# Rodar testes com verbose
pytest -v

# Rodar teste específico
pytest tests/test_file.py::test_function -v

# Ver cobertura de mocks
pytest --cov=my_module tests/
```

---

## 🚀 Próximos Passos

1. **Praticar:** Faça os katas do projeto
2. **Refatorar:** Pegue código legado e isole I/O
3. **Medir:** Use coverage para ver o que está testado
4. **Compartilhar:** Ensine mocking para seu time

**Happy Mocking! 🎭**
