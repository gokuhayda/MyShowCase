# 🎭 Mock de Classe vs Mock de Instância: O Guia Definitivo

## 📚 Índice

1. [O Problema Fundamental](#o-problema-fundamental)
2. [Classe vs Instância: Revisão Rápida](#classe-vs-instância-revisão-rápida)
3. [Como o @patch Funciona](#como-o-patch-funciona)
4. [A Mágica do return_value](#a-mágica-do-return_value)
5. [Exemplo Completo Passo a Passo](#exemplo-completo-passo-a-passo)
6. [Visualização Gráfica](#visualização-gráfica)
7. [Armadilhas Comuns](#armadilhas-comuns)
8. [Padrões e Boas Práticas](#padrões-e-boas-práticas)
9. [Exercícios Práticos](#exercícios-práticos)

---

## 🎯 O Problema Fundamental

Quando testamos código que usa bibliotecas externas (como scikit-learn), enfrentamos um dilema:

```python
# Código de Produção
class ModelTrainer:
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        model = RandomForestClassifier(n_estimators=100)  # ← Demora 30 minutos!
        model.fit(X_train, y_train)                       # ← Muito lento!
        predictions = model.predict(X_test)               # ← Pesado!
        return accuracy_score(y_test, predictions)
```

**Queremos testar a LÓGICA (o fluxo), não a MATEMÁTICA (o algoritmo).**

**Solução:** Substituir `RandomForestClassifier` por um "objeto falso" (Mock) que responde instantaneamente.

---

## 🏗️ Classe vs Instância: Revisão Rápida

Antes de mergulhar em mocks, vamos relembrar os conceitos básicos:

```python
# CLASSE: A "fábrica" de objetos (o molde)
class Carro:
    def __init__(self, cor):
        self.cor = cor
    
    def buzinar(self):
        return "Beep!"

# INSTÂNCIA: Um objeto criado a partir da classe
meu_carro = Carro("vermelho")  # ← Chamando a CLASSE (construtor)
som = meu_carro.buzinar()       # ← Chamando um MÉTODO da INSTÂNCIA
```

**Analogia:**
- **Classe** = Planta arquitetônica de uma casa
- **Instância** = A casa física construída a partir da planta

---

## 🔧 Como o @patch Funciona

O decorator `@patch` **substitui** a classe no módulo onde ela é **usada** (não onde é definida):

```python
from unittest.mock import patch

# ❌ ERRADO: Mockar onde a classe foi definida
@patch('sklearn.ensemble.RandomForestClassifier')

# ✅ CORRETO: Mockar onde a classe é USADA
@patch('katas.b02_ml_pipeline.model_trainer.RandomForestClassifier')
def test_algo(self, mock_rf_class):
    #                  ^^^^^^^^^^^^^
    #                  Este argumento É A CLASSE mockada
    pass
```

### O Que Realmente Acontece?

```python
# Sem mock (código real):
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()  # ← Cria um objeto REAL

# Com mock (no teste):
@patch('katas.b02_ml_pipeline.model_trainer.RandomForestClassifier')
def test_...(self, mock_rf_class):
    # Agora 'RandomForestClassifier' no código de produção 
    # aponta para 'mock_rf_class' (um MagicMock)
    
    # Quando o código fizer:
    model = RandomForestClassifier()  # ← Retorna mock_rf_class.return_value
```

---

## ✨ A Mágica do `return_value`

### Conceito Central

**Tudo que é "chamável" (callable) em Python tem um `.return_value` no mock.**

```python
# Classes são "chamáveis" (você chama o construtor)
obj = MinhaClasse()  # ← Isso é uma CHAMADA

# Métodos também são "chamáveis"
resultado = obj.meu_metodo()  # ← Isso é uma CHAMADA
```

### Dois Níveis de `return_value`

```python
mock_rf_class.return_value              # ← O que a CLASSE retorna (a instância)
              ^^^^^^^^^^^^
              
mock_instance.predict.return_value      # ← O que o MÉTODO retorna (o resultado)
                      ^^^^^^^^^^^^
```

### Exemplo Visual

```python
@patch('katas.b02_ml_pipeline.model_trainer.RandomForestClassifier')
def test_...(self, mock_rf_class):
    
    # NÍVEL 1: Configurar o que a CLASSE retorna quando é chamada
    mock_instance = mock_rf_class.return_value
    #               ^^^^^^^^^^^^^
    #               Classe mockada (construtor falso)
    #                             ^^^^^^^^^^^^
    #                             "O objeto que será criado"
    
    # NÍVEL 2: Configurar o que um MÉTODO da instância retorna
    mock_instance.predict.return_value = np.array([1, 0])
    #             ^^^^^^^
    #             Método da instância
    #                     ^^^^^^^^^^^^
    #                     "O resultado quando predict() for chamado"
```

---

## 🎬 Exemplo Completo Passo a Passo

Vamos seguir o fluxo linha por linha:

### 1️⃣ Código de Produção

```python
# katas/b02_ml_pipeline/model_trainer.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        # Linha A: Instanciar
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Linha B: Treinar
        model.fit(X_train, y_train)
        
        # Linha C: Prever
        predictions = model.predict(X_test)
        
        # Linha D: Avaliar
        accuracy = accuracy_score(y_test, predictions)
        return accuracy
```

### 2️⃣ Código do Teste

```python
# tests/test_02_ml_pipeline.py
import unittest
from unittest.mock import patch
import numpy as np
from katas.b02_ml_pipeline.model_trainer import ModelTrainer

class TestMLPipeline(unittest.TestCase):
    
    @patch('katas.b02_ml_pipeline.model_trainer.RandomForestClassifier')
    def test_train_and_evaluate_flow(self, mock_rf_class):
        
        # --- ARRANGE (Preparar) ---
        
        # Passo 1: Obter referência ao objeto que será criado
        mock_instance = mock_rf_class.return_value
        
        # Passo 2: Configurar o comportamento do método .predict()
        mock_instance.predict.return_value = np.array([1, 0])
        
        # Passo 3: Preparar dados de teste
        trainer = ModelTrainer()
        X_train = np.array([[1, 1], [2, 2]])
        y_train = np.array([1, 0])
        X_test  = np.array([[3, 3], [4, 4]])
        y_test  = np.array([1, 0])
        
        # --- ACT (Executar) ---
        accuracy = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        # --- ASSERT (Verificar) ---
        
        # Verificação 1: O construtor foi chamado corretamente?
        mock_rf_class.assert_called_once_with(n_estimators=100, random_state=42)
        
        # Verificação 2: O método fit() foi chamado?
        mock_instance.fit.assert_called_once_with(X_train, y_train)
        
        # Verificação 3: O método predict() foi chamado?
        mock_instance.predict.assert_called_once_with(X_test)
        
        # Verificação 4: A acurácia foi calculada corretamente?
        self.assertEqual(accuracy, 1.0)
```

### 3️⃣ O Fluxo Durante a Execução

```python
# Quando o teste roda...

# 1. O @patch substitui RandomForestClassifier por mock_rf_class

# 2. Quando o código de produção executa a Linha A:
model = RandomForestClassifier(n_estimators=100, random_state=42)
#       ^^^^^^^^^^^^^^^^^^^^^^^
#       Isso agora chama mock_rf_class (não a classe real!)
#       
#       O que acontece?
#       model = mock_rf_class(n_estimators=100, random_state=42)
#       model = mock_rf_class.return_value  # ← Retorna mock_instance
#       
#       Ou seja: model = mock_instance

# 3. Quando o código de produção executa a Linha B:
model.fit(X_train, y_train)
# É o mesmo que:
mock_instance.fit(X_train, y_train)
# Como é um mock, NÃO treina nada (instantâneo!)
# Mas REGISTRA que foi chamado (para verificações posteriores)

# 4. Quando o código de produção executa a Linha C:
predictions = model.predict(X_test)
# É o mesmo que:
predictions = mock_instance.predict(X_test)
# Retorna o que configuramos:
predictions = np.array([1, 0])

# 5. Quando o código de produção executa a Linha D:
accuracy = accuracy_score(y_test, predictions)
#                                 ^^^^^^^^^^^
#                                 np.array([1, 0])
# Como y_test = [1, 0] e predictions = [1, 0]
# accuracy = 1.0 (100% de acerto!)
```

---

## 📊 Visualização Gráfica

### Diagrama do Fluxo

```
┌─────────────────────────────────────────────────────────────────┐
│                         NO TESTE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  @patch('...RandomForestClassifier')                             │
│  def test_...(self, mock_rf_class):                              │
│         │                                                         │
│         │  mock_instance = mock_rf_class.return_value            │
│         │                                                         │
│         │  mock_instance.predict.return_value = [1, 0]           │
│         │                                                         │
│         ▼                                                         │
│  trainer.train_and_evaluate(...)                                 │
│         │                                                         │
└─────────┼─────────────────────────────────────────────────────────┘
          │
          │ Chama o código de produção
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   NO CÓDIGO DE PRODUÇÃO                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  model = RandomForestClassifier(n_estimators=100)                │
│           └──────┬──────────┘                                    │
│                  │                                                │
│                  │ O @patch intercepta!                          │
│                  ▼                                                │
│  model = mock_rf_class.return_value                              │
│  model = mock_instance  ◄────────────── Configurado no teste    │
│           │                                                       │
│           │                                                       │
│  model.fit(X_train, y_train)                                     │
│  mock_instance.fit(X_train, y_train)  ◄─ Não faz nada (mock!)   │
│           │                                                       │
│           │                                                       │
│  predictions = model.predict(X_test)                             │
│  predictions = mock_instance.predict(X_test)                     │
│  predictions = [1, 0]  ◄────────────── Retorna o que definimos  │
│           │                                                       │
│           │                                                       │
│  accuracy = accuracy_score(y_test, predictions)                  │
│  accuracy = 1.0  ◄────────────────────── Cálculo real           │
│           │                                                       │
│           ▼                                                       │
│  return accuracy                                                 │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Tabela Comparativa

| Conceito | No Código Real | No Mock | Configurado Via |
|----------|---------------|---------|-----------------|
| **Classe (Construtor)** | `RandomForestClassifier` | `mock_rf_class` | `@patch(...)` |
| **Instância (Objeto)** | `model` | `mock_instance` | `mock_rf_class.return_value` |
| **Método .fit()** | Treina o modelo (lento!) | Não faz nada (instantâneo) | Automático (MagicMock) |
| **Método .predict()** | Calcula predições (lento!) | Retorna valor fake | `mock_instance.predict.return_value` |
| **Resultado** | Predições reais | `np.array([1, 0])` | Definido no teste |

---

## ⚠️ Armadilhas Comuns

### Armadilha 1: Mockar no Lugar Errado

```python
# ❌ ERRADO
@patch('sklearn.ensemble.RandomForestClassifier')
def test_algo(self, mock_rf):
    # Não funciona! O mock foi aplicado no módulo sklearn,
    # mas o código importou para outro lugar!
    pass

# ✅ CORRETO
@patch('katas.b02_ml_pipeline.model_trainer.RandomForestClassifier')
def test_algo(self, mock_rf):
    # Funciona! Mock aplicado onde a classe é USADA
    pass
```

**Regra de Ouro:** Mocka onde é USADO, não onde é DEFINIDO.

---

### Armadilha 2: Esquecer de Configurar o return_value

```python
@patch('katas.b02_ml_pipeline.model_trainer.RandomForestClassifier')
def test_algo(self, mock_rf_class):
    # ❌ ERRADO: Não configurou o return_value do predict
    trainer = ModelTrainer()
    
    # O código chama model.predict(X_test)
    # Mas não configuramos o que deve retornar!
    # Resultado: mock_rf_class.return_value.predict retorna outro MagicMock
    # Isso pode causar erros ou comportamentos estranhos!
```

```python
@patch('katas.b02_ml_pipeline.model_trainer.RandomForestClassifier')
def test_algo(self, mock_rf_class):
    # ✅ CORRETO: Configurou o comportamento esperado
    mock_rf_class.return_value.predict.return_value = np.array([1, 0])
    
    trainer = ModelTrainer()
    accuracy = trainer.train_and_evaluate(...)
    # Agora o teste sabe exatamente o que esperar!
```

---

### Armadilha 3: Confundir Classe com Instância nas Verificações

```python
@patch('katas.b02_ml_pipeline.model_trainer.RandomForestClassifier')
def test_algo(self, mock_rf_class):
    mock_instance = mock_rf_class.return_value
    mock_instance.predict.return_value = np.array([1, 0])
    
    trainer = ModelTrainer()
    trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # ❌ ERRADO: Verificar na classe
    mock_rf_class.fit.assert_called_once()  # Não funciona!
    
    # ✅ CORRETO: Verificar na instância
    mock_instance.fit.assert_called_once_with(X_train, y_train)
```

**Por quê?** 
- `mock_rf_class` = A classe (chamada 1 vez: no construtor)
- `mock_instance` = O objeto criado (onde os métodos são chamados)

---

### Armadilha 4: Não Verificar os Parâmetros do Construtor

```python
@patch('katas.b02_ml_pipeline.model_trainer.RandomForestClassifier')
def test_algo(self, mock_rf_class):
    mock_rf_class.return_value.predict.return_value = np.array([1, 0])
    
    trainer = ModelTrainer()
    accuracy = trainer.train_and_evaluate(...)
    
    # ❌ INCOMPLETO: Não verifica se o modelo foi criado corretamente
    # E se alguém mudar de n_estimators=100 para n_estimators=10?
    # O teste continua passando, mas o comportamento mudou!
```

```python
@patch('katas.b02_ml_pipeline.model_trainer.RandomForestClassifier')
def test_algo(self, mock_rf_class):
    mock_rf_class.return_value.predict.return_value = np.array([1, 0])
    
    trainer = ModelTrainer()
    accuracy = trainer.train_and_evaluate(...)
    
    # ✅ COMPLETO: Verifica todos os comportamentos críticos
    mock_rf_class.assert_called_once_with(n_estimators=100, random_state=42)
    mock_rf_class.return_value.fit.assert_called_once()
    mock_rf_class.return_value.predict.assert_called_once()
```

---

## 🏆 Padrões e Boas Práticas

### Padrão 1: Variável Explícita para a Instância

```python
# ✅ BOM: Mais legível
@patch('...RandomForestClassifier')
def test_algo(self, mock_rf_class):
    mock_instance = mock_rf_class.return_value
    mock_instance.predict.return_value = np.array([1, 0])
    
    # Código do teste...
    
    mock_instance.fit.assert_called_once()
```

```python
# ⚠️ FUNCIONA MAS MENOS LEGÍVEL: Acesso direto
@patch('...RandomForestClassifier')
def test_algo(self, mock_rf_class):
    mock_rf_class.return_value.predict.return_value = np.array([1, 0])
    
    # Código do teste...
    
    mock_rf_class.return_value.fit.assert_called_once()
```

**Recomendação:** Use variável explícita (`mock_instance`) para clareza.

---

### Padrão 2: Estrutura AAA (Arrange-Act-Assert)

```python
@patch('...RandomForestClassifier')
def test_algo(self, mock_rf_class):
    # --- ARRANGE (Preparar) ---
    mock_instance = mock_rf_class.return_value
    mock_instance.predict.return_value = np.array([1, 0])
    
    trainer = ModelTrainer()
    X_train, y_train = ..., ...
    X_test, y_test = ..., ...
    
    # --- ACT (Executar) ---
    accuracy = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # --- ASSERT (Verificar) ---
    mock_rf_class.assert_called_once_with(n_estimators=100, random_state=42)
    mock_instance.fit.assert_called_once_with(X_train, y_train)
    mock_instance.predict.assert_called_once_with(X_test)
    self.assertEqual(accuracy, 1.0)
```

---

### Padrão 3: Verificações Completas

Um bom teste verifica **3 camadas**:

```python
# Camada 1: CONSTRUÇÃO (a classe foi instanciada corretamente?)
mock_rf_class.assert_called_once_with(n_estimators=100, random_state=42)

# Camada 2: COMPORTAMENTO (os métodos foram chamados?)
mock_instance.fit.assert_called_once_with(X_train, y_train)
mock_instance.predict.assert_called_once_with(X_test)

# Camada 3: RESULTADO (a lógica produziu o resultado esperado?)
self.assertEqual(accuracy, 1.0)
```

---

### Padrão 4: Nomenclatura Clara

```python
# ✅ BOM: Nomes descritivos
@patch('...RandomForestClassifier')
def test_algo(self, mock_rf_class):
    mock_model_instance = mock_rf_class.return_value

# ⚠️ FUNCIONA MAS CONFUSO: Nomes genéricos
@patch('...RandomForestClassifier')
def test_algo(self, mock_class):
    mock_obj = mock_class.return_value
```

**Convenções sugeridas:**
- `mock_<NomeDaClasse>_class` para a classe mockada
- `mock_<nomeDaVariavel>_instance` ou `mock_instance` para o objeto

---

## 🎓 Exercícios Práticos

### Exercício 1: Mock Simples

**Cenário:** Tens uma classe `EmailSender` que usa `smtplib.SMTP`.

```python
# Código de Produção
import smtplib

class EmailSender:
    def send_email(self, recipient, subject, body):
        smtp = smtplib.SMTP('smtp.gmail.com', 587)
        smtp.starttls()
        smtp.login('user@example.com', 'password')
        smtp.sendmail('user@example.com', recipient, f"Subject: {subject}\n\n{body}")
        smtp.quit()
        return True
```

**Desafio:** Escreve um teste que verifica se todos os métodos do SMTP foram chamados corretamente, **sem enviar emails reais**.

<details>
<summary>💡 Solução</summary>

```python
from unittest.mock import patch

class TestEmailSender(unittest.TestCase):
    
    @patch('email_sender.smtplib.SMTP')
    def test_send_email_flow(self, mock_smtp_class):
        # Arrange
        mock_smtp_instance = mock_smtp_class.return_value
        
        sender = EmailSender()
        recipient = 'test@example.com'
        subject = 'Test Subject'
        body = 'Test Body'
        
        # Act
        result = sender.send_email(recipient, subject, body)
        
        # Assert
        mock_smtp_class.assert_called_once_with('smtp.gmail.com', 587)
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with('user@example.com', 'password')
        mock_smtp_instance.sendmail.assert_called_once()
        mock_smtp_instance.quit.assert_called_once()
        self.assertTrue(result)
```

</details>

---

### Exercício 2: Mock com Múltiplas Instâncias

**Cenário:** Tens código que cria **duas** instâncias da mesma classe.

```python
# Código de Produção
class DataProcessor:
    def process_with_two_models(self, data):
        model1 = RandomForestClassifier(n_estimators=50)
        model2 = RandomForestClassifier(n_estimators=100)
        
        pred1 = model1.predict(data)
        pred2 = model2.predict(data)
        
        return (pred1 + pred2) / 2
```

**Desafio:** Como mockar duas instâncias diferentes?

<details>
<summary>💡 Solução</summary>

```python
from unittest.mock import patch, MagicMock

class TestDataProcessor(unittest.TestCase):
    
    @patch('data_processor.RandomForestClassifier')
    def test_process_with_two_models(self, mock_rf_class):
        # Arrange: Criar dois mocks diferentes
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        
        # Configurar side_effect para retornar mocks diferentes
        mock_rf_class.side_effect = [mock_model1, mock_model2]
        
        # Configurar retornos
        mock_model1.predict.return_value = np.array([1, 2, 3])
        mock_model2.predict.return_value = np.array([3, 4, 5])
        
        processor = DataProcessor()
        data = np.array([[1, 2], [3, 4], [5, 6]])
        
        # Act
        result = processor.process_with_two_models(data)
        
        # Assert
        self.assertEqual(mock_rf_class.call_count, 2)
        mock_model1.predict.assert_called_once_with(data)
        mock_model2.predict.assert_called_once_with(data)
        np.testing.assert_array_equal(result, np.array([2, 3, 4]))
```

</details>

---

### Exercício 3: Mock com Context Manager

**Cenário:** Tens código que usa context managers (`with`).

```python
# Código de Produção
class FileProcessor:
    def read_and_process(self, filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        return content.upper()
```

**Desafio:** Como mockar `open()` que é usado como context manager?

<details>
<summary>💡 Solução</summary>

```python
from unittest.mock import patch, mock_open

class TestFileProcessor(unittest.TestCase):
    
    @patch('builtins.open', new_callable=mock_open, read_data='hello world')
    def test_read_and_process(self, mock_file):
        # Arrange
        processor = FileProcessor()
        
        # Act
        result = processor.read_and_process('test.txt')
        
        # Assert
        mock_file.assert_called_once_with('test.txt', 'r')
        self.assertEqual(result, 'HELLO WORLD')
```

**Explicação:** `mock_open` é um helper especial para mockar arquivos.

</details>

---

## 📖 Glossário

| Termo | Definição | Exemplo |
|-------|-----------|---------|
| **Classe** | Molde/template para criar objetos | `RandomForestClassifier` |
| **Instância** | Objeto criado a partir de uma classe | `model = RandomForestClassifier()` |
| **Construtor** | Método especial que cria instâncias | `__init__` ou chamar a classe como função |
| **Mock** | Objeto falso que simula comportamento real | `MagicMock()` |
| **@patch** | Decorator que substitui objetos por mocks | `@patch('module.Class')` |
| **return_value** | O que um mock retorna quando é chamado | `mock.return_value = 42` |
| **side_effect** | Comportamento customizado (exceções, múltiplos retornos) | `mock.side_effect = [1, 2, 3]` |
| **assert_called** | Verifica se um mock foi chamado | `mock.assert_called_once()` |

---

## 🎁 Dica Bônus: Debugging de Mocks

Se o teu teste não funciona, adiciona estes prints para ver o que está acontecendo:

```python
@patch('...RandomForestClassifier')
def test_algo(self, mock_rf_class):
    mock_instance = mock_rf_class.return_value
    mock_instance.predict.return_value = np.array([1, 0])
    
    # ... código do teste ...
    
    # DEBUG: Ver todas as chamadas
    print("\n=== DEBUG MOCKS ===")
    print(f"Classe foi chamada? {mock_rf_class.called}")
    print(f"Quantas vezes? {mock_rf_class.call_count}")
    print(f"Com quais argumentos? {mock_rf_class.call_args}")
    print(f"\nInstância.fit foi chamado? {mock_instance.fit.called}")
    print(f"Com quais argumentos? {mock_instance.fit.call_args}")
    print(f"\nInstância.predict foi chamado? {mock_instance.predict.called}")
    print(f"Retornou o quê? {mock_instance.predict.return_value}")
```

---

## 📚 Recursos Adicionais

- [Documentação oficial do unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Real Python: Understanding the Python Mock Object Library](https://realpython.com/python-mock-library/)
- [Python Testing with pytest (livro)](https://pragprog.com/titles/bopytest/python-testing-with-pytest/)

---

## ✅ Checklist de Revisão

Antes de finalizar um teste com mocks, verifica:

- [ ] Mockei no lugar certo (onde é USADO, não onde é DEFINIDO)?
- [ ] Configurei o `return_value` para todos os métodos que serão chamados?
- [ ] Criei uma variável explícita para a instância (`mock_instance`)?
- [ ] Verifiquei que o construtor foi chamado com os parâmetros corretos?
- [ ] Verifiquei que os métodos foram chamados na ordem/frequência esperada?
- [ ] Verifiquei o resultado final da função testada?
- [ ] Meu teste está claro e bem documentado (estrutura AAA)?

---

## 🎯 Resumo Final

```python
# O Padrão Completo em 10 Linhas

@patch('modulo_onde_eh_usado.NomeDaClasse')
def test_algo(self, mock_class):
    # 1. Obter referência à instância
    mock_instance = mock_class.return_value
    
    # 2. Configurar comportamentos
    mock_instance.metodo.return_value = valor_esperado
    
    # 3. Executar código
    resultado = codigo_de_producao()
    
    # 4. Verificar construtor, métodos e resultado
    mock_class.assert_called_once_with(parametros)
    mock_instance.metodo.assert_called_once()
    self.assertEqual(resultado, valor_esperado)
```

**Lembra sempre:**
- `mock_class` = A **CLASSE** mockada (o construtor)
- `mock_class.return_value` = A **INSTÂNCIA** que será criada
- `mock_instance.metodo.return_value` = O **RESULTADO** do método

---

**Autor:** Eric | **Data:** 2024  
**Licença:** MIT (use à vontade!)

---

🎓 **Próximo Passo:** Pratica com os exercícios e depois aplica no teu projeto real!
