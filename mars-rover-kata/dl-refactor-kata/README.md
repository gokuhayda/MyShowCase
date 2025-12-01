# 🔥 Deep Learning Trainer Refactoring Kata

[![Tests](https://github.com/seu-usuario/dl-refactor-kata/actions/workflows/tests.yml/badge.svg)](https://github.com/seu-usuario/dl-refactor-kata/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Demonstração de **Software Engineering aplicado a Deep Learning**: como transformar código espaguete de treinamento em sistema extensível e testável usando **Callback Pattern**.

---

## 🎯 Objetivo

Este projeto demonstra a aplicação de princípios SOLID e Design Patterns em código de Machine Learning, especificamente:

- ✅ **Strategy Pattern** via Callbacks
- ✅ **Open/Closed Principle** (extensível sem modificar core)
- ✅ **Dependency Inversion** (abstrações antes de implementações)
- ✅ **Testabilidade** (callbacks isolados do loop de treino)

---

## 📖 O Problema

### ❌ Código Espaguete (Antes)
```python
# Típico código de Kaggle/Research
for epoch in range(100):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

    # Mistura I/O, logging e lógica de parada
    if epoch % 10 == 0:
        print(f"Epoch {epoch} loss={loss}")
        torch.save(model, "ckpt.pth")
    if loss < 0.01:
        break  # Early stopping hardcoded
```

**Problemas:**
- 🚫 Não testável (precisa rodar rede inteira)
- 🚫 Não extensível (adicionar Slack notification = mexer no loop)
- 🚫 Responsabilidades misturadas (matemática + I/O + controle)

---

### ✅ Solução com Trainer Pattern (Depois)
```python
trainer = Trainer(
    model=MyModel(),
    optimizer=Adam(),
    loss_fn=MSELoss(),
    callbacks=[
        EarlyStopping(patience=5),
        ModelCheckpoint(filepath="best_model.pth"),
        TensorBoardLogger(),
        # Quer Slack? Só adicionar: SlackNotifier()
    ]
)
trainer.fit(train_loader, epochs=100)
```

**Benefícios:**
- ✅ Loop matemático limpo e puro
- ✅ Extensível via composição (não herança)
- ✅ Cada callback testável isoladamente
- ✅ Segue Open/Closed Principle (SOLID)

---

## 🚀 Instalação

### Opção 1: Poetry (recomendado)
```bash
git clone https://github.com/seu-usuario/dl-refactor-kata.git
cd dl-refactor-kata
poetry install
poetry shell
```

### Opção 2: pip
```bash
git clone https://github.com/seu-usuario/dl-refactor-kata.git
cd dl-refactor-kata
pip install -e .
```

---

## 💻 Uso

### Exemplo Básico
```python
from dl_trainer import Trainer
from dl_trainer.callbacks import EarlyStopping, ModelCheckpoint

# Configuração via composição
trainer = Trainer(
    model="SimulatedModel",
    optimizer="Adam",
    loss_fn="MSE",
    callbacks=[
        EarlyStopping(patience=3),
        ModelCheckpoint(filepath="model.pth")
    ]
)

# Dados simulados (para demo sem PyTorch)
fake_dataloader = [1, 2, 3]

# Rodar treino
trainer.fit(fake_dataloader, epochs=10)
```

**Saída esperada:**

```
🚀 Training started with 2 callbacks
Epoch 0 | Loss: 0.30
💾 Checkpoint: Saving model to model.pth
Epoch 1 | Loss: 0.30
🛑 Early stopping triggered at epoch 1 (patience: 3)
```

### Exemplo Avançado (Múltiplos Callbacks)

Ver `examples/advanced_usage.py` para:
- Custom metrics logging
- Learning rate scheduling
- Gradient clipping
- Slack notifications

---

## 🧪 Testes

Executar todos os testes:
```bash
pytest
```

Com coverage:
```bash
pytest --cov=src/dl_trainer --cov-report=html
```

Watch mode (rodar a cada mudança):
```bash
pytest-watch
```

---

## 📚 Arquitetura

### Diagrama de Classes

```
┌─────────────────┐
│    Trainer      │
│─────────────────│
│ + fit()         │───────┐
│ + _notify()     │       │
└─────────────────┘       │
│ usa                    │
▼                        │
┌─────────────────┐      │
│ <<Protocol>>    │      │
│   Callback      │      │
│─────────────────│      │
│ + on_train_begin│      │
│ + on_epoch_end  │      │
└─────────────────┘      │
▲                        │
│ implementa             │
┌────────────────┼────────────────┐
│                │                │
┌────────┴──────┐  ┌──────┴──────┐  ┌─────┴────────┐
│EarlyStopping  │  │ModelCheckpoint│ │CustomCallback│
└───────────────┘  └──────────────┘  └──────────────┘
```

### Design Patterns Aplicados

1. **Strategy Pattern**: Cada callback é uma estratégia intercambiável
2. **Observer Pattern**: Trainer notifica eventos para callbacks
3. **Template Method**: `fit()` define estrutura, callbacks customizam comportamento
4. **Dependency Inversion**: Trainer depende da abstração `Callback`, não de implementações concretas

Ver `docs/PATTERNS.md` para detalhes.

---

## 🎓 Conceitos Demonstrados

### 1. Open/Closed Principle (SOLID)
```python
# Aberto para extensão (adicionar SlackCallback)
# Fechado para modificação (Trainer não muda)
class SlackNotifier(Callback):
    def on_train_begin(self, logs):
        send_slack("Treino iniciado!")
    def on_epoch_end(self, epoch, logs):
        send_slack(f"Época {epoch} concluída")
        return False

# Uso sem mudar Trainer
trainer = Trainer(
    callbacks=[SlackNotifier()]  # Só adicionar!
)
```

### 2. Single Responsibility Principle

Cada classe tem **uma** responsabilidade:
- `Trainer`: Executar loop de treino
- `EarlyStopping`: Decidir quando parar
- `ModelCheckpoint`: Salvar modelo

### 3. Testabilidade
```python
# Testar early stopping SEM treinar rede neural!
def test_early_stopping():
    callback = EarlyStopping(patience=2)
    # Simular épocas ruins
    callback.on_epoch_end(0, {"loss": 1.0})
    should_stop = callback.on_epoch_end(1, {"loss": 1.0})
    assert should_stop is True
```

---

## 🔧 Extensões Possíveis

Exemplos de callbacks que você pode adicionar:

- **LearningRateScheduler**: Ajustar LR dinamicamente
- **GradientClipper**: Limitar gradientes
- **MetricsLogger**: Log em W&B, MLflow, TensorBoard
- **ProgressBar**: UI com tqdm
- **EmailNotifier**: Avisar quando treino terminar
- **ProfilerCallback**: Detectar bottlenecks

Todos seguem a mesma interface `Callback`.

---

## 📖 Referências

Este padrão é inspirado em:
- **Keras**: `model.fit(callbacks=[...])`
- **PyTorch Lightning**: `Trainer(callbacks=[...])`
- **FastAI**: `Learner.fit(..., cbs=[...])`

---

## 🤝 Contribuindo

1. Fork o projeto
2. Crie branch: `git checkout -b feature/novo-callback`
3. Commit: `git commit -m 'Add: GradientClipper callback'`
4. Push: `git push origin feature/novo-callback`
5. Abra Pull Request

---

## 📝 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

---

## 👤 Autor

**[Seu Nome]**
- GitHub: [@seu-usuario](https://github.com/seu-usuario)
- LinkedIn: [seu-perfil](https://linkedin.com/in/seu-perfil)

---

## 🎯 ThoughtWorks Context

Este projeto foi criado como parte da preparação para entrevistas técnicas onde:
- ✅ Clean Code e SOLID são esperados
- ✅ Cientistas de dados devem pensar como engenheiros
- ✅ Código de ML deve ser testável e manutenível

> "Move code out of notebooks into Python modules as early as possible. That way, they can rest within the safe confines of unit tests and domain boundaries." - ThoughtWorks Blog

---

**⭐ Se este projeto te ajudou, considere dar uma estrela!**

