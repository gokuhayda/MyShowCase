# 🏗️ Arquitetura do Sistema

## Visão Geral

Este projeto implementa o **Trainer Pattern** para desacoplar a lógica de treino de Deep Learning das responsabilidades auxiliares (logging, checkpointing, early stopping).

## Decisões de Design

### 1. Por que Protocol ao invés de ABC?

**Decisão:** Usar `Protocol` (PEP 544) ao invés de `abc.ABC`.

**Razão:**
- ✅ Duck typing nativo do Python
- ✅ Não força herança (composição > herança)
- ✅ Mais flexível para testes (mocks simples)
```python
# ❌ Com ABC (força herança)
class MyCallback(Callback):
    def on_epoch_end(self, ...): ...

# ✅ Com Protocol (duck typing)
class MyCallback:  # Não precisa herdar!
    def on_epoch_end(self, ...): ...
```

### 2. Por que retornar bool em on_epoch_end?

**Decisão:** `on_epoch_end` retorna `bool` indicando se deve parar.

**Razão:**
- ✅ Simples e explícito
- ✅ Evita efeitos colaterais ocultos (callbacks não modificam Trainer)
- ✅ Segue convenção de Keras/PyTorch Lightning

**Alternativas consideradas:**
- ❌ Callback modificar `trainer.stop_training` diretamente (tight coupling)
- ❌ Lançar exceção `StopTraining` (exceptions for control flow)

### 3. Por que não usar herança para Trainer?

**Decisão:** Trainer é uma classe concreta, não abstrata.

**Razão:**
- ✅ Composição via callbacks é mais flexível
- ✅ Evita "explosion" de subclasses (TrainerWithEarlyStopping, TrainerWithCheckpoint...)
- ✅ Open/Closed Principle: estender via callbacks, não herança

## Fluxo de Execução
```
┌─────────────────────────────────────┐
│  trainer.fit(dataloader, epochs=10) │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  _notify_callbacks("on_train_begin")│
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │  Loop Épocas  │
       └───────┬───────┘
               │
     ┌─────────▼──────────┐
     │  _train_one_epoch() │
     └─────────┬───────────┘
               │
               ▼
   ┌────────────────────────────┐
   │ _notify_callbacks(         │
   │   "on_epoch_end",          │
   │   epoch, logs              │
   │ )                          │
   └────────────┬───────────────┘
                │
                ▼
         ┌──────────────┐
         │ Verificar se │
         │ deve parar   │
         └──────────────┘
```

## Testabilidade

### Princípio: Isolar Responsabilidades

Cada callback pode ser testado **sem rodar rede neural**:
```python
# Testar early stopping sem GPU, sem dados, sem modelo!
def test_early_stopping():
    callback = EarlyStopping(patience=2)
    
    # Simular épocas ruins
    callback.on_epoch_end(0, {"loss": 1.0})
    should_stop = callback.on_epoch_end(1, {"loss": 1.0})
    
    assert should_stop is True
```

### Benefícios:
- ⚡ Testes rápidos (milissegundos vs. minutos)
- 🔬 Isolamento perfeito (bug em checkpoint não afeta early stopping)
- 📊 Coverage alto (testar todos edge cases é viável)

## Extensibilidade

### Adicionar novo callback: 3 passos

1. **Criar classe com métodos do Protocol:**
```python
class WandbLogger:
    def on_train_begin(self, logs):
        wandb.init(project="my-project")
    
    def on_epoch_end(self, epoch, logs):
        wandb.log(logs)
        return False
```

2. **Não modificar Trainer** (Open/Closed Principle)

3. **Usar via composição:**
```python
trainer = Trainer(callbacks=[WandbLogger()])
```

## Comparação com Frameworks Reais

| Aspecto | Este Projeto | Keras | PyTorch Lightning |
|---------|-------------|-------|-------------------|
| Interface | Protocol | ABC | ABC |
| Hooks | 2 (begin, epoch_end) | 7+ | 20+ |
| Complexidade | Educacional | Produção | Produção |
| Propósito | Demonstrar padrão | Framework completo | Framework completo |

## Trade-offs

### ✅ Vantagens desta abordagem:
- Simples de entender
- Fácil de testar
- Extensível sem modificação

### ⚠️ Limitações (por ser educacional):
- Não cobre batch-level hooks
- Não suporta multi-GPU
- Não tem logger integrado (W&B, MLflow)

Para produção, use PyTorch Lightning ou Keras diretamente.

## Referências

- [Keras Callbacks Guide](https://keras.io/guides/writing_your_own_callbacks/)
- [PyTorch Lightning Callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html)
- [Design Patterns: Elements of Reusable OO Software](https://en.wikipedia.org/wiki/Design_Patterns) (Gang of Four)
