# 🎨 Design Patterns Aplicados

## 1. Strategy Pattern

**Definição:** Encapsular algoritmos intercambiáveis em objetos separados.

**Aplicação:** Cada callback é uma **estratégia** para lidar com eventos do treino.
```python
# Estratégia 1: Parar quando loss não melhora
strategy1 = EarlyStopping(patience=3)

# Estratégia 2: Salvar modelo periodicamente
strategy2 = ModelCheckpoint()

# Composição de estratégias
trainer = Trainer(callbacks=[strategy1, strategy2])
```

**Benefício:** Trocar comportamento em runtime sem modificar Trainer.

---

## 2. Observer Pattern

**Definição:** Observadores são notificados quando o sujeito muda de estado.

**Aplicação:** Callbacks **observam** o progresso do treino.
```
Subject (Trainer) ──notifica──> Observer 1 (EarlyStopping)
                   └──notifica──> Observer 2 (ModelCheckpoint)
                   └──notifica──> Observer 3 (Logger)
```

**Benefício:** Desacoplamento. Trainer não sabe quantos/quais observers existem.

---

## 3. Template Method Pattern

**Definição:** Definir esqueleto de algoritmo, delegando passos para subclasses.

**Aplicação:** `Trainer.fit()` define a estrutura do loop:
```python
def fit(self, dataloader, epochs):
    self._notify("on_train_begin")  # Hook 1
    
    for epoch in range(epochs):
        self._train_one_epoch()
        self._notify("on_epoch_end")  # Hook 2
```

**Benefício:** Loop fixo, comportamento customizável via hooks.

---

## 4. Dependency Inversion Principle (SOLID)

**Definição:** Depender de abstrações, não implementações concretas.

**Aplicação:** Trainer depende do `Protocol Callback`, não de classes concretas.
```python
# ✅ Depende da abstração
class Trainer:
    def __init__(self, callbacks: List[Callback]):
        ...

# ❌ Seria errado depender de implementação
class Trainer:
    def __init__(self, early_stopping: EarlyStopping):
        ...
```

**Benefício:** Trainer funciona com **qualquer** callback, até os que não existem ainda.

---

## 5. Open/Closed Principle (SOLID)

**Definição:** Aberto para extensão, fechado para modificação.

**Demonstração:**

**Requisito novo:** "Enviar email quando treino acabar"
```python
# ✅ Solução: Criar novo callback (extensão)
class EmailNotifier:
    def on_epoch_end(self, epoch, logs):
        if epoch == logs.get("total_epochs") - 1:
            send_email("Treino concluído!")
        return False

trainer = Trainer(callbacks=[EmailNotifier()])
```

**❌ Alternativa ruim:** Modificar `Trainer.fit()` para adicionar `if send_email: ...`

**Benefício:** Sistema cresce sem quebrar código existente.

---

## 6. Single Responsibility Principle (SOLID)

**Definição:** Cada classe deve ter uma única razão para mudar.

**Aplicação:**

| Classe | Responsabilidade Única |
|--------|----------------------|
| `Trainer` | Executar loop de treino |
| `EarlyStopping` | Decidir quando parar |
| `ModelCheckpoint` | Salvar modelo |
| `MetricsLogger` | Registrar métricas |

**Anti-pattern (violaria SRP):**
```python
# ❌ Trainer fazendo tudo
class Trainer:
    def fit(self):
        for epoch in range(epochs):
            ...
            # Mistura logging, salvamento, decisão de parada
            if epoch % 10 == 0:
                print(...)
                torch.save(...)
            if loss < threshold:
                break
```

---

## Exercício: Identificar Patterns

Analise este código e identifique os patterns:
```python
trainer = Trainer(
    callbacks=[
        EarlyStopping(patience=5),      # Qual pattern?
        ModelCheckpoint(),              # Qual pattern?
        CustomLogger()                  # Qual pattern?
    ]
)
```

**Respostas:**
1. **Strategy:** Cada callback é uma estratégia
2. **Observer:** Callbacks observam o treino
3. **Composition:** Trainer composto por callbacks (não herda deles)

---

## Comparação: Herança vs. Composição

### ❌ Abordagem com Herança (inflexível)
```python
class TrainerWithEarlyStopping(Trainer):
    ...

class TrainerWithCheckpoint(Trainer):
    ...

# E se quiser ambos? Herança múltipla? 😱
class TrainerWithBoth(TrainerWithEarlyStopping, TrainerWithCheckpoint):
    ...
```

**Problemas:**
- Explosão de subclasses
- Difícil adicionar combinações
- Tight coupling

### ✅ Abordagem com Composição (flexível)
```python
# Qualquer combinação em runtime!
trainer = Trainer(callbacks=[
    EarlyStopping(),
    ModelCheckpoint(),
    CustomCallback1(),
    CustomCallback2()
])
```

**Vantagens:**
- Combinações ilimitadas
- Adicionar/remover em runtime
- Loose coupling

---

## Referências de Patterns

- **Strategy:** [Refactoring Guru](https://refactoring.guru/design-patterns/strategy)
- **Observer:** [Refactoring Guru](https://refactoring.guru/design-patterns/observer)
- **Template Method:** [Refactoring Guru](https://refactoring.guru/design-patterns/template-method)
- **SOLID Principles:** [Uncle Bob's Blog](https://blog.cleancoder.com/uncle-bob/2020/10/18/Solid-Relevance.html)
