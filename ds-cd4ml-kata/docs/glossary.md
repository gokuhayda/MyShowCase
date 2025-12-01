# 📖 Glossário Técnico - CD4ML e MLOps

**Índice Alfabético:** [A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [H](#h) | [I](#i) | [K](#k) | [L](#l) | [M](#m) | [O](#o) | [P](#p) | [Q](#r) | [S](#s) | [T](#t) | [V](#v) | [W](#w)

---

## A

### Accuracy (Acurácia)
**Definição:** Proporção de predições corretas sobre o total de predições.

**Fórmula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Exemplo:**
```
100 predições:
- 85 corretas (TP + TN)
- 15 erradas (FP + FN)
Accuracy = 85/100 = 0.85 (85%)
```

**Quando usar:**
- Classes balanceadas (50/50)

**Quando NÃO usar:**
- Classes desbalanceadas (ex: 95/5)
- Nesse caso, modelo que sempre prediz classe majoritária tem 95% accuracy mas é inútil!

**Ver também:** [Precision](#precision-precisão), [Recall](#recall-sensibilidade)

---

### Artifact
**Definição:** Qualquer arquivo produzido durante o pipeline de ML.

**Exemplos:**
- Modelo treinado (`model.pkl`)
- Dados processados (`features.csv`)
- Plots (`confusion_matrix.png`)
- Métricas (`metrics.json`)

**Por que importante:**
- Versionamento (DVC, MLflow)
- Reprodutibilidade
- Auditoria

**No projeto:**
```bash
models/
├── model.pkl          # Artifact (modelo)
└── metrics.json       # Artifact (métricas)
```

---

### Autostage (DVC)
**Definição:** Configuração do DVC que automaticamente adiciona arquivos `.dvc` ao staging do Git.

**Sem autostage:**
```bash
dvc add data.csv
git add data.csv.dvc    # ← Você precisa lembrar!
git commit -m "Add data"
```

**Com autostage:**
```bash
dvc config core.autostage true
dvc add data.csv        # Já adiciona ao git automaticamente!
git commit -m "Add data"
```

**Por que usar:**
- Economiza tempo
- Evita esquecer de versionar ponteiros DVC
- Workflow mais fluido

**Ativar:**
```bash
dvc config core.autostage true
```

**Ver também:** [DVC](#dvc-data-version-control)

---

## B

### Batch Prediction
**Definição:** Fazer predições para múltiplas amostras de uma vez (em lote).

**Comparação:**

| Tipo | Quando | Latência | Uso |
|------|--------|----------|-----|
| **Single** | 1 amostra por vez | ~1ms | APIs real-time |
| **Batch** | 1000+ amostras juntas | ~100ms total | Relatórios offline |

**Exemplo:**
```python
# Single prediction
prediction = model.predict(sample)  # 1ms

# Batch prediction (mais eficiente!)
predictions = model.predict(batch_1000)  # 50ms total
# = 0.05ms por amostra (20x mais rápido!)
```

**Por que usar:**
- Eficiência: Overhead fixo amortizado
- Throughput maior
- Melhor uso de GPU/CPU

**Ver também:** [Latência](#latência)

---

### Blue-Green Deployment
**Definição:** Estratégia de deploy com dois ambientes idênticos (Blue=atual, Green=novo).

**Como funciona:**
```
Antes do deploy:
Load Balancer → Blue (v1) ← 100% tráfego
                Green (v2) ← 0% tráfego (idle)

Durante deploy:
Load Balancer → switch instantâneo

Após deploy:
Load Balancer → Blue (v1) ← 0% tráfego (backup)
                Green (v2) ← 100% tráfego
```

**Vantagens:**
- ✅ Rollback instantâneo (< 1 segundo)
- ✅ Zero downtime
- ✅ Testável antes do switch

**Desvantagens:**
- ❌ 2x infraestrutura (custo)
- ❌ Complexidade operacional

**Quando usar:**
- SLA alto (99.99%+)
- Custo de downtime > custo de infra duplicada

**Ver também:** [Canary Deployment](#canary-deployment), [Shadow Mode](#shadow-mode)

---

## C

### Canary Deployment
**Definição:** Estratégia de deploy gradual, liberando nova versão para pequena % de usuários primeiro.

**Analogia:** "Canário na mina" - soltavam canário para testar se o ar estava seguro antes dos mineiros entrarem.

**Progressão típica:**
```
Hora 0:  95% → v1 (old)
          5% → v2 (canary) 🐤

Hora 1:  90% → v1
         10% → v2  (se métricas OK)

Hora 2:  50% → v1
         50% → v2  (se ainda OK)

Hora 3:   0% → v1
        100% → v2  (promover!)
```

**Implementação:**
```python
def get_model(user_id):
    # Hash garante que mesmo usuário sempre vê mesma versão
    if hash(user_id) % 100 < 5:  # 5% dos usuários
        return model_v2  # Canary
    else:
        return model_v1  # Stable
```

**Por que funciona:**
- Hash deterministico: mesmo usuário = mesma versão
- Fácil aumentar %: mude `< 5` para `< 10`, `< 50`, etc
- Minimiza blast radius (raio de impacto)

**Métricas a monitorar:**
- Error rate (taxa de erro)
- Latency (p50, p95, p99)
- Business metrics (conversão, receita)

**Decisão:**
```
Se canary_error_rate > stable_error_rate + threshold:
    → ROLLBACK! (canary tem problema)
Senão:
    → Aumentar % gradualmente
```

**Vantagens:**
- ✅ Risco minimizado (só 5% afetados se der ruim)
- ✅ Validação com tráfego real
- ✅ Rollback fácil (stop em qualquer %)

**Desvantagens:**
- ❌ Complexidade (gerenciar 2 versões simultaneamente)
- ❌ Lento (pode levar horas/dias)

**Ver também:** [Blue-Green](#blue-green-deployment), [A/B Testing](#ab-testing)

---

### CD (Continuous Delivery/Deployment)
**Definição:** Prática de fazer deploy de código automaticamente após passar em testes.

**Continuous Delivery vs Deployment:**

| Aspecto | Delivery | Deployment |
|---------|----------|------------|
| **Automação** | Até staging | Até produção |
| **Aprovação humana** | Necessária para prod | Totalmente automática |
| **Deploy frequência** | Sob demanda | A cada commit |

**Exemplo Continuous Delivery:**
```
git push → tests pass → build → deploy to staging
                                      ↓
                            [Human approval needed]
                                      ↓
                            Deploy to production
```

**Exemplo Continuous Deployment:**
```
git push → tests pass → build → deploy to staging
                              → deploy to production ✅
                              (TUDO automático!)
```

**Benefícios:**
- ✅ Deploy frequente (múltiplos por dia)
- ✅ Feedback rápido
- ✅ Bugs pequenos (fácil debugar)
- ✅ Rollback fácil

**Ver também:** [CI](#ci-continuous-integration), [CD4ML](#cd4ml-continuous-delivery-for-machine-learning)

---

### CD4ML (Continuous Delivery for Machine Learning)
**Definição:** Aplicação de práticas de CI/CD ao Machine Learning, adaptando para desafios únicos de ML.

**Por que ML é diferente de software tradicional?**

| Aspecto | Software | ML |
|---------|----------|-----|
| **Input** | Código | Código + **DADOS** |
| **Output** | Determinístico | **Probabilístico** |
| **Testes** | `assert x == 5` | `assert accuracy > 0.9` |
| **Degradação** | Bug no código | **Data drift** |
| **Reprodução** | `git checkout` | `git + DVC + ambiente` |

**Os 4 Pilares do CD4ML:**
```
1. VERSIONAMENTO
   ├─ Código: Git
   ├─ Dados: DVC ← Novo!
   ├─ Modelos: MLflow ← Novo!
   └─ Ambiente: Docker

2. AUTOMAÇÃO
   ├─ ETL automático
   ├─ Treino automático
   ├─ Validação automática
   └─ Deploy automático

3. TESTES (3 camadas)
   ├─ Dados: Schema, ranges, drift
   ├─ Modelo: Accuracy > threshold
   └─ Inferência: Latency < 100ms

4. MONITORAMENTO
   ├─ Data drift (P(X) mudou?)
   ├─ Model drift (P(Ŷ) mudou?)
   └─ Concept drift (P(Y|X) mudou?)
```

**Exemplo de pipeline CD4ML:**
```
git push
    ↓
[CI/CD Pipeline]
    ├─ Test data quality ✅
    ├─ Train model
    ├─ Test model metrics ✅
    ├─ Test inference ✅
    └─ Deploy (se tudo passar)
```

**Referência:** [Martin Fowler - CD4ML](https://martinfowler.com/articles/cd4ml.html)

**Ver também:** [Drift](#drift), [DVC](#dvc-data-version-control), [MLflow](#mlflow)

---

### CI (Continuous Integration)
**Definição:** Prática de integrar código frequentemente (várias vezes ao dia), com testes automáticos a cada integração.

**Problema que resolve:**

**❌ Sem CI (Integration Hell):**
```
Semana toda:
Dev A codifica na branch dele
Dev B codifica na branch dele
Dev C codifica na branch dele

Sexta à tarde:
"Vamos juntar tudo!" → CONFLITOS! 😱
Fim de semana inteiro debugando...
```

**✅ Com CI:**
```
Dev A: git push → testes rodam (5 min) → ✅ passou
Dev B: git push → testes rodam (5 min) → ❌ quebrou!
       Dev B corrige IMEDIATAMENTE (problema ainda fresco)
Dev C: git push → testes rodam (5 min) → ✅ passou
```

**Exemplo GitHub Actions:**
```yaml
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v  # Automático!
```

**Benefícios:**
- ✅ Problemas detectados em minutos (não em dias)
- ✅ Código sempre em estado "funcionando"
- ✅ Menos conflitos de merge
- ✅ Feedback rápido

**Regras de ouro do CI:**
1. Commitar frequentemente (várias vezes ao dia)
2. Testes devem ser rápidos (< 10 min)
3. Se testes quebram, consertar é prioridade #1

**Ver também:** [CD](#cd-continuous-deliverydeployment), [CI/CD](#cicd)

---

### CI/CD
**Definição:** Combinação de Continuous Integration + Continuous Delivery/Deployment.

**Fluxo completo:**
```
Desenvolvedor:
├─ Escreve código
├─ Escreve testes
├─ git commit
└─ git push
      ↓
CI (Continuous Integration):
├─ Detecta push automaticamente
├─ Instala dependências
├─ Roda testes
└─ ✅ Passou? → próximo passo
   ❌ Falhou? → notifica e para
      ↓
CD (Continuous Delivery):
├─ Build da aplicação
├─ Deploy para staging
├─ Testes de integração
└─ Aprovação (manual ou automática)
      ↓
CD (Continuous Deployment):
├─ Deploy para produção
├─ Smoke tests
└─ Monitoramento
```

**Ferramentas populares:**
- GitHub Actions (usado neste projeto)
- GitLab CI
- Jenkins
- CircleCI
- Travis CI

**Ver também:** [CI](#ci-continuous-integration), [CD](#cd-continuous-deliverydeployment)

---

### Class Weight (Peso de Classe)
**Definição:** Técnica para balancear classes desbalanceadas atribuindo pesos diferentes a cada classe.

**Problema:**
```
Dataset desbalanceado:
Class 0 (bad wine):  100 amostras
Class 1 (good wine):  10 amostras

Sem balanceamento:
Modelo aprende: "Sempre prediz 0"
Accuracy: 90% 🎉 ... mas é INÚTIL! (ignora classe 1)
```

**Solução com class_weight:**
```python
# Scikit-learn calcula pesos automaticamente
model = RandomForestClassifier(class_weight='balanced')

# Ou manualmente:
# class_weight = {0: 1.0, 1: 10.0}  # Classe 1 vale 10x mais
```

**Como funciona:**
```
Perda (loss) sem peso:
loss = erro_classe_0 + erro_classe_1

Perda COM peso balanceado:
loss = (1.0 * erro_classe_0) + (10.0 * erro_classe_1)
                                  ↑
                    Modelo "se importa" 10x mais com classe 1!
```

**Fórmula de 'balanced':**
```
weight_class_i = n_samples / (n_classes * n_samples_class_i)

Exemplo:
Total: 110 amostras, 2 classes
Class 0: 100 amostras → weight = 110 / (2 * 100) = 0.55
Class 1:  10 amostras → weight = 110 / (2 * 10)  = 5.50
```

**Quando usar:**
- Classes desbalanceadas (ex: 90/10, 95/5)
- Classe minoritária é importante (ex: detecção de fraude)

**Alternativas:**
- Oversampling (SMOTE)
- Undersampling
- Threshold adjustment

**Ver também:** [Precision](#precision-precisão), [Recall](#recall-sensibilidade)

---

### Concept Drift
**Definição:** Mudança na relação entre features (X) e target (Y), ou seja, P(Y|X) muda.

**Tipo mais grave de drift!**

**Exemplo real - COVID:**
```
ANTES (2019):
Features: Renda alta + Score 750
Target: Baixo risco de crédito ✅
P(inadimplente | renda_alta) = 5%

DURANTE (2020 - COVID):
Features: Renda alta + Score 750  (MESMAS features!)
Target: ALTO risco de crédito ❌
P(inadimplente | renda_alta) = 30%  ← MUDOU!

Causa: Economia mudou, relação mudou!
```

**Outro exemplo - Spam:**
```
ANTES:
"Compre Viagra" → Spam (100%)

DEPOIS:
"Compre Viagra" → Legit (farmácia real)
                   ↑
          Contexto mudou!
```

**Como detectar:**
```python
# Monitorar acurácia ao longo do tempo
if accuracy_semana_atual < accuracy_baseline - threshold:
    print("⚠️ Possível concept drift!")
```

**Soluções:**
1. **Feature engineering** (adicionar contexto)
```python
   # Adicionar features temporais
   df['is_pandemic'] = (df['date'] >= '2020-03-01')
```

2. **Janela temporal menor**
```python
   # Treinar apenas com últimos 3 meses
   train_data = df[df['date'] > '2024-10-01']
```

3. **Online learning**
```python
   # Retreinar continuamente
   model.partial_fit(new_data_batch)
```

4. **Retreinar periodicamente**
```python
   # Scheduled retraining (ex: mensal)
   if datetime.now().day == 1:
       retrain_model()
```

**Ver também:** [Data Drift](#data-drift), [Model Drift](#model-drift)

---

### Cross-Validation (Validação Cruzada)
**Definição:** Técnica para estimar performance do modelo de forma mais robusta, dividindo dados em K folds.

**Por que usar?**
```
❌ Treino/teste simples:
Train (80%) → Test (20%)
Problema: E se test set for "fácil" ou "difícil" por sorte?
```

**✅ 5-Fold Cross-Validation:**
```
Fold 1: [Train][Train][Train][Train][TEST]
Fold 2: [Train][Train][Train][TEST][Train]
Fold 3: [Train][Train][TEST][Train][Train]
Fold 4: [Train][TEST][Train][Train][Train]
Fold 5: [TEST][Train][Train][Train][Train]

Resultado final: Média dos 5 testes ± desvio padrão
```

**Exemplo:**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
# scores = [0.85, 0.87, 0.84, 0.86, 0.88]

print(f"Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
# Accuracy: 0.86 ± 0.01
```

**Interpretação:**
```
Accuracy: 0.86 ± 0.01
          ↑      ↑
       média   variação

Std baixo (< 0.05): Modelo estável ✅
Std alto (> 0.10):  Modelo instável ❌ (sensível ao split)
```

**Variantes:**
- **K-Fold**: K divisões iguais
- **Stratified K-Fold**: Mantém proporção de classes (SEMPRE use para classificação!)
- **Leave-One-Out**: K = n_samples (muito lento)
- **Time Series Split**: Respeita ordem temporal

**Ver também:** [Overfitting](#overfitting)

---

## D

### Data Drift
**Definição:** Mudança na distribuição das features (input), ou seja, P(X) muda, mas P(Y|X) permanece.

**Exemplo:**
```
E-commerce:

ANTES (2019):
Usuários: 70% formal, 30% casual

DEPOIS (Pandemia 2020):
Usuários: 10% formal, 90% casual ← P(X) mudou!

Problema:
Modelo foi treinado com 70% formal
Agora vê 90% casual → Performa mal!
```

**Como detectar:**
```python
from scipy.stats import ks_2samp

# Comparar distribuição de 'age' entre treino e produção
stat, p_value = ks_2samp(train_data['age'], prod_data['age'])

if p_value < 0.05:
    print("🚨 Data drift detectado na feature 'age'!")
```

**Interpretação do p-value:**
```
p_value = 0.8  (alto)  → Distribuições parecidas ✅
p_value = 0.001 (baixo) → Distribuições DIFERENTES 🚨
```

**Solução:**
```python
# Retreinar com dados recentes
new_train_data = get_recent_data(last_6_months=True)
model.fit(new_train_data)
```

**Ferramentas:**
- **Evidently**: Dashboard de drift
- **KS Test**: Teste estatístico (usado acima)
- **PSI** (Population Stability Index)
- **Chi-squared test**: Para features categóricas

**Ver também:** [Model Drift](#model-drift), [Concept Drift](#concept-drift), [KS Test](#ks-test-kolmogorov-smirnov)

---

### DVC (Data Version Control)
**Definição:** "Git para dados" - ferramenta para versionar datasets, modelos e pipelines de ML.

**Por que DVC existe?**

**❌ Problema com Git:**
```bash
# Dataset de 10 GB
git add data/train.csv  # ← Git guarda TUDO no repo
git commit

# 10 versões = 100 GB no repo Git 😱
# git clone demora HORAS
# GitHub rejeita (limite: 100 MB por arquivo)
```

**✅ Solução com DVC:**
```bash
# DVC guarda apenas ponteiro (~1 KB)
dvc add data/train.csv

# Cria:
# 1. data/train.csv.dvc (ponteiro, vai pro Git)
# 2. Move arquivo para .dvc/cache
# 3. Adiciona train.csv ao .gitignore

git add data/train.csv.dvc
git commit -m "Add training data"

# Repo Git: apenas ponteiro (1 KB)
# Dados reais: em S3/GCS (10 GB)
```

**Como funciona:**
```
Git armazena:               DVC armazena:
├─ Código (.py)            ├─ Dados (.csv)
├─ Configs (.yaml)         ├─ Modelos (.pkl)
└─ Ponteiros (.dvc)        └─ Artifacts grandes
   (~1 KB cada)               (GB/TB)
```

**Comandos essenciais:**
```bash
# Inicializar
dvc init --subdir  # Em subdiretório de um repo Git

# Adicionar dados
dvc add data/train.csv

# Configurar remote (S3, GCS, Azure, etc)
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Push/Pull
dvc push  # Upload para remote
dvc pull  # Download do remote

# Reproduzir pipeline
dvc repro

# Ver DAG
dvc dag
```

**Reprodutibilidade:**
```bash
# Experimento de 6 meses atrás
git checkout <commit-6-meses>
dvc checkout

# Agora você tem:
# ✅ Código exato (Git)
# ✅ Dados exatos (DVC)
# ✅ Pode retreinar e obter MESMO resultado!
```

**Ver também:** [Hash](#hash-md5), [Autostage](#autostage-dvc), [DVC Pipeline](#dvc-pipeline)

---

### DVC Pipeline
**Definição:** Grafo de dependências (DAG) que define etapas do pipeline de ML e suas relações.

**Definido em `dvc.yaml`:**
```yaml
stages:
  prepare_data:
    cmd: python src/data/make_dataset.py
    deps:
      - data/raw/wine_quality.csv
      - src/data/make_dataset.py
    outs:
      - data/processed/wine_features.csv
  
  train:
    cmd: python src/models/train.py
    deps:
      - data/processed/wine_features.csv
      - src/models/train.py
      - params.yaml
    params:
      - model
      - data
    outs:
      - models/model.pkl
    metrics:
      - models/metrics.json:
          cache: false
```

**Visualizar DAG:**
```bash
dvc dag

# Output:
#   +---------------+
#   | prepare_data  |
#   +---------------+
#           *
#           *
#           *
#      +-------+
#      | train |
#      +-------+
```

**Executar pipeline:**
```bash
# Reproduzir TUDO
dvc repro

# DVC é inteligente:
# - Se nada mudou → "Everything is up to date"
# - Se params.yaml mudou → Re-treina (mas não refaz ETL)
# - Se dados mudaram → Refaz TUDO
```

**Benefícios:**
- ✅ Cache inteligente (não refaz o que não precisa)
- ✅ Reprodutibilidade (DAG versionado no Git)
- ✅ Paralelização automática
- ✅ Rastreabilidade (quem depende de quem)

**Ver também:** [DVC](#dvc-data-version-control)

---

## E

### ETL (Extract, Transform, Load)
**Definição:** Processo de extrair dados da fonte, transformar (limpar, processar) e carregar para destino.

**No contexto de ML:**
```
EXTRACT:
└─ Baixar dataset (API, CSV, DB)

TRANSFORM:
├─ Limpeza (remover NaN, duplicatas)
├─ Validação (schema, ranges)
├─ Feature engineering
└─ Normalização/encoding

LOAD:
└─ Salvar features processadas
```

**Exemplo neste projeto:**
```python
# src/data/make_dataset.py

def main():
    # EXTRACT
    df_raw = load_raw_data("data/raw/wine_quality.csv")
    
    # TRANSFORM
    df_processed = create_features(df_raw)
    
    # LOAD
    save_processed_data(df_processed, "data/processed/wine_features.csv")
```

**Boas práticas:**
- ✅ Validar na entrada (Pandera schemas)
- ✅ Idempotente (rodar 2x = mesmo resultado)
- ✅ Logado (saber o que aconteceu)
- ✅ Testado (testes de dados)

**Ver também:** [Feature Engineering](#feature-engineering)

---

## F

### F1-Score
**Definição:** Média harmônica entre Precision e Recall. Balanceia ambas as métricas.

**Fórmula:**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Por que média harmônica (não aritmética)?**
```
Caso patológico:
Precision = 1.0 (perfeita!)
Recall    = 0.01 (péssimo!)

Média aritmética: (1.0 + 0.01) / 2 = 0.505  ← ENGANOSO!
Média harmônica:  2 * (1.0 * 0.01) / 1.01 = 0.019  ← HONESTO!
                  ↑
          Penaliza valores muito diferentes
```

**Exemplo:**
```
Confusion Matrix:
       Pred 0  Pred 1
True 0   120     30     (TN=120, FP=30)
True 1    15    155     (FN=15, TP=155)

Precision = 155 / (155 + 30) = 0.838
Recall    = 155 / (155 + 15) = 0.912

F1 = 2 * (0.838 * 0.912) / (0.838 + 0.912)
   = 2 * 0.764 / 1.75
   = 0.873
```

**Quando usar:**
- Classes desbalanceadas
- Você se importa IGUALMENTE com Precision e Recall
- Métrica única para comparar modelos

**Variantes:**
- **F2-Score**: Dá 2x mais peso ao Recall
```
  F2 = 5 * P * R / (4*P + R)
```
- **F0.5-Score**: Dá 2x mais peso ao Precision

**Ver também:** [Precision](#precision-precisão), [Recall](#recall-sensibilidade)

---

### Feature Engineering
**Definição:** Processo de criar novas features (variáveis) a partir das existentes para melhorar performance do modelo.

**Tipos:**

**1. Transformações matemáticas:**
```python
# Log transform (reduzir skewness)
df['log_price'] = np.log1p(df['price'])

# Polynomial features
df['age_squared'] = df['age'] ** 2

# Razões
df['debt_to_income'] = df['debt'] / df['income']
```

**2. Binning (discretização):**
```python
# Idade contínua → Grupos
df['age_group'] = pd.cut(df['age'], 
                          bins=[0, 18, 30, 50, 100],
                          labels=['teen', 'young', 'adult', 'senior'])
```

**3. Encoding categóricas:**
```python
# One-hot encoding
pd.get_dummies(df['city'])

# Label encoding
df['size_encoded'] = df['size'].map({'S': 1, 'M': 2, 'L': 3})
```

**4. Features temporais:**
```python
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])
df['hour'] = df['timestamp'].dt.hour
```

**5. Agregações:**
```python
# Média por grupo
df['avg_price_by_category'] = df.groupby('category')['price'].transform('mean')
```

**6. Interações:**
```python
# Multiplicar features
df['feature_interaction'] = df['feature_A'] * df['feature_B']
```

**No nosso projeto:**
```python
# src/data/make_dataset.py

def create_features(df):
    # Target engineering
    df['quality_binary'] = (df['quality'] >= 6).astype(int)
    
    # Possíveis melhorias:
    # df['acidity_ratio'] = df['fixed_acidity'] / df['volatile_acidity']
    # df['sugar_to_alcohol'] = df['residual_sugar'] / df['alcohol']
    
    return df
```

**Dica:** Começar simples, adicionar complexidade conforme necessário.

**Ver também:** [ETL](#etl-extract-transform-load)

---

### Feature Store
**Definição:** Sistema centralizado para armazenar, servir e compartilhar features entre treino e inferência.

**Problema que resolve (Training-Serving Skew):**
```
❌ SEM Feature Store:

TREINO (Data Scientist):
def calculate_features(df):
    return df['price'].rolling(7).mean()  # Python/Pandas

PRODUÇÃO (Engineer):
def calculate_features(data):
    // Reimplementado em Java
    return rollingAverage(data.price, 7);  // BUG: implementação diferente!

Resultado:
Treino: 95% accuracy
Produção: 70% accuracy 😱 (WTF?!)
```

**✅ COM Feature Store:**
```
Feature Store (Feast, Tecton, etc)
          │
    ┌─────┴─────┐
    │           │
  Treino    Inferência
(MESMAS    (MESMAS
features!)  features!)
```

**Exemplo (Feast):**
```python
from feast import FeatureStore

# Definir feature UMA VEZ
@feast.feature_view(...)
def user_features():
    return DataFrame([
        Feature("age", ValueType.INT64),
        Feature("activity_30d", ValueType.INT64)
    ])

# TREINO
training_df = store.get_historical_features(
    entity_df=entities,
    features=["user_features:age", "user_features:activity_30d"]
)

# INFERÊNCIA (MESMAS features!)
online_features = store.get_online_features(
    features=["user_features:age", "user_features:activity_30d"],
    entity_rows=[{"user_id": 123}]
)
```

**Benefícios:**
- ✅ **Consistência**: Treino = Inferência (sem skew)
- ✅ **Reuso**: Times compartilham features
- ✅ **Performance**: Features pré-computadas
- ✅ **Governança**: Versionamento, lineage

**Ferramentas:**
- Feast (open source)
- Tecton (enterprise)
- Databricks Feature Store
- AWS SageMaker Feature Store

**Ver também:** [Training-Serving Skew](#training-serving-skew)

---

## G

### Git
**Definição:** Sistema de controle de versão distribuído para código.

**Comandos básicos:**
```bash
# Clonar repositório
git clone <url>

# Status
git status

# Adicionar mudanças
git add arquivo.py
git add .  # Todos os arquivos

# Commit
git commit -m "Mensagem descritiva"

# Push (enviar para remoto)
git push origin main

# Pull (baixar do remoto)
git pull origin main

# Branches
git checkout -b feature/nova-feature
git merge feature/nova-feature

# Ver histórico
git log --oneline

# Voltar no tempo
git checkout <commit-hash>
```

**Boas práticas de commit:**
```
feat: Add new feature
fix: Fix bug in model training
docs: Update README
test: Add unit tests for ETL
refactor: Simplify data pipeline
chore: Update dependencies
```

**Ver também:** [DVC](#dvc-data-version-control), [GitHub Actions](#github-actions)

---

### GitHub Actions
**Definição:** Plataforma de CI/CD integrada ao GitHub que executa workflows automaticamente.

**Exemplo básico:**
```yaml
# .github/workflows/test.yaml

name: Run Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run tests
        run: pytest tests/ -v
```

**Triggers:**
- `push`: A cada push
- `pull_request`: A cada PR
- `schedule`: Cron (ex: diário às 2am)
- `workflow_dispatch`: Manual

**Ver também:** [CI/CD](#cicd)

---

## H

### Hash (MD5)
**Definição:** Função que transforma dados de qualquer tamanho em uma "impressão digital" fixa (32 caracteres no MD5).

**Analogia:** Impressão digital humana
```
Pessoa A → Impressão digital: ∞∞∞∞∞ (única)
Pessoa B → Impressão digital: ≈≈≈≈≈ (única)
```

**Hash de arquivos:**
```
Arquivo X → Hash: a3e4f5c6d7e8... (único)
Arquivo Y → Hash: 7b8c9d0e1f2a... (único)
```

**Propriedades mágicas:**

**1. Determinístico:**
```
arquivo.csv → md5sum → a3e4f5c6...
arquivo.csv → md5sum → a3e4f5c6...  (sempre igual!)
```

**2. Sensível a mudanças:**
```
"João,25" → Hash: a3e4f5c6...
"João,26" → Hash: 7b8c9d0e...  (totalmente diferente!)
      ↑
   Mudou 1 caractere!
```

**3. Tamanho fixo:**
```
1 KB   → Hash: a3e4f5c6... (32 chars)
5 GB   → Hash: 7b8c9d0e... (32 chars)
```

**Exemplo prático:**
```bash
# Calcular hash de um arquivo
md5sum data/train.csv
# Output: a3e4f5c6d7e8f9a0b1c2d3e4f5a6b7c8  data/train.csv

# 6 meses depois...
md5sum data/train.csv
# Output: a3e4f5c6d7e8f9a0b1c2d3e4f5a6b7c8

# Hash igual = arquivo NÃO mudou! ✅
```

**Por que DVC usa hash:**
```
❌ Comparar arquivos byte por byte:
5 GB × 5 GB = MUITO LENTO (minutos)

✅ Comparar hashes:
32 chars = 32 chars? INSTANTÂNEO (milissegundos)
```

**Calcular em Python:**
```python
import hashlib

def calcular_hash(arquivo):
    md5 = hashlib.md5()
    
    with open(arquivo, 'rb') as f:
        while chunk := f.read(8192):
            md5.update(chunk)
    
    return md5.hexdigest()

hash_result = calcular_hash('train.csv')
print(hash_result)  # a3e4f5c6d7e8f9a0b1c2d3e4f5a6b7c8
```

**Ver também:** [DVC](#dvc-data-version-control)

---

## I

### Inference (Inferência)
**Definição:** Processo de usar modelo treinado para fazer predições em dados novos.

**Tipos:**

**1. Single prediction (tempo real):**
```python
sample = {'age': 25, 'income': 50000}
prediction = model.predict(sample)
# Latência: ~1ms
```

**2. Batch prediction (offline):**
```python
batch = pd.read_csv('new_customers.csv')
predictions = model.predict(batch)
# Latência: ~100ms para 1000 samples
```

**Exemplo completo:**
```python
# src/models/predict.py

class WineQualityPredictor:
    def __init__(self):
        self.model = pickle.load(open('models/model.pkl', 'rb'))
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_with_confidence(self, X):
        pred = self.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        
        return {
            'prediction': int(pred),
            'confidence': float(proba[pred]),
            'probabilities': {
                0: float(proba[0]),
                1: float(proba[1])
            }
        }
```

**Testes de inferência:**
- ✅ Latência < threshold
- ✅ Formato correto
- ✅ Determinismo
- ✅ Edge cases

**Ver também:** [Latência](#latência), [Batch Prediction](#batch-prediction)

---

## K

### KS Test (Kolmogorov-Smirnov)
**Definição:** Teste estatístico que compara duas distribuições para verificar se são diferentes.

**Uso em ML:** Detectar data drift

**Exemplo:**
```python
from scipy.stats import ks_2samp

# Comparar idades entre treino e produção
stat, p_value = ks_2samp(train_data['age'], prod_data['age'])

if p_value < 0.05:
    print("🚨 Drift detectado!")
else:
    print("✅ Distribuições parecidas")
```

**Interpretação do p-value:**
```
p_value > 0.05:
"Não há evidência de que as distribuições sejam diferentes"
Interpretação: Distribuições parecem IGUAIS ✅

p_value < 0.05:
"Há forte evidência de que as distribuições são diferentes"
Interpretação: Distribuições são DIFERENTES 🚨
```

**Visualização:**
```
Treino:    |        ████████
           |       ██████████
           |      ████████████
           10    20    30    40

Produção:  |                ████████
           |               ██████████
           |              ████████████
           10    20    30    40    50

KS statistic = Máxima distância vertical entre curvas
p_value = Probabilidade dessa distância ser "normal"
```

**Ver também:** [Data Drift](#data-drift), [P-value](#p-value)

---

## L

### Latência
**Definição:** Tempo entre enviar requisição e receber resposta.

**Em ML:** Tempo para fazer uma predição

**Medidas típicas:**

| Cenário | Latência | Aceitável? |
|---------|----------|------------|
| **APIs real-time** | < 100ms | ✅ |
| **Batch processing** | < 1s para 100 samples | ✅ |
| **GPU inference** | < 10ms | ✅ |
| **API lenta** | > 1s | ❌ (usuário sente) |

**Medindo latência:**
```python
import time

start = time.time()
prediction = model.predict(sample)
latency_ms = (time.time() - start) * 1000

print(f"Latency: {latency_ms:.2f}ms")
```

**Percentis (mais importante que média!):**
```python
# Fazer 1000 predições
latencies = []
for _ in range(1000):
    start = time.time()
    model.predict(sample)
    latencies.append((time.time() - start) * 1000)

# Analisar distribuição
p50 = np.percentile(latencies, 50)  # Mediana
p95 = np.percentile(latencies, 95)  # 95% das requests
p99 = np.percentile(latencies, 99)  # 99% das requests

print(f"p50: {p50:.2f}ms")  # Metade < isso
print(f"p95: {p95:.2f}ms")  # 95% < isso
print(f"p99: {p99:.2f}ms")  # 99% < isso

# Exemplo:
# p50: 5ms   ← Metade é super rápido
# p95: 50ms  ← 95% OK
# p99: 500ms ← 1% muito lento! (investigar)
```

**Por que p99 importa:**
```
1 milhão de requests/dia
1% com 500ms de latência
= 10.000 usuários frustrados! 😠
```

**Ver também:** [Inference](#inference-inferência)

---

## M

### Makefile
**Definição:** Arquivo que define comandos úteis (atalhos) para tarefas comuns do projeto.

**Exemplo:**
```makefile
# Makefile

.PHONY: test train clean

# Rodar testes
test:
	pytest src/tests/ -v

# Treinar modelo
train:
	python src/models/train.py

# Limpar cache
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete

# Docker build
docker-build:
	docker build -t cd4ml-wine .

# Docker run
docker-run:
	docker run --rm cd4ml-wine
```

**Uso:**
```bash
make test         # Ao invés de: pytest src/tests/ -v
make train        # Ao invés de: python src/models/train.py
make docker-build # Ao invés de: docker build -t cd4ml-wine .
```

**Benefícios:**
- ✅ Comandos padronizados
- ✅ Documentação viva
- ✅ Onboarding mais fácil
- ✅ Menos erros de digitação

---

### MLflow
**Definição:** Plataforma open-source para gerenciar o ciclo de vida de ML: tracking, projetos, modelos e deployment.

**4 Componentes:**

**1. MLflow Tracking** (mais usado):
```python
import mlflow

# Logar experimento
with mlflow.start_run():
    # Parâmetros
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Métricas
    mlflow.log_metric("accuracy", 0.87)
    mlflow.log_metric("f1", 0.85)
    
    # Artifacts
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("plots/confusion_matrix.png")
    
    # Tags
    mlflow.set_tag("author", "eric@company.com")
```

**2. MLflow Projects:**
Definir ambiente reprodutível (MLproject file)

**3. MLflow Models:**
Formato padrão para empacotar modelos

**4. MLflow Model Registry:**
Gerenciar modelos em produção (Staging → Production)

**Ver experimentos:**
```bash
# Iniciar UI
mlflow ui

# Abrir navegador:
# http://localhost:5000
```

**Benefícios:**
- ✅ Rastreabilidade (quem, quando, o quê)
- ✅ Comparação de experimentos
- ✅ Reprodutibilidade
- ✅ Compliance ready

**Ver também:** [Experiment Tracking](#experiment-tracking)

---

### Model Drift
**Definição:** Mudança na distribuição das predições do modelo, ou seja, P(Ŷ) muda.

**Exemplo:**
```
Spam detector:

TREINO:
Predições: 5% spam, 95% ham

PRODUÇÃO (depois de semanas):
Predições: 30% spam, 70% ham  ← P(Ŷ) mudou!

Causas possíveis:
1. Data drift (input mudou)
2. Bug no código
3. Adversarial attack (spammers se adaptaram)
```

**Como detectar:**
```python
# Monitorar distribuição de predições
train_pred_dist = model.predict(train_data).mean()  # 0.50
prod_pred_dist = model.predict(prod_data).mean()    # 0.75

if abs(prod_pred_dist - train_pred_dist) > 0.10:
    print("⚠️ Model drift detectado!")
```

**Soluções:**
1. Investigar causa raiz (data drift? bug?)
2. Retreinar se necessário
3. A/B test (novo vs antigo)

**Ver também:** [Data Drift](#data-drift), [Concept Drift](#concept-drift)

---

## O

### Overfitting
**Definição:** Modelo "decora" os dados de treino ao invés de aprender padrões generalizáveis.

**Sintoma:**
```
Train accuracy: 99% 🎉
Test accuracy:  60% 😱

Gap = 39% ← OVERFITTING!
```

**Analogia:** Estudante que decora respostas da prova passada mas não entende a matéria.

**Causas:**
- Modelo muito complexo (árvore profunda demais)
- Poucos dados
- Features ruins (data leakage)

**Soluções:**

**1. Regularização:**
```python
# L1 (Lasso) ou L2 (Ridge)
model = RandomForestClassifier(
    max_depth=5,           # Limitar complexidade
    min_samples_split=10,  # Exigir mais samples
)
```

**2. Mais dados:**
```python
# Data augmentation, oversampling, etc
```

**3. Cross-validation:**
```python
# Avaliar em múltiplos folds
scores = cross_val_score(model, X, y, cv=5)
```

**4. Early stopping:**
```python
# Parar treino quando validação não melhora
```

**Ver também:** [Cross-Validation](#cross-validation-validação-cruzada), [Quality Gates](#quality-gates)

---

## P

### Pandera
**Definição:** Biblioteca Python para validação de DataFrames (tipo Pydantic para dados).

**Por que usar:**
```python
# ❌ Sem validação
df = pd.read_csv('data.csv')
model.fit(df)  # 💥 Quebra em produção se dados mudarem!

# ✅ Com Pandera
schema.validate(df)  # Valida ANTES de treinar
model.fit(df)  # Seguro!
```

**Exemplo:**
```python
import pandera as pa
from pandera import Column, Check

# Definir schema
schema = pa.DataFrameSchema({
    "age": Column(int, Check.in_range(0, 120)),
    "income": Column(float, Check.greater_than(0)),
    "city": Column(str, Check.isin(['SP', 'RJ', 'MG'])),
}, strict=True)

# Validar
try:
    schema.validate(df)
    print("✅ Dados válidos!")
except pa.errors.SchemaError as e:
    print(f"❌ Erro: {e}")
```

**No projeto:**
```python
# src/data/schemas.py

raw_schema = pa.DataFrameSchema({
    "pH": Column(float, Check.in_range(2.5, 4.5)),
    "alcohol": Column(float, Check.in_range(8, 15)),
    # ...
})
```

**Ver também:** [Schema Validation](#schema-validation)

---

### Precision (Precisão)
**Definição:** Das predições positivas, quantas estavam corretas?

**Fórmula:**
```
Precision = TP / (TP + FP)
           verdadeiros positivos
           -----------------------
           todos os positivos preditos
```

**Exemplo:**
```
Detector de spam:

Predisse "spam" 100 vezes:
- 85 eram realmente spam (TP)
- 15 eram ham (FP - falso alarme!)

Precision = 85 / 100 = 0.85 (85%)
```

**Interpretação:**
```
Precision alta (90%+):
"Quando digo que é spam, PROVAVELMENTE é spam"
Poucos falsos positivos ✅

Precision baixa (50%):
"Quando digo que é spam, pode não ser..."
Muitos falsos positivos ❌
```

**Quando otimizar Precision:**
- Custo de FP é alto
- Exemplo: Aprovar fraude (perda de dinheiro)
- Exemplo: Enviar email para spam (usuário perde email importante)

**Trade-off com Recall:**
```
Modelo conservador:
├─ Só prediz positivo quando MUITO confiante
├─ Precision ALTA ✅
└─ Recall BAIXO ❌ (perde muitos positivos)

Modelo agressivo:
├─ Prediz positivo com pouca confiança
├─ Precision BAIXA ❌
└─ Recall ALTO ✅ (pega quase tudo)
```

**Ver também:** [Recall](#recall-sensibilidade), [F1-Score](#f1-score), [Confusion Matrix](#confusion-matrix)

---

### P-value
**Definição:** Probabilidade de observar um resultado tão extremo quanto o observado, assumindo que a hipótese nula é verdadeira.

**Tradução simples:** "Quão improvável é esse resultado se não houver efeito real?"

**Analogia - moeda:**
```
Você joga moeda 100 vezes:
Resultado: 90 caras, 10 coroas

Pergunta: "Essa moeda é viciada?"

P-value responde:
"Se a moeda fosse JUSTA, qual probabilidade de ver 90/10?"

p-value = 0.0001 (muito baixo!)
Interpretação: "É MUITO improvável ver 90/10 em moeda justa"
Conclusão: Moeda provavelmente é viciada! 🎲
```

**No contexto de Drift:**
```python
stat, p_value = ks_2samp(train_age, prod_age)

p_value = 0.001 (baixo)
Interpretação:
"É muito improvável que essas distribuições sejam iguais"
Conclusão: DRIFT detectado! 🚨

p_value = 0.73 (alto)
Interpretação:
"É bem possível que essas distribuições sejam iguais"
Conclusão: Sem drift ✅
```

**Regra prática:**
```
p-value < 0.05: Rejeita hipótese nula (há diferença!)
p-value ≥ 0.05: Não rejeita (pode ser igual)
```

**CUIDADO:** P-value NÃO é "probabilidade de estar errado"!

**Ver também:** [KS Test](#ks-test-kolmogorov-smirnov), [Data Drift](#data-drift)

---

### Pytest
**Definição:** Framework de testes para Python.

**Exemplo básico:**
```python
# test_example.py

def test_addition():
    assert 1 + 1 == 2

def test_list_length():
    my_list = [1, 2, 3]
    assert len(my_list) == 3
```

**Rodar:**
```bash
pytest test_example.py -v
```

**Fixtures:**
```python
import pytest

@pytest.fixture
def sample_data():
    return pd.DataFrame({'age': [25, 30, 35]})

def test_mean_age(sample_data):
    assert sample_data['age'].mean() == 30
```

**Parametrize:**
```python
@pytest.mark.parametrize("a,b,expected", [
    (1, 1, 2),
    (2, 3, 5),
    (10, 5, 15),
])
def test_addition(a, b, expected):
    assert a + b == expected
```

**Ver também:** [TDD](#tdd-test-driven-development)

---

## Q

### Quality Gates
**Definição:** Thresholds mínimos que o modelo DEVE atingir para ser considerado "production-ready".

**Exemplo neste projeto:**
```yaml
# params.yaml

metrics:
  min_accuracy: 0.75    # 75%
  min_precision: 0.73
  min_recall: 0.73
  min_f1: 0.73
  max_train_test_gap: 0.10  # Máx 10% overfitting
```

**Implementação:**
```python
def validate_quality_gates(metrics, thresholds):
    passed = True
    
    if metrics['accuracy'] < thresholds['min_accuracy']:
        print("❌ Accuracy below threshold")
        passed = False
    
    if metrics['overfitting_gap'] > thresholds['max_gap']:
        print("❌ Overfitting detected")
        passed = False
    
    return passed

# No CI/CD:
if not validate_quality_gates(metrics, thresholds):
    exit(1)  # Fail pipeline!
```

**Por que importante:**
- ✅ Evita deploy de modelos ruins
- ✅ Padronização (todos os modelos passam pelos mesmos critérios)
- ✅ Auditabilidade (compliance)

**Como definir thresholds:**
1. Baseline (modelo simples)
2. Requisitos de negócio
3. Benchmarks da literatura
4. A/B test com threshold gradualmente maior

**Ver também:** [Accuracy](#accuracy-acurácia), [Overfitting](#overfitting)

---

## R

### Recall (Sensibilidade)
**Definição:** Dos casos positivos reais, quantos o modelo detectou?

**Fórmula:**
```
Recall = TP / (TP + FN)
        verdadeiros positivos
        ----------------------
        todos os positivos reais
```

**Exemplo:**
```
Detector de câncer:

100 pacientes COM câncer (ground truth):
- 90 foram detectados (TP)
- 10 foram perdidos! (FN)

Recall = 90 / 100 = 0.90 (90%)
```

**Interpretação:**
```
Recall alto (95%+):
"Pego QUASE TODOS os casos positivos"
Poucos falsos negativos ✅

Recall baixo (60%):
"Perco MUITOS casos positivos"
Muitos falsos negativos ❌
```

**Quando otimizar Recall:**
- Custo de FN é alto
- Exemplo: Diagnóstico de doença grave (perder um caso = fatal)
- Exemplo: Detecção de fraude (perder fraude = perda financeira)

**Analogia:**
```
Pescador:
Precision = % de peixes vs lixo na rede
Recall = % de peixes do lago que você pegou

Recall alto: Pegou quase todos os peixes! (mas também muito lixo)
Precision alto: Só pegou peixes! (mas deixou muitos no lago)
```

**Ver também:** [Precision](#precision-precisão), [F1-Score](#f1-score)

---

### Requirements.txt
**Definição:** Arquivo que lista todas as dependências Python do projeto.

**Exemplo:**
```
# requirements.txt
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
pytest==7.4.3
```

**Gerar:**
```bash
# Instalar tudo que precisa
pip install pandas scikit-learn pytest

# Salvar versões exatas
pip freeze > requirements.txt
```

**Usar:**
```bash
# Em nova máquina
pip install -r requirements.txt
```

**Por que versões fixas:**
```
❌ pandas (sem versão)
Problema: Update quebra código!

✅ pandas==2.1.4
Garantia: Sempre mesma versão = funciona!
```

**Ver também:** [Reprodutibilidade](#reprodutibilidade)

---

### Reprodutibilidade
**Definição:** Capacidade de obter exatamente o mesmo resultado ao repetir um experimento.

**Em ML, requer versionar:**
```
✅ Código (Git)
✅ Dados (DVC)
✅ Hiperparâmetros (params.yaml)
✅ Dependências (requirements.txt)
✅ Ambiente (Docker)
✅ Seeds (random_state=42)
```

**Exemplo:**
```bash
# Experimento de 6 meses atrás
git checkout <commit-hash>
dvc checkout
dvc repro

# Resultado: EXATAMENTE o mesmo modelo!
```

**Por que importante:**
- ✅ Compliance (regulação exige)
- ✅ Debugging (reproduzir bug)
- ✅ Ciência (validação externa)
- ✅ Produção (deploy confiável)

**Ver também:** [DVC](#dvc-data-version-control), [MLflow](#mlflow)

---

## S

### Schema Validation
**Definição:** Validar que dados atendem estrutura esperada (colunas, tipos, constraints).

**Por que necessário:**
```
❌ Sem validação:
df = pd.read_csv('data.csv')
model.fit(df)
# 💥 Quebra em produção se:
#    - Coluna renomeada
#    - Tipo mudou (str → float)
#    - Valores fora do range
```

**✅ Com validação:**
```python
schema.validate(df)  # Fail fast se algo errado!
model.fit(df)
```

**Exemplo:**
```python
import pandera as pa

schema = pa.DataFrameSchema({
    "age": Column(int, Check.in_range(0, 120)),
    "income": Column(float, Check.greater_than(0)),
}, strict=True)  # strict: não permite colunas extras
```

**Ver também:** [Pandera](#pandera)

---

### Shadow Mode
**Definição:** Estratégia de deploy onde novo modelo roda em paralelo ao antigo, mas NÃO serve usuários (só loga predições).

**Como funciona:**
```
Request →
    ├─ Champion (v1) → Serve user ✅
    └─ Challenger (v2) → Only logs 📝 (não serve!)
```

**Exemplo:**
```python
def predict(request):
    # Champion serve usuário
    prediction_v1 = model_v1.predict(request)
    
    # Challenger só loga (background)
    prediction_v2 = model_v2.predict(request)
    log_to_monitoring({
        'v1': prediction_v1,
        'v2': prediction_v2,
        'request_id': request.id
    })
    
    return prediction_v1  # Usuário recebe v1
```

**Após dias/semanas:**
```python
# Analisar logs
compare_predictions(v1_logs, v2_logs)

# Se v2 melhor:
promote_to_production(model_v2)
```

**Vantagens:**
- ✅ **Zero risco** (usuários não afetados)
- ✅ Validação com tráfego REAL
- ✅ Métricas side-by-side

**Desvantagens:**
- ❌ **2x custo computacional** (roda 2 modelos)
- ❌ Não valida latência real (não está no critical path)

**Quando usar:**
- Sistemas críticos (saúde, finanças)
- Custo de erro > custo de infra 2x

**Ver também:** [Canary](#canary-deployment), [A/B Testing](#ab-testing)

---

## T

### TDD (Test-Driven Development)
**Definição:** Metodologia onde você escreve TESTES antes do código.

**Ciclo Red-Green-Refactor:**
```
1. RED: Escrever teste (que falha)
   test_accuracy():
       assert accuracy > 0.9

2. GREEN: Escrever código mínimo (que passa)
   def train():
       return model_with_90_accuracy

3. REFACTOR: Melhorar código
   def train():
       # Código limpo e otimizado
```

**Benefícios:**
- ✅ Testes garantem funcionalidade
- ✅ Código testável por design
- ✅ Documentação viva

**Ver também:** [Pytest](#pytest)

---

### Training-Serving Skew
**Definição:** Diferença entre dados/features usados no treino vs produção.

**Problema:**
```
TREINO:
features = calculate_features_v1(data)  # Python
model.fit(features)
Accuracy: 95% ✅

PRODUÇÃO:
features = calculate_features_v2(data)  # Java (reimplementado)
model.predict(features)
Accuracy: 70% 😱 (WTF?!)

Causa: Features DIFERENTES!
```

**Exemplo real:**
```python
# TREINO (Data Scientist):
df['avg_7d'] = df['price'].rolling(7).mean()

# PRODUÇÃO (Engineer reimplementou):
avg = sum(last_7_prices) / 7  # BUG: não usa padding!
```

**Solução: Feature Store**
```
Single source of truth para features
Treino e produção usam MESMAS features
```

**Ver também:** [Feature Store](#feature-store)

---

## V

### Versionamento
**Definição:** Rastrear mudanças ao longo do tempo.

**No CD4ML, versionar:**
```
1. Código → Git
   git commit -m "Add feature X"

2. Dados → DVC
   dvc add data/train.csv

3. Modelos → DVC + MLflow
   dvc add models/model.pkl
   mlflow.sklearn.log_model(model, "model")

4. Hiperparâmetros → params.yaml (Git)
   git add params.yaml

5. Ambiente → requirements.txt + Docker
   pip freeze > requirements.txt
```

**Por que importante:**
```
Sem versionamento:
"Qual modelo está em produção?" 🤷
"Quais dados usei há 6 meses?" 🤷
"Por que accuracy caiu?" 🤷

Com versionamento:
git log  → Vejo código exato
dvc log  → Vejo dados exatos
mlflow ui → Vejo experimentos
```

**Ver também:** [Git](#git), [DVC](#dvc-data-version-control), [MLflow](#mlflow)

---

## W

### Workflow
**Definição:** Sequência de passos automatizados no CI/CD.

**Exemplo GitHub Actions:**
```yaml
name: ML Pipeline

on: [push]

jobs:
  test-data:
    steps:
      - Run data tests
  
  train:
    needs: test-data
    steps:
      - Train model
  
  deploy:
    needs: train
    steps:
      - Deploy to prod
```

**Ver também:** [GitHub Actions](#github-actions), [CI/CD](#cicd)

---

## 🎓 RESUMO DOS TERMOS MAIS IMPORTANTES

Para entrevista ThoughtWorks, memorize especialmente:

1. **CD4ML** - O conceito principal
2. **DVC** - Versionamento de dados
3. **Hash** - Como DVC identifica arquivos
4. **Drift** (3 tipos) - Data, Model, Concept
5. **CI/CD** - Automação
6. **Canary Deployment** - Deploy gradual
7. **Quality Gates** - Thresholds de produção
8. **Feature Store** - Evitar training-serving skew
9. **Precision vs Recall** - Métricas fundamentais
10. **Overfitting** - Problema comum

---

## 📚 REFERÊNCIAS

- [Martin Fowler - CD4ML](https://martinfowler.com/articles/cd4ml.html)
- [Google - Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

<div align="center">

**📖 Glossário criado para CD4ML Production Project**

*Última atualização: Dezembro 2025*

</div>