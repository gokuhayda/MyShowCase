# 🚀 CD4ML Production Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DVC](https://img.shields.io/badge/-Data_Version_Control-white.svg?logo=data-version-control&style=social)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-ready-326CE5.svg?logo=kubernetes)](https://kubernetes.io/)
[![CD4ML Pipeline](https://github.com/gokuhayda/MyShowCase/actions/workflows/train_cml.yaml/badge.svg)](https://github.com/gokuhayda/MyShowCase/actions/workflows/train_cml.yaml)

**Continuous Delivery for Machine Learning (CD4ML)** - Um projeto de referência demonstrando práticas de MLOps em produção.

---

## 📖 Sobre o Projeto

Este projeto implementa um **pipeline completo de Machine Learning** aplicando princípios de **CD4ML** (Continuous Delivery for Machine Learning), incluindo:

- ✅ **Versionamento**: Git (código) + DVC (dados/modelos) + MLflow (experimentos)
- ✅ **Validação de Dados**: Schemas com Pandera + testes automatizados (16 tests)
- ✅ **Quality Gates**: Thresholds de performance para deploy
- ✅ **CI/CD**: GitHub Actions com testes automáticos (3 jobs)
- ✅ **API REST**: FastAPI com 5 endpoints + documentação Swagger
- ✅ **Testes Completos**: 71 testes (data: 16, model: 8, inference: 9, API: 38)
- ✅ **Docker**: Container otimizado (~500 MB)
- ✅ **Kubernetes**: Deploy production-ready com auto-scaling
- ✅ **Makefile**: 50+ comandos para automação completa
- ✅ **Reprodutibilidade**: DVC pipeline + Docker + params.yaml
- ✅ **Monitoramento**: Preparado para drift detection

### 🎯 Problema de Negócio

**Classificação de qualidade de vinhos** baseado em propriedades físico-químicas.

- **Dataset**: Wine Quality (UCI Machine Learning Repository)
- **Tipo**: Classificação binária (Vinho bom ≥6 vs Vinho ruim <6)
- **Features**: 11 atributos físico-químicos (acidez, pH, álcool, etc)
- **Amostras**: 1.599 vinhos tintos

### 🏆 Resultados

| Métrica | Valor | Threshold |
|---------|-------|-----------|
| **Accuracy** | 86.56% | ≥ 75% ✅ |
| **Precision** | 85.71% | ≥ 73% ✅ |
| **Recall** | 91.23% | ≥ 73% ✅ |
| **F1-Score** | 88.39% | ≥ 73% ✅ |
| **AUC-ROC** | 92.87% | ≥ 70% ✅ |
| **CV Accuracy** | 85.38% ± 2.61% | - |

---

## 🏗️ Arquitetura do Projeto
```
ds-cd4ml-kata/
│
├── .github/workflows/          # CI/CD Pipeline
│   └── train_cml.yaml         # 3 jobs: test-data → train → deploy
│
├── .dvc/                       # DVC configuration
│   ├── config                 # DVC remote (DagsHub)
│   └── cache/                 # Local cache
│
├── api/                        # FastAPI REST API
│   ├── __init__.py
│   ├── main.py                # 5 endpoints + middleware
│   ├── models.py              # Pydantic schemas
│   ├── predictor.py           # Prediction service
│   └── tests/                 # API tests (38 tests)
│       ├── conftest.py
│       ├── test_endpoints.py
│       ├── test_validation.py
│       ├── test_performance.py
│       └── test_integration.py
│
├── data/
│   ├── raw/                   # Dados originais (DVC tracked)
│   │   └── wine_quality.csv
│   └── processed/             # Features engineering (DVC tracked)
│       └── wine_features.csv
│
├── k8s/                        # Kubernetes manifests
│   ├── deployment.yaml        # Pods + HPA (3-10 replicas)
│   ├── service.yaml           # LoadBalancer
│   ├── ingress.yaml           # HTTP routing + SSL
│   ├── configmap.yaml         # Configuration
│   ├── secret.yaml            # Secrets (template)
│   ├── pvc.yaml               # Persistent storage
│   ├── namespace.yaml         # Environments
│   └── kustomization.yaml     # Kustomize config
│
├── models/
│   ├── model.pkl              # Modelo treinado (DVC tracked)
│   └── metrics.json           # Métricas de avaliação
│
├── src/
│   ├── data/
│   │   ├── download_data.py   # Download do dataset (UCI)
│   │   ├── make_dataset.py    # ETL + Feature Engineering
│   │   └── schemas.py         # Validação com Pandera
│   │
│   ├── models/
│   │   ├── train.py           # Pipeline de treino (MLflow)
│   │   └── predict.py         # Inferência
│   │
│   └── tests/
│       ├── test_data_quality.py     # 16 tests
│       ├── test_model_metrics.py    # 8 tests
│       └── test_inference.py        # 9 tests
│
├── docs/                       # Documentação completa
│   └── GLOSSARY.md            # Glossário técnico (A-Z)
│
├── Dockerfile                  # Multi-stage build (~500 MB)
├── docker-compose.yml          # Orquestração (app + mlflow)
├── dvc.yaml                    # Pipeline DVC (3 stages)
├── params.yaml                 # Hiperparâmetros centralizados
├── requirements.txt            # Dependências Python
├── requirements-dev.txt        # Dev dependencies
├── pytest.ini                  # Pytest configuration
├── Makefile                    # 50+ comandos úteis
├── run_api.py                  # API startup script
├── test_api_client.py          # API test client
└── README.md                   # Este arquivo
```

---

## 🚀 Quick Start

### Pré-requisitos

- Python 3.10+
- Git
- DVC
- Docker (opcional)
- Kubernetes (opcional para deploy)

### 1️⃣ Clone o Repositório
```bash
git clone git@github.com:gokuhayda/MyShowCase.git
cd MyShowCase/ds-cd4ml-kata
```

### 2️⃣ Setup Completo (Makefile)
```bash
# Setup automático (venv + deps + data)
make setup

# Ou manualmente:
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 3️⃣ Baixar Dados com DVC
```bash
# Pull do DVC remote
dvc pull

# Ou baixar manualmente
python src/data/download_data.py
```

### 4️⃣ Executar Pipeline ETL
```bash
make data-prepare

# Ou:
python src/data/make_dataset.py
```

**Saída esperada:**
```
============================================================
🚀 Starting ETL Pipeline
============================================================
📂 Loading data from data/raw/wine_quality.csv
✅ Validating raw data schema...
   Shape: (1599, 12)
🔧 Creating features...
   Features: 12 columns
   Target distribution:
      Class 0: 744 (46.5%)
      Class 1: 855 (53.5%)
💾 Saving processed data...
✅ ETL Pipeline completed successfully!
```

### 5️⃣ Treinar Modelo
```bash
make train

# Ou:
python src/models/train.py
```

### 6️⃣ Rodar Testes
```bash
# Todos os testes (71 total)
make test-all

# Por categoria
make test-data      # 16 tests
make test-model     # 8 tests
make test-inference # 9 tests
make test-api       # 38 tests

# Com coverage
make test-cov
```

### 7️⃣ Iniciar API
```bash
# Via Makefile
make api

# Ou diretamente
python run_api.py

# Acessar:
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

### 8️⃣ MLflow UI
```bash
make experiments

# Ou:
mlflow ui

# Acessar: http://localhost:5000
```

---

## 📊 Pipeline DVC

O projeto usa DVC para orquestrar o pipeline de ML:
```bash
# Ver DAG (grafo de dependências)
make dvc-dag

# Reproduzir pipeline completo
make dvc-repro

# Ver métricas
make dvc-metrics

# Comparar experimentos
dvc metrics diff HEAD~1 HEAD
```

**Pipeline definido em `dvc.yaml`:**
```
download_data → prepare_data → train
     ↓              ↓            ↓
wine_quality.csv  features.csv  model.pkl
                               metrics.json
```

---

## 🧪 Testes Automatizados

O projeto possui **71 testes** organizados em 4 camadas:

### 1. Testes de Dados (16 tests) ✅
```bash
make test-data
```
- Schema compliance (Pandera)
- No missing values
- No duplicates
- Target distribution (min 30% por classe)
- Feature ranges válidos
- No data leakage (correlação < 0.95)
- Sample size mínimo (≥1000)
- Feature count correto (12)
- Target binário (0 ou 1)

### 2. Testes de Modelo (8 tests) ✅
```bash
make test-model
```
- Accuracy ≥ threshold (75%)
- Precision ≥ threshold (73%)
- Recall ≥ threshold (73%)
- F1-score ≥ threshold (73%)
- Overfitting gap ≤ threshold (10%)
- Cross-validation estável (std ≤ 0.05)
- AUC ≥ 0.70
- Métricas essenciais presentes

### 3. Testes de Inferência (9 tests) ✅
```bash
make test-inference
```
- Predição retorna classe válida (0 ou 1)
- Probabilidades somam 1.0
- Latência < 100ms (single)
- Latência < 1s (batch de 100)
- Features faltando gera erro
- Determinismo
- Consistência batch/single
- Edge cases
- Formato de resposta correto

### 4. Testes de API (38 tests) ✅
```bash
make test-api
```
**Endpoints (23 tests):**
- GET / (root)
- GET /health (health check)
- POST /predict (single)
- POST /predict/batch (batch)
- GET /model/info
- Error handling (404, 405, 422)

**Validation (11 tests):**
- Range validation (parametrized)
- Type validation
- Required fields
- Boundary values

**Performance (5 tests):**
- Latency thresholds
- Throughput

**Integration (6 tests):**
- End-to-end workflows
- Consistency

---

## 🔧 Configuração (params.yaml)

Todos os hiperparâmetros são centralizados:
```yaml
model:
  algorithm: RandomForest
  n_estimators: 100        # Número de árvores
  max_depth: 10            # Profundidade máxima
  min_samples_split: 5     # Mín. amostras para split
  min_samples_leaf: 2      # Mín. amostras por folha
  random_state: 42         # Seed (reprodutibilidade)
  class_weight: balanced   # Balanceamento de classes

data:
  test_size: 0.2           # 80% treino, 20% teste
  random_state: 42
  stratify: true

metrics:
  min_accuracy: 0.75       # Quality gates
  min_precision: 0.73
  min_recall: 0.73
  min_f1: 0.73
  max_train_test_gap: 0.10

cv:
  n_splits: 5              # 5-fold cross-validation
  shuffle: true
```

---

## 🐳 Docker

### Build
```bash
make docker-build

# Ou:
docker build -t cd4ml-wine-quality:latest .
```

### Run
```bash
# Treino
make docker-train

# Testes
make docker-test

# API
docker run --rm -p 8000:8000 cd4ml-wine-quality:latest

# Shell interativo
make docker-shell
```

### Docker Compose
```bash
# Iniciar todos os serviços (app + mlflow)
make compose-up

# Ver logs
make compose-logs

# Parar
make compose-down
```

---

## ☸️ Kubernetes

### Deploy
```bash
# 1. Build e push da imagem
docker build -t your-registry/cd4ml-wine-quality:v1.0.0 .
docker push your-registry/cd4ml-wine-quality:v1.0.0

# 2. Criar secrets
kubectl create secret generic wine-quality-secrets \
  --from-literal=api-key=your-api-key \
  -n production

# 3. Deploy
kubectl apply -f k8s/ -n production

# 4. Verificar
kubectl get all -n production
```

### Features K8s

- **Deployment**: 3 replicas com rolling updates
- **HPA**: Auto-scaling (3-10 pods baseado em CPU/Memory)
- **Service**: LoadBalancer externo
- **Ingress**: HTTP routing + SSL/TLS
- **ConfigMap**: Configurações não-sensíveis
- **Secret**: Dados sensíveis (API keys, tokens)
- **PVC**: Storage persistente para modelos
- **Namespaces**: Isolamento (dev/staging/prod)
- **Health Checks**: Liveness + Readiness probes

### Acesso
```bash
# Get external IP
kubectl get service wine-quality-api -n production

# Acessar API
curl http://<EXTERNAL-IP>/health
```

---

## 🔄 CI/CD (GitHub Actions)

Pipeline automatizado em 3 jobs:

### 1. test-data-quality (~30s)
- Valida schema dos dados
- Roda 16 testes de qualidade
- Upload processed data como artifact

### 2. train-and-test (~3min)
- Treina modelo
- Valida quality gates (5 métricas)
- Roda testes de modelo e inferência
- Gera metrics report
- Upload modelo como artifact
- Push para DVC remote (se configurado)

### 3. deploy (apenas main, ~30s)
- Cria version tag
- Cria GitHub Release
- Deploy para produção (futuro: K8s)

**Total runtime:** ~4 minutos  
**Custo:** GRATUITO (GitHub Actions free tier)

---

## 🛠️ Comandos Úteis (Makefile)
```bash
# SETUP
make help              # Ver todos os comandos
make setup             # Setup inicial completo
make install           # Instalar dependências

# DATA
make data-download     # Baixar dataset
make data-prepare      # ETL pipeline
make data-validate     # Validar dados

# TESTES
make test              # Todos os testes
make test-data         # Testes de dados
make test-model        # Testes de modelo
make test-api          # Testes de API
make test-cov          # Com coverage

# TREINO
make train             # Treinar modelo
make experiments       # MLflow UI

# DVC
make dvc-repro         # Executar pipeline
make dvc-dag           # Ver DAG
make dvc-metrics       # Ver métricas

# DOCKER
make docker-build      # Build imagem
make docker-test       # Testes no Docker
make docker-train      # Treinar no Docker

# API
make api               # Iniciar API (dev mode)
make api-prod          # Iniciar API (production)
make api-test          # Testes da API

# CI/CD
make ci                # Simular CI/CD localmente
make ci-docker         # CI/CD no Docker

# LIMPEZA
make clean             # Limpar cache
make clean-all         # Limpeza completa

# INFO
make info              # Info do projeto
```

---

## 📚 Conceitos CD4ML Implementados

### 1. Versionamento
```
Código → Git
Dados → DVC (hash MD5)
Modelos → DVC + MLflow Registry
Hiperparâmetros → params.yaml (Git)
Ambiente → requirements.txt + Docker
```

### 2. Testes em ML

Pirâmide de testes adaptada:
```
       ┌─────────────┐
       │  Inference  │  ← Latência, formato (9 tests)
       └─────────────┘
    ┌──────────────────┐
    │  Model Metrics   │  ← Quality gates (8 tests)
    └──────────────────┘
 ┌───────────────────────┐
 │   Data Quality        │  ← Schema, ranges (16 tests)
 └───────────────────────┘
```

### 3. Validação de Dados

**Pandera schemas** garantem:
- Colunas esperadas
- Tipos corretos
- Ranges válidos
- Constraints de negócio

### 4. Quality Gates

Modelo **só é promovido** se:
- ✅ Accuracy ≥ 75%
- ✅ Precision ≥ 73%
- ✅ Recall ≥ 73%
- ✅ F1 ≥ 73%
- ✅ Overfitting gap ≤ 10%

### 5. Experiment Tracking

**MLflow** registra:
- Hiperparâmetros
- Métricas (treino/teste)
- Artifacts (modelo, plots)
- Git commit + timestamp

### 6. Reprodutibilidade
```bash
git checkout <commit-6-meses>
dvc checkout
dvc repro
# → Mesmo resultado garantido!
```

---

## 📖 Documentação Adicional

- **[GLOSSARY.md](GLOSSARY.md)**: Glossário técnico completo (A-Z)
  - 50+ termos explicados
  - Exemplos práticos
  - Analogias
  - Referências

---

## 🔮 API REST (FastAPI)

### Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Root (API info) |
| GET | `/health` | Health check detalhado |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction (max 100) |
| GET | `/model/info` | Model information |

### Exemplo de Uso
```python
import requests

# Single prediction
sample = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

response = requests.post("http://localhost:8000/predict", json=sample)
print(response.json())
# {
#   "prediction": 1,
#   "confidence": 0.87,
#   "probabilities": {"0": 0.13, "1": 0.87},
#   "interpretation": "Good Wine (High confidence: 87.0%)",
#   "latency_ms": 5.23,
#   "timestamp": "2024-12-01T10:30:00"
# }
```

### Documentação Interativa

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📈 Estatísticas do Projeto

| Categoria | Quantidade |
|-----------|------------|
| **Linhas de código Python** | ~3.500 |
| **Arquivos Python** | 18 |
| **Testes automatizados** | 71 |
| **Comandos Makefile** | 50+ |
| **Endpoints API** | 5 |
| **Manifests Kubernetes** | 8 |
| **Coverage** | ~90% |
| **Docker image size** | ~500 MB |
| **Pipeline CI/CD time** | ~4 min |

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

**Certifique-se de:**
- ✅ Passar em todos os testes (`make test-all`)
- ✅ Seguir PEP 8 (`make lint`)
- ✅ Adicionar testes para novas features
- ✅ Atualizar documentação

---

## 📄 Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👤 Autor

**Eric Silva**

- GitHub: [@gokuhayda](https://github.com/gokuhayda)
- LinkedIn: [Eric Silva](https://www.linkedin.com/in/eric-nextgen)
- Projeto: [MyShowCase](https://github.com/gokuhayda/MyShowCase)

---

## 🙏 Agradecimentos

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) pelo dataset
- [DVC](https://dvc.org/) pela ferramenta de versionamento
- [MLflow](https://mlflow.org/) pelo experiment tracking
- [FastAPI](https://fastapi.tiangolo.com/) pelo framework web
- [ThoughtWorks](https://www.thoughtworks.com/) pela inspiração em CD4ML

---

## 📚 Referências

- [CD4ML: Continuous Delivery for Machine Learning](https://martinfowler.com/articles/cd4ml.html)
- [Rules of Machine Learning (Google)](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [MLOps Principles (Microsoft)](https://docs.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment)
- [Hidden Technical Debt in ML Systems (Google)](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

---

<div align="center">

**⭐ Se este projeto foi útil, considere dar uma estrela!**

Made with ❤️ and ☕ for the ML community

**Total de arquivos criados:** 60+  
**Total de linhas de código:** ~5.000+  
**Tempo de desenvolvimento:** Projeto completo production-ready

</div>