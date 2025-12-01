
# ============================================================================
# Makefile - CD4ML Production Project
# ============================================================================
# Comandos úteis para desenvolvimento, testes e deploy
#
# Por que Makefile:
#   ✅ Padronização: Todos usam mesmos comandos
#   ✅ Documentação: Lista de comandos disponíveis (make help)
#   ✅ Produtividade: Comandos longos viram "make test"
#   ✅ Onboarding: Novos devs aprendem rápido
#
# Uso:
#   make help          # Ver todos os comandos
#   make test          # Rodar testes
#   make train         # Treinar modelo
#   make docker-build  # Build Docker image
#
# Convenções:
#   - Targets em lowercase com hífens (test-data, docker-build)
#   - Cores ANSI para output legível
#   - .PHONY para targets que não criam arquivos
# ============================================================================

# Configurações
.DEFAULT_GOAL := help
SHELL := /bin/bash
PYTHON := python
PIP := pip
PYTEST := pytest
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# Variáveis de configuração
PROJECT_NAME := cd4ml-wine-quality
DOCKER_IMAGE := $(PROJECT_NAME):latest
DOCKER_IMAGE_TEST := $(PROJECT_NAME):test

# Cores para output (ANSI)
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE := \033[34m
COLOR_CYAN := \033[36m

# ============================================================================
# HELP - Lista de comandos disponíveis
# ============================================================================
.PHONY: help
help: ## 📚 Mostra esta mensagem de ajuda
	@echo ""
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)═══════════════════════════════════════════════════════════════════$(COLOR_RESET)"
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)  🚀 CD4ML Production Project - Makefile Commands$(COLOR_RESET)"
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)═══════════════════════════════════════════════════════════════════$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)Uso:$(COLOR_RESET) make $(COLOR_GREEN)<comando>$(COLOR_RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(COLOR_GREEN)%-20s$(COLOR_RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(COLOR_BOLD)Exemplos:$(COLOR_RESET)"
	@echo "  make setup           # Setup inicial do projeto"
	@echo "  make test            # Rodar todos os testes"
	@echo "  make train           # Treinar modelo"
	@echo "  make docker-build    # Build Docker image"
	@echo ""

# ============================================================================
# SETUP - Configuração inicial do projeto
# ============================================================================
.PHONY: setup
setup: ## 🔧 Setup inicial (venv + deps + data)
	@echo "$(COLOR_BOLD)$(COLOR_BLUE)🔧 Setting up project...$(COLOR_RESET)"
	@$(MAKE) venv
	@$(MAKE) install
	@$(MAKE) data-download
	@echo "$(COLOR_GREEN)✅ Setup completed!$(COLOR_RESET)"

.PHONY: venv
venv: ## 🐍 Criar virtual environment
	@echo "$(COLOR_BLUE)Creating virtual environment...$(COLOR_RESET)"
	@python -m venv venv
	@echo "$(COLOR_GREEN)✅ Virtual environment created!$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)💡 Activate with: source venv/bin/activate$(COLOR_RESET)"

.PHONY: install
install: ## 📦 Instalar dependências
	@echo "$(COLOR_BLUE)Installing dependencies...$(COLOR_RESET)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "$(COLOR_GREEN)✅ Dependencies installed!$(COLOR_RESET)"

.PHONY: install-dev
install-dev: ## 📦 Instalar deps de desenvolvimento
	@echo "$(COLOR_BLUE)Installing dev dependencies...$(COLOR_RESET)"
	@$(PIP) install -r requirements-dev.txt || $(PIP) install pytest pytest-cov black flake8 mypy
	@echo "$(COLOR_GREEN)✅ Dev dependencies installed!$(COLOR_RESET)"

# ============================================================================
# DATA - Gerenciamento de dados
# ============================================================================
.PHONY: data-download
data-download: ## 📥 Baixar dataset
	@echo "$(COLOR_BLUE)Downloading dataset...$(COLOR_RESET)"
	@$(PYTHON) src/data/download_data.py
	@echo "$(COLOR_GREEN)✅ Dataset downloaded!$(COLOR_RESET)"

.PHONY: data-prepare
data-prepare: ## 🔧 Preparar dados (ETL)
	@echo "$(COLOR_BLUE)Running ETL pipeline...$(COLOR_RESET)"
	@$(PYTHON) src/data/make_dataset.py
	@echo "$(COLOR_GREEN)✅ Data prepared!$(COLOR_RESET)"

.PHONY: data-validate
data-validate: ## ✅ Validar qualidade dos dados
	@echo "$(COLOR_BLUE)Validating data quality...$(COLOR_RESET)"
	@$(PYTEST) src/tests/test_data_quality.py -v
	@echo "$(COLOR_GREEN)✅ Data validation passed!$(COLOR_RESET)"

# ============================================================================
# TESTES - Executar diferentes tipos de testes
# ============================================================================
.PHONY: test
test: ## 🧪 Rodar TODOS os testes
	@echo "$(COLOR_BOLD)$(COLOR_BLUE)🧪 Running all tests...$(COLOR_RESET)"
	@$(PYTEST) src/tests/ -v --tb=short
	@echo "$(COLOR_GREEN)✅ All tests passed!$(COLOR_RESET)"

.PHONY: test-data
test-data: ## 🧪 Rodar testes de dados
	@echo "$(COLOR_BLUE)Running data quality tests...$(COLOR_RESET)"
	@$(PYTEST) src/tests/test_data_quality.py -v --tb=short
	@echo "$(COLOR_GREEN)✅ Data tests passed!$(COLOR_RESET)"

.PHONY: test-model
test-model: ## 🧪 Rodar testes de modelo
	@echo "$(COLOR_BLUE)Running model tests...$(COLOR_RESET)"
	@$(PYTEST) src/tests/test_model_metrics.py -v --tb=short
	@echo "$(COLOR_GREEN)✅ Model tests passed!$(COLOR_RESET)"

.PHONY: test-inference
test-inference: ## 🧪 Rodar testes de inferência
	@echo "$(COLOR_BLUE)Running inference tests...$(COLOR_RESET)"
	@$(PYTEST) src/tests/test_inference.py -v --tb=short
	@echo "$(COLOR_GREEN)✅ Inference tests passed!$(COLOR_RESET)"

.PHONY: test-cov
test-cov: ## 📊 Rodar testes com coverage
	@echo "$(COLOR_BLUE)Running tests with coverage...$(COLOR_RESET)"
	@$(PYTEST) src/tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(COLOR_GREEN)✅ Coverage report generated!$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)💡 Open htmlcov/index.html to view report$(COLOR_RESET)"

.PHONY: test-watch
test-watch: ## 👀 Rodar testes em modo watch
	@echo "$(COLOR_BLUE)Running tests in watch mode...$(COLOR_RESET)"
	@$(PYTEST) src/tests/ -v --tb=short -f

# ============================================================================
# TREINO - Pipeline de treinamento
# ============================================================================
.PHONY: train
train: ## 🎓 Treinar modelo
	@echo "$(COLOR_BOLD)$(COLOR_BLUE)🎓 Training model...$(COLOR_RESET)"
	@$(PYTHON) src/models/train.py
	@echo "$(COLOR_GREEN)✅ Model trained!$(COLOR_RESET)"
	@$(MAKE) test-model

.PHONY: predict
predict: ## 🔮 Fazer predições (exemplo)
	@echo "$(COLOR_BLUE)Running prediction example...$(COLOR_RESET)"
	@$(PYTHON) src/models/predict.py
	@echo "$(COLOR_GREEN)✅ Prediction completed!$(COLOR_RESET)"

.PHONY: experiments
experiments: ## 📊 Abrir MLflow UI
	@echo "$(COLOR_BLUE)Starting MLflow UI...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)💡 Access at: http://localhost:5000$(COLOR_RESET)"
	@mlflow ui

# ============================================================================
# DVC - Data Version Control
# ============================================================================
.PHONY: dvc-init
dvc-init: ## 🔧 Inicializar DVC
	@echo "$(COLOR_BLUE)Initializing DVC...$(COLOR_RESET)"
	@dvc init --subdir
	@dvc config core.autostage true
	@echo "$(COLOR_GREEN)✅ DVC initialized!$(COLOR_RESET)"

.PHONY: dvc-repro
dvc-repro: ## 🔄 Reproduzir pipeline DVC
	@echo "$(COLOR_BLUE)Reproducing DVC pipeline...$(COLOR_RESET)"
	@dvc repro
	@echo "$(COLOR_GREEN)✅ Pipeline reproduced!$(COLOR_RESET)"

.PHONY: dvc-dag
dvc-dag: ## 📊 Visualizar DAG do pipeline
	@echo "$(COLOR_BLUE)DVC Pipeline DAG:$(COLOR_RESET)"
	@dvc dag

.PHONY: dvc-metrics
dvc-metrics: ## 📈 Ver métricas do pipeline
	@echo "$(COLOR_BLUE)DVC Metrics:$(COLOR_RESET)"
	@dvc metrics show

.PHONY: dvc-push
dvc-push: ## ⬆️  Push artifacts para DVC remote
	@echo "$(COLOR_BLUE)Pushing to DVC remote...$(COLOR_RESET)"
	@dvc push
	@echo "$(COLOR_GREEN)✅ Artifacts pushed!$(COLOR_RESET)"

.PHONY: dvc-pull
dvc-pull: ## ⬇️  Pull artifacts do DVC remote
	@echo "$(COLOR_BLUE)Pulling from DVC remote...$(COLOR_RESET)"
	@dvc pull
	@echo "$(COLOR_GREEN)✅ Artifacts pulled!$(COLOR_RESET)"

# ============================================================================
# DOCKER - Containerização
# ============================================================================
.PHONY: docker-build
docker-build: ## 🐳 Build Docker image
	@echo "$(COLOR_BOLD)$(COLOR_BLUE)🐳 Building Docker image...$(COLOR_RESET)"
	@$(DOCKER) build -t $(DOCKER_IMAGE) .
	@echo "$(COLOR_GREEN)✅ Docker image built: $(DOCKER_IMAGE)$(COLOR_RESET)"

.PHONY: docker-build-no-cache
docker-build-no-cache: ## 🐳 Build Docker sem cache
	@echo "$(COLOR_BLUE)Building Docker image (no cache)...$(COLOR_RESET)"
	@$(DOCKER) build --no-cache -t $(DOCKER_IMAGE) .
	@echo "$(COLOR_GREEN)✅ Docker image built!$(COLOR_RESET)"

.PHONY: docker-test
docker-test: ## 🧪 Rodar testes no Docker
	@echo "$(COLOR_BLUE)Running tests in Docker...$(COLOR_RESET)"
	@$(DOCKER) run --rm $(DOCKER_IMAGE) pytest src/tests/ -v
	@echo "$(COLOR_GREEN)✅ Docker tests passed!$(COLOR_RESET)"

.PHONY: docker-train
docker-train: ## 🎓 Treinar modelo no Docker
	@echo "$(COLOR_BLUE)Training model in Docker...$(COLOR_RESET)"
	@$(DOCKER) run --rm -v $(PWD)/models:/app/models $(DOCKER_IMAGE) python src/models/train.py
	@echo "$(COLOR_GREEN)✅ Model trained in Docker!$(COLOR_RESET)"

.PHONY: docker-shell
docker-shell: ## 🐚 Shell interativo no Docker
	@echo "$(COLOR_BLUE)Starting Docker shell...$(COLOR_RESET)"
	@$(DOCKER) run --rm -it $(DOCKER_IMAGE) /bin/bash

.PHONY: docker-mlflow
docker-mlflow: ## 📊 MLflow UI no Docker
	@echo "$(COLOR_BLUE)Starting MLflow UI in Docker...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)💡 Access at: http://localhost:5000$(COLOR_RESET)"
	@$(DOCKER) run --rm -p 5000:5000 -v $(PWD)/mlruns:/app/mlruns $(DOCKER_IMAGE) mlflow ui --host 0.0.0.0

.PHONY: docker-clean
docker-clean: ## 🧹 Limpar imagens Docker
	@echo "$(COLOR_BLUE)Cleaning Docker images...$(COLOR_RESET)"
	@$(DOCKER) rmi $(DOCKER_IMAGE) 2>/dev/null || true
	@$(DOCKER) rmi $(DOCKER_IMAGE_TEST) 2>/dev/null || true
	@$(DOCKER) system prune -f
	@echo "$(COLOR_GREEN)✅ Docker images cleaned!$(COLOR_RESET)"

# ============================================================================
# DOCKER COMPOSE - Orquestração
# ============================================================================
.PHONY: compose-up
compose-up: ## 🐙 Iniciar serviços (docker-compose)
	@echo "$(COLOR_BLUE)Starting services with docker-compose...$(COLOR_RESET)"
	@$(DOCKER_COMPOSE) up -d
	@echo "$(COLOR_GREEN)✅ Services started!$(COLOR_RESET)"
	@$(DOCKER_COMPOSE) ps

.PHONY: compose-down
compose-down: ## 🐙 Parar serviços (docker-compose)
	@echo "$(COLOR_BLUE)Stopping services...$(COLOR_RESET)"
	@$(DOCKER_COMPOSE) down
	@echo "$(COLOR_GREEN)✅ Services stopped!$(COLOR_RESET)"

.PHONY: compose-logs
compose-logs: ## 📋 Ver logs (docker-compose)
	@$(DOCKER_COMPOSE) logs -f

.PHONY: compose-build
compose-build: ## 🔨 Rebuild serviços (docker-compose)
	@echo "$(COLOR_BLUE)Rebuilding services...$(COLOR_RESET)"
	@$(DOCKER_COMPOSE) build
	@echo "$(COLOR_GREEN)✅ Services rebuilt!$(COLOR_RESET)"

# ============================================================================
# CODE QUALITY - Linters e formatação
# ============================================================================
.PHONY: lint
lint: ## 🔍 Rodar linters (flake8)
	@echo "$(COLOR_BLUE)Running linters...$(COLOR_RESET)"
	@flake8 src/ --max-line-length=100 --exclude=venv,__pycache__
	@echo "$(COLOR_GREEN)✅ Linting passed!$(COLOR_RESET)"

.PHONY: format
format: ## ✨ Formatar código (black)
	@echo "$(COLOR_BLUE)Formatting code...$(COLOR_RESET)"
	@black src/ --line-length=100
	@echo "$(COLOR_GREEN)✅ Code formatted!$(COLOR_RESET)"

.PHONY: format-check
format-check: ## ✅ Verificar formatação
	@echo "$(COLOR_BLUE)Checking code format...$(COLOR_RESET)"
	@black src/ --check --line-length=100
	@echo "$(COLOR_GREEN)✅ Format check passed!$(COLOR_RESET)"

.PHONY: type-check
type-check: ## 🔍 Type checking (mypy)
	@echo "$(COLOR_BLUE)Running type checker...$(COLOR_RESET)"
	@mypy src/ --ignore-missing-imports
	@echo "$(COLOR_GREEN)✅ Type checking passed!$(COLOR_RESET)"

.PHONY: quality
quality: format lint type-check ## ✨ Rodar todos os checks de qualidade
	@echo "$(COLOR_GREEN)✅ All quality checks passed!$(COLOR_RESET)"

# ============================================================================
# LIMPEZA - Remover arquivos temporários
# ============================================================================
.PHONY: clean
clean: ## 🧹 Limpar cache e arquivos temporários
	@echo "$(COLOR_BLUE)Cleaning temporary files...$(COLOR_RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@rm -rf htmlcov/ .tox/ dist/ build/ 2>/dev/null || true
	@echo "$(COLOR_GREEN)✅ Cleanup completed!$(COLOR_RESET)"

.PHONY: clean-data
clean-data: ## 🧹 Limpar dados (CUIDADO!)
	@echo "$(COLOR_YELLOW)⚠️  This will delete all data files!$(COLOR_RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/raw/* data/processed/* models/*.pkl models/*.json; \
		echo "$(COLOR_GREEN)✅ Data cleaned!$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_YELLOW)Cancelled.$(COLOR_RESET)"; \
	fi

.PHONY: clean-all
clean-all: clean clean-data docker-clean ## 🧹 Limpeza completa (código + data + docker)
	@echo "$(COLOR_GREEN)✅ Complete cleanup finished!$(COLOR_RESET)"

# ============================================================================
# GIT - Comandos úteis de Git
# ============================================================================
.PHONY: git-status
git-status: ## 📊 Git status formatado
	@echo "$(COLOR_BLUE)Git Status:$(COLOR_RESET)"
	@git status -sb

.PHONY: git-log
git-log: ## 📜 Git log formatado
	@git log --oneline --graph --decorate --all -10

.PHONY: git-diff
git-diff: ## 📊 Git diff
	@git diff

# ============================================================================
# CI/CD - Simular pipeline CI/CD localmente
# ============================================================================
.PHONY: ci
ci: ## 🚀 Simular CI/CD localmente
	@echo "$(COLOR_BOLD)$(COLOR_BLUE)🚀 Running CI/CD pipeline locally...$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)Step 1/4: Data Quality Tests$(COLOR_RESET)"
	@$(MAKE) test-data
	@echo ""
	@echo "$(COLOR_BOLD)Step 2/4: Train Model$(COLOR_RESET)"
	@$(MAKE) train
	@echo ""
	@echo "$(COLOR_BOLD)Step 3/4: Model Tests$(COLOR_RESET)"
	@$(MAKE) test-model
	@echo ""
	@echo "$(COLOR_BOLD)Step 4/4: Inference Tests$(COLOR_RESET)"
	@$(MAKE) test-inference
	@echo ""
	@echo "$(COLOR_GREEN)$(COLOR_BOLD)✅ CI/CD pipeline completed successfully!$(COLOR_RESET)"

.PHONY: ci-docker
ci-docker: ## 🚀 Simular CI/CD no Docker
	@echo "$(COLOR_BOLD)$(COLOR_BLUE)🚀 Running CI/CD in Docker...$(COLOR_RESET)"
	@$(MAKE) docker-build
	@$(MAKE) docker-test
	@$(MAKE) docker-train
	@echo "$(COLOR_GREEN)$(COLOR_BOLD)✅ Docker CI/CD completed!$(COLOR_RESET)"

# ============================================================================
# UTILITÁRIOS - Comandos de diagnóstico
# ============================================================================
.PHONY: info
info: ## ℹ️  Mostrar informações do projeto
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)═══════════════════════════════════════════$(COLOR_RESET)"
	@echo "$(COLOR_BOLD)  📊 Project Information$(COLOR_RESET)"
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)═══════════════════════════════════════════$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)Project:$(COLOR_RESET)       $(PROJECT_NAME)"
	@echo "$(COLOR_BOLD)Python:$(COLOR_RESET)        $$(python --version 2>&1)"
	@echo "$(COLOR_BOLD)Pip:$(COLOR_RESET)           $$(pip --version 2>&1 | cut -d' ' -f1-2)"
	@echo "$(COLOR_BOLD)Pytest:$(COLOR_RESET)        $$(pytest --version 2>&1 | head -1)"
	@echo "$(COLOR_BOLD)DVC:$(COLOR_RESET)           $$(dvc version 2>&1 | head -1)"
	@echo "$(COLOR_BOLD)Docker:$(COLOR_RESET)        $$(docker --version 2>&1)"
	@echo "$(COLOR_BOLD)Git Branch:$(COLOR_RESET)    $$(git branch --show-current 2>&1)"
	@echo "$(COLOR_BOLD)Git Commit:$(COLOR_RESET)    $$(git rev-parse --short HEAD 2>&1)"
	@echo ""
	@echo "$(COLOR_BOLD)Files:$(COLOR_RESET)"
	@echo "  Data:   $$(find data -type f 2>/dev/null | wc -l) files"
	@echo "  Models: $$(find models -type f 2>/dev/null | wc -l) files"
	@echo "  Tests:  $$(find src/tests -name 'test_*.py' 2>/dev/null | wc -l) files"
	@echo ""

.PHONY: check-deps
check-deps: ## ✅ Verificar dependências instaladas
	@echo "$(COLOR_BLUE)Checking dependencies...$(COLOR_RESET)"
	@$(PYTHON) -c "import pandas; import numpy; import sklearn; import mlflow; import dvc; import pytest; print('$(COLOR_GREEN)✅ All core dependencies installed!$(COLOR_RESET)')"

.PHONY: tree
tree: ## 🌲 Mostrar estrutura do projeto
	@echo "$(COLOR_BLUE)Project Structure:$(COLOR_RESET)"
	@tree -L 3 -I 'venv|__pycache__|*.pyc|.git|.dvc/cache|mlruns' || ls -R | grep ":$$" | sed -e 's/:$$//' -e 's/[^-][^\/]*\//--/g' -e 's/^/   /' -e 's/-/|/'


# ============================================================================
# API - FastAPI
# ============================================================================
.PHONY: api
api: ## 🚀 Iniciar API (dev mode com reload)
	@echo "$(COLOR_BOLD)$(COLOR_BLUE)🚀 Starting FastAPI...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)💡 API: http://localhost:8000$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)💡 Docs: http://localhost:8000/docs$(COLOR_RESET)"
	@python run_api.py

.PHONY: api-prod
api-prod: ## 🚀 Iniciar API (production mode)
	@echo "$(COLOR_BLUE)Starting API (production)...$(COLOR_RESET)"
	@uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

.PHONY: api-test
api-test: ## 🧪 Testar API
	@echo "$(COLOR_BLUE)Testing API...$(COLOR_RESET)"
	@pytest api/tests/ -v --tb=short

.PHONY: api-docs
api-docs: ## 📚 Abrir documentação da API
	@echo "$(COLOR_BLUE)Opening API docs...$(COLOR_RESET)"
	@python -m webbrowser http://localhost:8000/docs

# ============================================================================
# API TESTS - Testes da API
# ============================================================================
.PHONY: test-api
test-api: ## 🧪 Rodar testes da API
	@echo "$(COLOR_BLUE)Running API tests...$(COLOR_RESET)"
	@$(PYTEST) api/tests/ -v --tb=short
	@echo "$(COLOR_GREEN)✅ API tests passed!$(COLOR_RESET)"

.PHONY: test-api-unit
test-api-unit: ## 🧪 Rodar apenas testes unitários da API
	@echo "$(COLOR_BLUE)Running API unit tests...$(COLOR_RESET)"
	@$(PYTEST) api/tests/test_endpoints.py api/tests/test_validation.py -v

.PHONY: test-api-integration
test-api-integration: ## 🧪 Rodar testes de integração da API
	@echo "$(COLOR_BLUE)Running API integration tests...$(COLOR_RESET)"
	@$(PYTEST) api/tests/test_integration.py -v

.PHONY: test-api-performance
test-api-performance: ## 🧪 Rodar testes de performance da API
	@echo "$(COLOR_BLUE)Running API performance tests...$(COLOR_RESET)"
	@$(PYTEST) api/tests/test_performance.py -v

.PHONY: test-api-cov
test-api-cov: ## 📊 Rodar testes da API com coverage
	@echo "$(COLOR_BLUE)Running API tests with coverage...$(COLOR_RESET)"
	@$(PYTEST) api/tests/ -v --cov=api --cov-report=html --cov-report=term
	@echo "$(COLOR_GREEN)✅ Coverage report generated!$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)💡 Open htmlcov/index.html$(COLOR_RESET)"

.PHONY: test-all
test-all: ## 🧪 Rodar TODOS os testes (src + api)
	@echo "$(COLOR_BOLD)$(COLOR_BLUE)🧪 Running ALL tests...$(COLOR_RESET)"
	@$(PYTEST) src/tests/ api/tests/ -v --tb=short
	@echo "$(COLOR_GREEN)$(COLOR_BOLD)✅ All tests passed!$(COLOR_RESET)"
	
# ============================================================================
# DOCUMENTAÇÃO - Gerar documentação
# ============================================================================
.PHONY: docs
docs: ## 📚 Gerar documentação (futuro)
	@echo "$(COLOR_YELLOW)📚 Documentation generation not yet implemented$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)💡 Future: Sphinx or MkDocs$(COLOR_RESET)"

# ============================================================================
# BENCHMARKS - Testes de performance
# ============================================================================
.PHONY: benchmark
benchmark: ## ⚡ Benchmark de inferência
	@echo "$(COLOR_BLUE)Running inference benchmark...$(COLOR_RESET)"
	@$(PYTHON) -m timeit -n 1000 -s "from src.models.predict import WineQualityPredictor; p = WineQualityPredictor(); sample = {'fixed_acidity': 7.4, 'volatile_acidity': 0.7, 'citric_acid': 0.0, 'residual_sugar': 1.9, 'chlorides': 0.076, 'free_sulfur_dioxide': 11.0, 'total_sulfur_dioxide': 34.0, 'density': 0.9978, 'pH': 3.51, 'sulphates': 0.56, 'alcohol': 9.4}" "p.predict(sample)"
	@echo "$(COLOR_GREEN)✅ Benchmark completed!$(COLOR_RESET)"

# ============================================================================
# ATALHOS - Aliases comuns
# ============================================================================
.PHONY: t
t: test ## Alias para 'make test'

.PHONY: r
r: train ## Alias para 'make train'

.PHONY: d
d: docker-build ## Alias para 'make docker-build'

# ============================================================================
# NOTAS
# ============================================================================
# - Todos os targets com .PHONY não criam arquivos
# - Cores ANSI funcionam em Linux/Mac (Windows WSL também)
# - Para desabilitar cores: make <target> 2>&1 | cat
# - Para debug: make -n <target> (dry-run)
# ============================================================================

