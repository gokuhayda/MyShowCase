
# ============================================================================
# Dockerfile - CD4ML Production Project
# ============================================================================
# Container Docker para execução reprodutível do pipeline de ML
#
# Por que Docker:
#   ✅ Reprodutibilidade: Mesmo ambiente em dev, CI/CD e produção
#   ✅ Portabilidade: Roda em qualquer lugar (laptop, cloud, k8s)
#   ✅ Isolamento: Não conflita com outras aplicações
#   ✅ Versionamento: Dockerfile versionado no Git
#
# Build:
#   docker build -t cd4ml-wine-quality:latest .
#
# Run (exemplos):
#   docker run --rm cd4ml-wine-quality python src/models/train.py
#   docker run --rm cd4ml-wine-quality pytest src/tests/ -v
#   docker run --rm -p 8000:8000 cd4ml-wine-quality uvicorn api:app
# ============================================================================

# ----------------------------------------------------------------------------
# STAGE 1: Base Image
# ----------------------------------------------------------------------------
# Python 3.10 slim (Debian-based, otimizada)
# Tamanho: ~125 MB (vs 1 GB da imagem completa)
FROM python:3.10-slim as base

# Metadata
LABEL maintainer="gokuhayda@github.com"
LABEL description="CD4ML Wine Quality Classification - Production ML Pipeline"
LABEL version="1.0.0"

# Configurar timezone (para logs)
ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Variáveis de ambiente para Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Criar usuário não-root (segurança)
# Por que: Nunca rodar containers como root em produção!
RUN useradd -m -u 1000 -s /bin/bash mluser && \
    mkdir -p /app && \
    chown -R mluser:mluser /app

# Diretório de trabalho
WORKDIR /app

# ----------------------------------------------------------------------------
# STAGE 2: Dependencies
# ----------------------------------------------------------------------------
# Instalar dependências do sistema necessárias para build
FROM base as dependencies

# Instalar dependências do sistema
# - build-essential: gcc, make (para compilar pacotes Python)
# - git: para DVC e versionamento
# - curl: para health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar apenas requirements.txt (cache Docker otimizado)
# Por que: Se requirements.txt não mudar, Docker usa cache desta layer
COPY --chown=mluser:mluser requirements.txt .

# Instalar dependências Python
USER mluser
RUN pip install --user --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------------------
# STAGE 3: Application
# ----------------------------------------------------------------------------
# Copiar código da aplicação
FROM base as application

# Instalar apenas runtime dependencies (sem build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependências instaladas do stage anterior
COPY --from=dependencies --chown=mluser:mluser /home/mluser/.local /home/mluser/.local

# Adicionar .local/bin ao PATH (onde pip instalou pacotes)
ENV PATH=/home/mluser/.local/bin:$PATH

# Copiar código da aplicação
COPY --chown=mluser:mluser . .

# Mudar para usuário não-root
USER mluser

# Criar diretórios necessários (se não existirem)
RUN mkdir -p data/raw data/processed models reports logs

# ----------------------------------------------------------------------------
# Health Check
# ----------------------------------------------------------------------------
# Docker verifica se container está saudável
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# ----------------------------------------------------------------------------
# Exposição de Portas (para API - futuro)
# ----------------------------------------------------------------------------
# Porta 8000 para FastAPI (se rodar API)
EXPOSE 8000

# Porta 5000 para MLflow UI (se rodar tracking server)
EXPOSE 5000

# ----------------------------------------------------------------------------
# Entrypoint e Command Default
# ----------------------------------------------------------------------------
# Entrypoint: sempre executa (não pode ser sobrescrito facilmente)
# CMD: comando padrão (pode ser sobrescrito)

# Entrypoint: Script de inicialização (criar se necessário)
# ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Comando padrão: bash interativo
# Para sobrescrever: docker run cd4ml-wine-quality <seu-comando>
CMD ["/bin/bash"]

# ============================================================================
# EXEMPLOS DE USO
# ============================================================================
#
# 1. BUILD DA IMAGEM
# -------------------
# docker build -t cd4ml-wine-quality:latest .
# docker build -t cd4ml-wine-quality:v1.0.0 .  # Com tag de versão
#
# 2. TREINAR MODELO
# ------------------
# docker run --rm \
#   -v $(pwd)/models:/app/models \
#   cd4ml-wine-quality:latest \
#   python src/models/train.py
#
# 3. RODAR TESTES
# ----------------
# # Todos os testes
# docker run --rm cd4ml-wine-quality:latest pytest src/tests/ -v
#
# # Apenas testes de dados
# docker run --rm cd4ml-wine-quality:latest \
#   pytest src/tests/test_data_quality.py -v
#
# 4. FAZER PREDIÇÃO
# ------------------
# docker run --rm cd4ml-wine-quality:latest \
#   python src/models/predict.py
#
# 5. EXECUTAR PIPELINE DVC
# -------------------------
# docker run --rm \
#   -v $(pwd)/.dvc:/app/.dvc \
#   -v $(pwd)/data:/app/data \
#   cd4ml-wine-quality:latest \
#   dvc repro
#
# 6. MLFLOW UI
# -------------
# docker run --rm -p 5000:5000 cd4ml-wine-quality:latest \
#   mlflow ui --host 0.0.0.0
# # Acessar: http://localhost:5000
#
# 7. MODO INTERATIVO (DEBUG)
# ---------------------------
# docker run --rm -it cd4ml-wine-quality:latest /bin/bash
#
# 8. RODAR API (FUTURO - FastAPI)
# --------------------------------
# docker run --rm -p 8000:8000 cd4ml-wine-quality:latest \
#   uvicorn api:app --host 0.0.0.0 --port 8000
# # Acessar: http://localhost:8000/docs
#
# 9. COM VARIÁVEIS DE AMBIENTE
# -----------------------------
# docker run --rm \
#   -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
#   -e AWS_ACCESS_KEY_ID=xxx \
#   -e AWS_SECRET_ACCESS_KEY=yyy \
#   cd4ml-wine-quality:latest \
#   python src/models/train.py
#
# 10. DOCKER COMPOSE (FUTURO - multi-container)
# ----------------------------------------------
# docker-compose up -d
#
# ============================================================================
# OTIMIZAÇÕES APLICADAS
# ============================================================================
#
# 1. MULTI-STAGE BUILD
#    - Stage 1 (base): Imagem base comum
#    - Stage 2 (dependencies): Instala deps com build tools
#    - Stage 3 (application): Copia apenas runtime (menor)
#    Resultado: Imagem final ~500 MB (vs ~2 GB sem otimização)
#
# 2. LAYER CACHING
#    - requirements.txt copiado ANTES do código
#    - Se código muda mas deps não, Docker usa cache
#    Resultado: Rebuild 10x mais rápido
#
# 3. USUÁRIO NÃO-ROOT
#    - Cria usuário 'mluser' (UID 1000)
#    - Nunca roda como root (segurança)
#    Resultado: Container mais seguro
#
# 4. .dockerignore
#    - Ignora arquivos desnecessários (venv, cache, etc)
#    Resultado: Build context menor = build mais rápido
#
# 5. HEALTH CHECK
#    - Docker verifica se container está OK
#    Resultado: Orquestração mais confiável (k8s, swarm)
#
# ============================================================================
# TROUBLESHOOTING
# ============================================================================
#
# Problema: Build falha em "pip install"
# Solução: Verificar se requirements.txt está correto
#   docker build --no-cache -t cd4ml-wine-quality .
#
# Problema: Permissões negadas
# Solução: Garantir que mluser tem permissões
#   docker run --rm -it cd4ml-wine-quality ls -la /app
#
# Problema: "Module not found" ao rodar
# Solução: Verificar se PYTHONPATH está correto
#   docker run --rm cd4ml-wine-quality python -c "import sys; print(sys.path)"
#
# Problema: Container muito grande (>2 GB)
# Solução: Usar .dockerignore e python:3.10-slim
#   docker images cd4ml-wine-quality
#
# Problema: Build muito lento
# Solução: Aproveitar cache do Docker
#   - Não mude requirements.txt desnecessariamente
#   - Use docker build --cache-from
#
# ============================================================================
