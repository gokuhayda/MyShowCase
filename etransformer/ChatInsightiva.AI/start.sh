#!/bin/bash

echo "♻️ Limpando índices antigos em storage/deeplake_local..."
rm -rf ~/chatCulturise.AIv4/storage/deeplake_local/*

echo "🔄 Reiniciando servidor FastAPI com proteção contra segmentation fault..."

# Caminho base do projeto (ajuste se necessário)
cd /root/chatCulturise.AIv4 || exit 1

# Ativa o ambiente virtual
source venv/bin/activate

# Mata qualquer processo anterior na porta 8001
fuser -k 8001/tcp

# Espera liberação da porta
sleep 2

# Exporta variáveis de ambiente que evitam crash
export OMP_NUM_THREADS=1
export MKL_SERVICE_FORCE_INTEL=1

# Inicia o servidor com nohup e log persistente
echo "🚀 Iniciando Uvicorn com logs em chatbot.log..."
nohup python -m uvicorn server:app --host 0.0.0.0 --port 8001 > chatbot.log 2>&1 &

echo "✅ Servidor iniciado com sucesso!"

