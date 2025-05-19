#!/bin/bash

echo "🔵 Reiniciando Chatbot FastAPI..."

# Entra na pasta do projeto
cd /root/chatCulturise.AIv4 || exit 1

# Ativa o ambiente virtual apenas se não estiver ativo
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "🧠 Ativando ambiente virtual..."
    source venv/bin/activate
else
    echo "🧠 Ambiente virtual já está ativo."
fi

# Mata processos antigos na porta 8001
echo "🛑 Matando possíveis processos antigos na porta 8001..."
fuser -k 8001/tcp || true

# Espera 2 segundos para garantir que liberou
sleep 2

# Limpa storage/deeplake_local
echo "🧹 Limpando storage/deeplake_local..."
rm -rf storage/deeplake_local/*
mkdir -p storage/deeplake_local

# Inicia o servidor com nohup
echo "🟢 Iniciando servidor na porta 8001..."
nohup python -m uvicorn server:app --host 0.0.0.0 --port 8001 > chatbot.log 2>&1 &

echo "✅ Chatbot reiniciado! Logs disponíveis em chatbot.log"

