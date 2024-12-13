# #!/bin/bash

# # Criar diretórios necessários
# mkdir -p datasets/raw_data
# mkdir -p datasets/processed_texts
# mkdir -p datasets/embeddings

# # Executar ingestão inicial e iniciar watcher
# echo "Executando memory_train.py para ingestão inicial e monitoramento..."
# python app/memory_train.py &

# # Iniciar o servidor do BOT
# echo "Iniciando o BOT..."
# exec python app/chatbot.py
