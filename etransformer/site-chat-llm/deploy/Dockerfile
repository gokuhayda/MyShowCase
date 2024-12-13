# Use uma imagem base do Python
FROM python:3.9-slim

# Configurar diretório de trabalho
WORKDIR /app

# Copiar dependências e instalá-las
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante do código
COPY . .

# Expor a porta
EXPOSE 5000

# Comando para iniciar o aplicativo
CMD ["python", "app/chatbot.py"]
