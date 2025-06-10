# Usando uma imagem base do Python 3.12
FROM python:3.12-slim

# Setar o diretório de trabalho dentro do container
WORKDIR /chatInsightiva.AIv4

# Copiar o conteúdo do seu diretório local para o container
COPY . /chatInsightiva.AIv4

# Defina o PYTHONPATH para incluir o diretório pai de api_server (onde o core está localizado)
ENV PYTHONPATH=/chatInsightiva.AIv4:$PYTHONPATH

# Instalar as dependências diretamente no sistema
RUN pip install --upgrade pip
RUN pip install crewai-tools crewai Flask flask_cors python-dotenv torch scikit-learn deep_translator PyYAML cohere llama-index llama-index-embeddings-openai llama-index-embeddings-huggingface llama-index-llms-openai llama-index-llms-ollama langchain-community langchain-openai llama-index-vector-stores-deeplake langchain-ollama PyMuPDF python-docx langfuse langchain-ollama llama-index-embeddings-langchain streamlit ragas matplotlib uvicorn

# Expor a porta que o FastAPI vai rodar
EXPOSE 8000

# Comando para rodar a aplicação (no caso do Flask ou FastAPI)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
