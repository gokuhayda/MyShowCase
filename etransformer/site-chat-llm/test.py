from dotenv import load_dotenv
import os

# Carregar variáveis do .env
load_dotenv("/home/hayda/Documentos/ssd/site-chat-llm/.env")

# Verificar se a chave está configurada
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("A variável OPENAI_API_KEY não está configurada ou acessível.")
