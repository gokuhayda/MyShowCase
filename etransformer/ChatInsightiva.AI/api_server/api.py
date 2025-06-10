import sys
sys.path.append('/chatInsightiva.AI')  # Adiciona o diretório raiz ao caminho de importação
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from uuid import uuid4
import os
from core.query_router import MultiIndexFAQRouter  # Importando o roteador
from sessions_manager import save_conversation, load_session_history  # Para gerenciar o histórico de sessão
import uuid
from utils_tools.config_loader import load_config
from utils_tools.model_logger import export_log_to_json
import logging
from dotenv import load_dotenv


# Carregar variáveis do arquivo .env (caso exista)
load_dotenv()

# Inicialização da API FastAPI
app = FastAPI()

# Gerar a API Key automaticamente se não houver configuração manual no .env
API_KEY = os.getenv('API_KEY', str(uuid4()))  # Se não tiver a chave no .env, gera uma nova com uuid4()

# Exibir a chave gerada (para fins de desenvolvimento, remova isso em produção)
print(f"Sua API Key é: {API_KEY}")

# Função para verificar a validade da API Key
def api_key_is_valid(api_key: str):
    return api_key == API_KEY

# Carregar a configuração
config = load_config()

# Inicializar o roteador do chatbot com as configurações
router = MultiIndexFAQRouter(
    salutations=config['salutations'],
    welcome_messages=config['welcome_messages'],
    notification_warning=config['notification_warning_bot']
)

class Query(BaseModel):
    pergunta: str
    session_id: str = None  # Adicionando session_id para manter o histórico

@app.post("/chat")
def chat(query: Query, api_key: str = Header(...)):  # Recebe a API Key no cabeçalho
    """
    Endpoint para processar as perguntas com base no histórico da sessão.
    """
    # Verificar a validade da API Key
    if not api_key_is_valid(api_key):
        raise HTTPException(status_code=403, detail="API Key inválida")  # Resposta para chave inválida

    user_message = query.pergunta.strip()

    if not user_message:
        return {"error": "Mensagem do usuário não fornecida"}

    # Criar um session_id caso não tenha sido fornecido
    if not query.session_id or not is_valid_uuid(query.session_id):
        session_id = str(uuid.uuid4())  # Gera um novo UUID para a sessão
    else:
        session_id = query.session_id

    # Carregar o histórico da sessão, se disponível
    session_history = load_session_history(session_id, config['storage']['sessions_path'])

    # Processar a resposta usando o router
    response_obj = router.responder(user_message, session_history=session_history)

    # Obter a resposta gerada
    response = None
    if hasattr(response_obj, "resposta"):
        response = response_obj.resposta or response_obj.comentario
    elif hasattr(response_obj, "final_output"):
        response = response_obj.final_output
    else:
        response = str(response_obj)

    # Salvar a conversa no histórico
    save_conversation(session_id, user_message, response, config['storage']['sessions_path'], config['storage']['conversations_path'], append=True)

    return {"session_id": session_id, "response": response}

def is_valid_uuid(val):
    """
    Verifica se o valor fornecido é um UUID válido.
    """
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False
