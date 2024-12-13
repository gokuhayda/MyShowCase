### Gerenciamento de Sessões e Histórico de Conversas
import json
from datetime import datetime
import uuid
import os

def create_session(SESSIONS_PATH, makedirs=False):
    """
    Cria uma sessão única para cada usuário usando UUID.
    """
    if makedirs:
        os.makedirs(SESSIONS_PATH, exist_ok=True)
    
    session_id = str(uuid.uuid4())
    session_data = {
        "session_id": session_id,
        "start_time": datetime.utcnow().isoformat(),
        "history": []
    }
    session_file = os.path.join(SESSIONS_PATH, f"{session_id}.json")
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(session_data, f)
    print(f"Nova sessão criada: {session_id}")
    return session_id

def save_conversation(session_id, user_message, bot_response, SESSIONS_PATH, CONVERSATIONS_PATH, makedirs=False):
    """
    Salva a conversa no histórico da sessão e no arquivo.
    """
    if makedirs:
        os.makedirs(CONVERSATIONS_PATH, exist_ok=True)
    
    session_file = os.path.join(SESSIONS_PATH, f"{session_id}.json")
    if not os.path.exists(session_file):
        raise FileNotFoundError(f"Sessão {session_id} não encontrada.")

    with open(session_file, "r", encoding="utf-8") as f:
        session_data = json.load(f)

    # Verificar se a estrutura do histórico está correta
    if "history" not in session_data:
        session_data["history"] = []

    # Atualizar histórico
    session_data["history"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "user_message": user_message,
        "bot_response": bot_response
    })

    # Salvar histórico atualizado
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=4)

    # Salvar conversa no log geral (não deletar o arquivo, apenas adicionar novo conteúdo)
    conversation_log = os.path.join(CONVERSATIONS_PATH, "conversation_log.json")
    
    # Verifica se o arquivo de log existe. Se existir, adiciona uma nova linha, senão cria o arquivo
    if os.path.exists(conversation_log):
        with open(conversation_log, "a", encoding="utf-8") as f:
            json.dump({
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_message": user_message,
                "bot_response": bot_response
            }, f, ensure_ascii=False)
            f.write("\n")  # Garante que as conversas são registradas em linhas separadas
    else:
        with open(conversation_log, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_message": user_message,
                "bot_response": bot_response
            }, f, ensure_ascii=False)
            f.write("\n")

def load_session_history(session_id, SESSIONS_PATH):
    session_file = os.path.join(SESSIONS_PATH, f"{session_id}.json")
    if not os.path.exists(session_file):
        raise FileNotFoundError(f"Sessão {session_id} não encontrada.")
    
    with open(session_file, "r", encoding="utf-8") as f:
        session_data = json.load(f)
    
    return session_data["history"]

