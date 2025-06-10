import json
import os
import uuid
from datetime import datetime, timezone

def create_session(SESSIONS_PATH: str, makedirs: bool = False) -> str:
    """
    Cria uma nova sessão única baseada em UUID e salva no caminho especificado.

    Args:
        SESSIONS_PATH (str): Caminho onde os arquivos de sessão serão salvos.
        makedirs (bool): Se True, cria os diretórios automaticamente.

    Returns:
        str: ID da sessão criada.
    """
    if makedirs:
        os.makedirs(SESSIONS_PATH, exist_ok=True)

    session_id = str(uuid.uuid4())
    session_data = {
        "session_id": session_id,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "history": []
    }

    session_file = os.path.join(SESSIONS_PATH, f"{session_id}.json")
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=4)

    print(f"✅ Nova sessão criada: {session_id}")
    return session_id

def save_conversation(session_id: str, user_message: str, bot_response: str,
                      SESSIONS_PATH: str, CONVERSATIONS_PATH: str, makedirs: bool = False) -> None:
    """
    Atualiza o histórico de uma sessão com uma nova interação e registra no log geral.

    Args:
        session_id (str): ID da sessão.
        user_message (str): Mensagem enviada pelo usuário.
        bot_response (str): Resposta do bot.
        SESSIONS_PATH (str): Caminho onde os arquivos de sessão estão.
        CONVERSATIONS_PATH (str): Caminho para salvar o log geral.
        makedirs (bool): Se True, cria os diretórios automaticamente.
    """
    if makedirs:
        os.makedirs(SESSIONS_PATH, exist_ok=True)
        os.makedirs(CONVERSATIONS_PATH, exist_ok=True)

    session_file = os.path.join(SESSIONS_PATH, f"{session_id}.json")

    if not os.path.exists(session_file):
        raise FileNotFoundError(f"❌ Sessão '{session_id}' não encontrada em {SESSIONS_PATH}.")

    with open(session_file, "r", encoding="utf-8") as f:
        session_data = json.load(f)

    session_data.setdefault("history", [])

    session_data["history"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_message": user_message,
        "bot_response": bot_response
    })

    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=4)

    conversation_log = os.path.join(CONVERSATIONS_PATH, "conversation_log.json")

    log_entry = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_message": user_message,
        "bot_response": bot_response
    }

    # Grava no log geral de conversas
    with open(conversation_log, "a", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False)
        f.write("\n")

def load_session_history(session_id: str, SESSIONS_PATH: str):
    """
    Carrega o histórico da sessão com base no ID informado.

    Args:
        session_id (str): ID da sessão.
        SESSIONS_PATH (str): Caminho dos arquivos de sessão.

    Returns:
        list: Lista de mensagens trocadas na sessão.
    """
    session_file = os.path.join(SESSIONS_PATH, f"{session_id}.json")

    if not os.path.exists(session_file):
        raise FileNotFoundError(f"❌ Sessão '{session_id}' não encontrada.")

    with open(session_file, "r", encoding="utf-8") as f:
        session_data = json.load(f)

    return session_data.get("history", [])

# ------------------------------
# Lógica adicional para controle de mensagens por sessão (fila)
# ------------------------------

from queue import Queue

_sessions_state = {}

def get_or_create_session_state(user_id):
    if user_id not in _sessions_state:
        _sessions_state[user_id] = {
            "message_queue": Queue(),
            "processing": False
        }
    return _sessions_state[user_id]

def enqueue_message(user_id, message):
    session = get_or_create_session_state(user_id)
    session["message_queue"].put(message)

def get_next_message(user_id):
    session = get_or_create_session_state(user_id)
    if not session["message_queue"].empty():
        return session["message_queue"].get()
    return None

def is_processing(user_id):
    return get_or_create_session_state(user_id)["processing"]

def set_processing(user_id, status):
    get_or_create_session_state(user_id)["processing"] = status