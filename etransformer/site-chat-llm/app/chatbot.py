import os
import logging
from yaml import safe_load
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from sessions_manager import save_conversation, create_session, load_session_history
from indexing import create_index
from query_index import query_index_rerank
from data_ingestion import scrape_web_pages, process_pdfs_text
from chat_history_manager import manage_history

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Carregar variáveis do arquivo .env
load_dotenv()

# Carregar configuração do arquivo YAML
try:
    with open("./config/config.yaml", "r") as file:
        CONFIG_PATH = safe_load(file)
except Exception as e:
    logger.error(f"Erro ao carregar o arquivo de configuração: {e}")
    exit(1)

SESSIONS_PATH = CONFIG_PATH['storage']['sessions_path']
CONVERSATIONS_PATH = CONFIG_PATH['storage']['conversations_path']

# Configuração do Flask
app = Flask(__name__, template_folder="../templates")

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

CORS(app, resources={r"/*": {"origins": "https://gokuhayda.github.io"}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://gokuhayda.github.io"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

try:
    logger.info("Iniciando processamento de dados e criação do índice...")
    scrape_web_pages(CONFIG_PATH)
    process_pdfs_text(CONFIG_PATH)
    index = create_index(CONFIG_PATH)
    logger.info("Processamento de dados e criação do índice concluídos com sucesso.")
except Exception as e:
    logger.error(f"Erro no processamento inicial: {e}")
    exit(1)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# Rota do chatbot
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat_alternative():
    try:
        if request.method == "OPTIONS":
            return jsonify({"message": "CORS preflight successful"}), 200

        if not request.is_json:
            logger.warning("Requisição inválida: não é um JSON.")
            return jsonify({"error": "Requisição inválida. Esperado JSON."}), 400

        data = request.json
        user_message = data.get("user_message", "").strip()
        session_id = data.get("session_id", create_session(SESSIONS_PATH))

        logger.info(f"Mensagem recebida: {user_message}, Session ID: {session_id}")

        if not user_message:
            logger.warning("Mensagem do usuário não fornecida.")
            return jsonify({"error": "Mensagem não fornecida."}), 400

        # Verificar se o índice suporta `as_query_engine`
        if hasattr(index, 'as_query_engine'):
            bot_response = query_index_rerank(index, user_message, CONFIG_PATH)
        else:
            logger.error("O índice não suporta 'as_query_engine'.")
            return jsonify({"error": "Índice inválido. Verifique a configuração."}), 500

        if not bot_response:
            bot_response = "Desculpe, não consegui entender sua mensagem."

        # Atualizando o histórico utilizando a função de gerenciamento
        save_conversation(session_id, user_message, bot_response, SESSIONS_PATH, CONVERSATIONS_PATH)

        session_history = load_session_history(session_id, SESSIONS_PATH)
        session_history = manage_history(session_history, user_message, bot_response, token_limit=1000, summary_percentage=0.5)

        logger.info(f"Resposta gerada: {bot_response}")
        return jsonify({"session_id": session_id, "response": bot_response}), 200

    except Exception as e:
        logger.error(f"Erro no processamento do chatbot: {e}")
        return jsonify({"error": "Erro ao processar a consulta. Verifique os logs para mais detalhes."}), 500



if __name__ == "__main__":
    # Obtém a porta configurada no ambiente, com fallback para 5000
    port = int(os.getenv("PORT", 5000))

    # Inicia o servidor Flask
    app.run(host="0.0.0.0", port=port)