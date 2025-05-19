# rm -rf storage/deeplake_local/*

import os
import logging
import random
import shutil
from difflib import get_close_matches
from flask import Flask, request, jsonify, render_template_string
from chat_history_manager import load_history_from_file
from flask_cors import CORS
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import atexit
from utils_tools.model_logger import export_log_to_json
os.makedirs("logs", exist_ok=True)
import uuid
from sessions_manager import save_conversation, create_session, load_session_history, enqueue_message, is_processing, set_processing, get_next_message
from chat_history_manager import manage_history
from utils_tools.config_loader import load_config
from core.query_router import MultiIndexFAQRouter
import os
os.environ["CREWAI_TELEMETRY_DISABLED"] = "1"

def is_saudacao(msg, lista_salutations, threshold=0.8):

    """
    Verifica se a mensagem do usuário é uma saudação.

    Parâmetros:
        msg (str): Mensagem enviada pelo usuário.
        lista_salutations (list): Lista de expressões de saudação.
        threshold (float): Limiar de similaridade.

    Retorna:
        bool: True se corresponder, False caso contrário.
    """
    msg_clean = msg.strip().lower()
    match = get_close_matches(msg_clean, lista_salutations, n=1, cutoff=threshold)
    return bool(match)

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Carregar variáveis do .env
load_dotenv()

# Carregar configuração do YAML
try:
    config = load_config()
except Exception as e:
    logger.error(f"Erro ao carregar o arquivo de configuração: {e}")
    exit(1)

# Caminho do DeepLake
vector_store_path = config["storage"]["vector_store_path"]

# Limpar o diretório
if os.path.exists(vector_store_path):
    shutil.rmtree(vector_store_path)
    print(f"🧹 Diretório {vector_store_path} limpo com sucesso.")
else:
    print(f"📂 Diretório {vector_store_path} não existe ainda.")
    
# Caminhos e mensagens
SESSIONS_PATH = config['storage']['sessions_path']
CONVERSATIONS_PATH = config['storage']['conversations_path']
NOTIFICATION_WARNING = config['notification_warning_bot']
WELCOME_MESSAGES = config['welcome_messages']
SALUTATIONS = config['salutations']
REFORCO = config["messages"].get("prompt_reforco", "")

# App Flask
app = Flask(__name__, template_folder="../templates")
CORS(app, resources={r"/*": {"origins": "https://gokuhayda.github.io"}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://gokuhayda.github.io"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# Pré-processamento
import traceback

router = MultiIndexFAQRouter(
    salutations=SALUTATIONS,
    welcome_messages=WELCOME_MESSAGES,
    notification_warning=NOTIFICATION_WARNING
)

@atexit.register

def salvar_logs_ao_encerrar():
    """
    Função executada ao encerrar o app para salvar os logs em JSON.
    """
    export_log_to_json("logs/chatbot_model_calls_log.json")
    
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat_alternative():
    """
    Endpoint principal que processa a interação com o chatbot.
    """
    from chat_history_manager import save_history_to_file 
    try:
        if request.method == "OPTIONS":
            return jsonify({"message": "CORS preflight successful"}), 200

        if not request.is_json:
            logger.warning("Requisição inválida: não é um JSON.")
            return jsonify({"error": "Requisição inválida. Esperado JSON."}), 400

        data = request.json
        user_message = data.get("user_message", "").strip()

        def is_valid_uuid(val):
            try:
                uuid.UUID(str(val))
                return True
            except ValueError:
                return False

        session_id = data.get("session_id")
        if not session_id or not is_valid_uuid(session_id):
            session_id = create_session(SESSIONS_PATH, makedirs=True)

        logger.info(f"Mensagem recebida: {user_message}, Session ID: {session_id}")

        if not user_message:
            logger.warning("Mensagem do usuário não fornecida.")
            return jsonify({"error": "Mensagem não fornecida."}), 400

        if is_processing(session_id):
            return jsonify({"response": "Aguarde, estou processando sua pergunta anterior..."})

        set_processing(session_id, True)

        # Carrega o histórico da sessão antes de responder
        session_history = load_session_history(session_id, SESSIONS_PATH)

        if is_saudacao(user_message, SALUTATIONS):
            response = random.choice(WELCOME_MESSAGES)
        else:
            user_message = f"{user_message}{REFORCO}"
            response_obj = router.responder(user_message, session_history=session_history)
            logger.debug(f"🔍 Resposta bruta recebida: {repr(response_obj)}")
            if hasattr(response_obj, "resposta"):
                response = response_obj.resposta or response_obj.comentario
            elif hasattr(response_obj, "final_output"):
                response = response_obj.final_output
            else:
                response = str(response_obj)

        set_processing(session_id, False)

        save_conversation(session_id, user_message, response, SESSIONS_PATH, CONVERSATIONS_PATH, True)

        # Atualiza e resume histórico
        if session_history and session_history[-1]["user_message"] == user_message and session_history[-1]["bot_response"] == response:
            pass  # já está no histórico
        else:
            session_history = manage_history(session_history, user_message, response, token_limit=1000, summary_percentage=0.5)

        save_history_to_file(session_history, filepath=f"{LOG_HISTORY_PATH}/historico_{session_id}.json")

        logger.info(f"Resposta gerada: {response}")
        return jsonify({"session_id": session_id, "response": str(response)}), 200

    except Exception as e:
        logger.exception("Erro no processamento do chatbot")
        return jsonify({"error": "Erro ao processar a consulta. Verifique os logs para mais detalhes."}), 500

LOG_HISTORY_PATH = "logs_sessions"
os.makedirs(LOG_HISTORY_PATH, exist_ok=True)

@app.route("/historico_html/<session_id>")
def historico_em_html(session_id):
    """
    Exibe o histórico da sessão em HTML com opções de exportar/copiar.
    """
    filepath = f"{LOG_HISTORY_PATH}/historico_{session_id}.json"
    if not os.path.exists(filepath):
        return f"<h3>Histórico da sessão <code>{session_id}</code> não encontrado.</h3>", 404

    try:
        historico = load_history_from_file(filepath)

        html_template = """
        <html>
        <head>
            <title>Histórico da Sessão {{ session_id }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; }
                .chat-box { margin-bottom: 20px; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 0 6px #ccc; }
                .user { color: #2c3e50; font-weight: bold; }
                .bot { color: #16a085; font-weight: bold; }
                .msg { margin: 5px 0 15px 0; }
                .top-bar { margin-bottom: 20px; }
                button {
                    background-color: #16a085;
                    border: none;
                    color: white;
                    padding: 10px 16px;
                    border-radius: 6px;
                    cursor: pointer;
                    margin-right: 10px;
                }
                button:hover {
                    background-color: #138d75;
                }
                code { background-color: #eee; padding: 2px 4px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="top-bar">
                <h2>Histórico da Sessão <code>{{ session_id }}</code></h2>
                <button onclick="exportarTXT()">Exportar como .txt</button>
                <button onclick="copiarParaArea()">Copiar conversa</button>
            </div>

            <div id="conversa">
                {% for item in historico %}
                    <div class="chat-box">
                        <div class="user">Usuário:</div>
                        <div class="msg">{{ item.user_message }}</div>
                        <div class="bot">Assistente:</div>
                        <div class="msg">{{ item.bot_response }}</div>
                    </div>
                {% endfor %}
            </div>

            <script>
                function exportarTXT() {
                    let texto = "";
                    document.querySelectorAll(".chat-box").forEach(box => {
                        const user = box.querySelector(".msg:nth-of-type(1)").innerText;
                        const bot = box.querySelector(".msg:nth-of-type(2)").innerText;
                        texto += "Usuário: " + user + "\n";
                        texto += "Assistente: " + bot + "\n\n";
                    });

                    const blob = new Blob([texto], { type: 'text/plain' });
                    const link = document.createElement("a");
                    link.href = URL.createObjectURL(blob);
                    link.download = "historico_{{ session_id }}.txt";
                    link.click();
                }

                function copiarParaArea() {
                    let texto = "";
                    document.querySelectorAll(".chat-box").forEach(box => {
                        const user = box.querySelector(".msg:nth-of-type(1)").innerText;
                        const bot = box.querySelector(".msg:nth-of-type(2)").innerText;
                        texto += "Usuário: " + user + "\n";
                        texto += "Assistente: " + bot + "\n\n";
                    });
                    navigator.clipboard.writeText(texto).then(() => {
                        alert("Conversa copiada para a área de transferência!");
                    });
                }
            </script>
        </body>
        </html>
        """

        return render_template_string(html_template, session_id=session_id, historico=historico)
    except Exception as e:
        logger.exception("Erro ao renderizar HTML")
        return f"<h3>Erro ao processar histórico: {str(e)}</h3>", 500
        
# ✅ Rota base para lidar com OPTIONS / e GET /
@app.route("/", methods=["GET", "OPTIONS"])
def root():
    """
    Rota base para verificação de status do chatbot.
    """
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight OK"}), 200
    return jsonify({"message": "API do chatbot está online"}), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="127.0.0.1", port=port)

# curl -X POST https://d0ca-2804-868-d041-58e1-cab7-64e3-c8d1-bf8.ngrok-free.app/chat \
#   -H "Content-Type: application/json" \
#   -d '{
#     "session_id": "20860588-68a1-44a1-9171-122f2f3a4fe6",
#     "user_message": "Como funciona a demonstração do processo dos ovos das galinas no diagnóstico oferecida pela Insightiva e o que está incluído nesse nível de acesso?"
#   }'
