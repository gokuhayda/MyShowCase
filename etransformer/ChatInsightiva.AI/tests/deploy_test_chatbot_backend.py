"""
Script utilitário para:
- Subir o backend do chatbot via tmux
- Criar túnel com ngrok
- Atualizar automaticamente o chatbot.js com a nova URL
- Realizar uma requisição de teste com curl
"""

import subprocess
import time
import re
import os
from pathlib import Path

# CONFIG
CHATBOT_JS_PATH = Path("chatbot.js")
TMUX_SESSION_NGROK = "chatbot_ngrok"
TMUX_SESSION_BACKEND = "chatbot_backend"
PORTA = 5000
BACKEND_SCRIPT = "chatbot.py"

#TEST_MESSAGE = "Principais desafios na implementação de mudanças culturais?"
TEST_MESSAGE = "Como funciona a demonstração do processo de diagnóstico oferecida pela Insightiva e o que está incluído nesse nível de acesso?"
def iniciar_tmux(nome, comando):
    """
    Inicia uma sessão tmux com nome e comando específicos, se ainda não estiver rodando.

    Parâmetros:
        nome (str): Nome da sessão.
        comando (str): Comando a ser executado dentro da sessão tmux.
    """
    result = subprocess.run(["tmux", "has-session", "-t", nome], capture_output=True)
    if result.returncode != 0:
        print(f"🚀 Iniciando sessão tmux: {nome}")
        subprocess.run(["tmux", "new-session", "-d", "-s", nome, comando])
    else:
        print(f"⚠️ Sessão tmux '{nome}' já está rodando.")

def iniciar_backend():
    """Inicia o backend do chatbot via tmux."""
    iniciar_tmux(TMUX_SESSION_BACKEND, f"python3 {BACKEND_SCRIPT}")

def iniciar_ngrok():
    """Inicia o túnel ngrok via tmux."""
    iniciar_tmux(TMUX_SESSION_NGROK, f"ngrok http {PORTA}")

def obter_link_ngrok():
    """
    Aguarda o link público do ngrok ficar disponível.

    Retorna:
        str | None: URL pública HTTPS ou None se falhar.
    """
    print("⏳ Aguardando link público do ngrok...")
    for _ in range(20):
        try:
            import requests
            response = requests.get("http://localhost:4040/api/tunnels")
            tunnels = response.json().get("tunnels", [])
            for t in tunnels:
                if t["proto"] == "https":
                    return t["public_url"]
        except Exception:
            pass
        time.sleep(1)
    return None

def atualizar_api_url_no_chatbot_js(nova_url):
    """
    Atualiza o valor da constante API_URL no arquivo chatbot.js com a nova URL.

    Parâmetros:
        nova_url (str): Novo link público fornecido pelo ngrok.

    Retorna:
        bool: True se sucesso, False se erro.
    """
    if not CHATBOT_JS_PATH.exists():
        print(f"❌ Arquivo {CHATBOT_JS_PATH} não encontrado.")
        return False

    conteudo = CHATBOT_JS_PATH.read_text(encoding="utf-8")
    novo_conteudo = re.sub(
        r"const API_URL = 'https://.*?';",
        f"const API_URL = '{nova_url}/chat';",
        conteudo
    )
    CHATBOT_JS_PATH.write_text(novo_conteudo, encoding="utf-8")
    print(f"✅ chatbot.js atualizado com: {nova_url}/chat")
    return True

def executar_curl_dinamico(base_url):
    """
    Executa um teste via curl para verificar se o backend está operacional.

    Parâmetros:
        base_url (str): URL base pública do ngrok.
    """
    print("📡 Executando teste com curl dinâmico...")
    curl_cmd = [
        "curl", "-X", "POST",
        f"{base_url}/chat",
        "-H", "Content-Type: application/json",
        "-d", f'{{"user_message": "{TEST_MESSAGE}"}}'
    ]
    try:
        resultado = subprocess.run(curl_cmd, capture_output=True, text=True)
        print("📥 Resposta do backend:\n")
        print(resultado.stdout.strip())
        if resultado.stderr:
            print("⚠️ Erro:\n", resultado.stderr.strip())
    except Exception as e:
        print(f"❌ Erro ao executar curl: {e}")

def main():
    """
    Executa a sequência completa:
    - Encerra qualquer tmux ativo
    - Sobe backend
    - Sobe ngrok
    - Atualiza chatbot.js
    - Faz teste via curl
    """
    # ✅ Encerra sessões tmux antigas
    subprocess.run(["tmux", "kill-server"])
    print("💥 Todas as sessões tmux anteriores foram encerradas.")

    iniciar_backend()
    iniciar_ngrok()
    link = obter_link_ngrok()

    if link:
        atualizar_api_url_no_chatbot_js(link)
        executar_curl_dinamico(link)
    else:
        print("❌ Não foi possível obter a URL do ngrok.")


if __name__ == "__main__":
    main()
