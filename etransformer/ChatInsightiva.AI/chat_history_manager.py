"""
Gerencia o histórico de interações do chatbot, com suporte a sumarização automática
para manter o contexto dentro de limites de token.
"""

import json
import os
from transformers import pipeline

# Inicializa o modelo de sumarização
summarizer = pipeline("summarization", model="t5-small")

def estimate_token_count(text):
    """
    Estima a quantidade de tokens com base na contagem de palavras.

    Parâmetros:
        text (str): Texto a ser avaliado.

    Retorna:
        int: Estimativa do número de tokens.
    """
    return len(text.split())

def summarize_history(history):
    """
    Resume uma lista de interações anteriores (usuário e bot).

    Parâmetros:
        history (list): Lista de dicionários com chaves 'user_message' e 'bot_response'.

    Retorna:
        str: Texto resumido da sessão.
    """
    combined_text = " ".join([f"Usuário: {entry['user_message']}\nBot: {entry['bot_response']}" for entry in history])
    summary = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def manage_history(session_history, user_message, bot_response, token_limit=3000, summary_percentage=0.7):
    """
    Gerencia o histórico da sessão, resumindo interações antigas quando o limite de tokens é ultrapassado.

    Parâmetros:
        session_history (list): Lista atual do histórico.
        user_message (str): Nova mensagem do usuário.
        bot_response (str): Resposta do assistente.
        token_limit (int): Número máximo de tokens permitido.
        summary_percentage (float): Percentual do histórico a ser resumido ao atingir o limite.

    Retorna:
        list: Histórico atualizado (com ou sem resumo).
    """
    session_history.append({"user_message": user_message, "bot_response": bot_response})

    total_tokens = sum(estimate_token_count(f"Usuário: {entry['user_message']}\nBot: {entry['bot_response']}") for entry in session_history)

    while total_tokens > token_limit:
        num_to_summarize = int(len(session_history) * summary_percentage)
        summary_text = summarize_history(session_history[:num_to_summarize])
        session_history = [{"user_message": "Resumo", "bot_response": summary_text}] + session_history[num_to_summarize:]
        total_tokens = sum(estimate_token_count(f"Usuário: {entry['user_message']}\nBot: {entry['bot_response']}") for entry in session_history)

    return session_history

def save_history_to_file(history, filepath="chat_history.json"):
    """
    Salva o histórico da sessão em um arquivo JSON.

    Parâmetros:
        history (list): Histórico da conversa.
        filepath (str): Caminho do arquivo para salvar.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def load_history_from_file(filepath="chat_history.json"):
    """
    Carrega o histórico da sessão de um arquivo JSON, se existir.

    Parâmetros:
        filepath (str): Caminho do arquivo.

    Retorna:
        list: Histórico carregado ou lista vazia.
    """
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
