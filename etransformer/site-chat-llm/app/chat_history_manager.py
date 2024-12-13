from transformers import pipeline

# Inicialize o modelo de sumarização gratuito
summarizer = pipeline("summarization", model="t5-small")

# Função para calcular o número de tokens (estimado como palavras aqui)
def estimate_token_count(text):
    return len(text.split())

# Resumir o histórico antigo
def summarize_history(history):
    # Combine o histórico antigo em um único texto
    combined_text = " ".join([f"Usuário: {entry['user_message']}\nBot: {entry['bot_response']}" for entry in history])
    # Resuma o texto
    summary = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Gerenciar histórico com limite percentual de tokens
def manage_history(session_history, user_message, bot_response, token_limit=3000, summary_percentage=0.7):
    """
    - session_history: Lista do histórico da sessão.
    - user_message: Mensagem atual do usuário.
    - bot_response: Resposta gerada pelo bot.
    - token_limit: Limite total de tokens permitidos.
    - summary_percentage: Porcentagem de tokens a ser resumida quando o limite é atingido.
    """
    # Adicionar a nova interação ao histórico
    session_history.append({"user_message": user_message, "bot_response": bot_response})

    # Calcular tokens usados pelo histórico completo
    total_tokens = sum(estimate_token_count(f"Usuário: {entry['user_message']}\nBot: {entry['bot_response']}") for entry in session_history)

    # Se ultrapassar o limite, aplique sumarização ao histórico antigo
    while total_tokens > token_limit:
        # Defina a quantidade de interações a serem resumidas
        num_to_summarize = int(len(session_history) * summary_percentage)

        # Resuma as interações mais antigas
        summary_text = summarize_history(session_history[:num_to_summarize])
        session_history = [{"user_message": "Resumo", "bot_response": summary_text}] + session_history[num_to_summarize:]

        # Recalcular os tokens
        total_tokens = sum(estimate_token_count(f"Usuário: {entry['user_message']}\nBot: {entry['bot_response']}") for entry in session_history)

    return session_history
