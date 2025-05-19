"""
Logger de rastreamento com Langfuse para monitorar execuções de LLMs.
"""

from langfuse import Langfuse
import os

# Inicializa com variáveis do ambiente (.env)
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

def registrar_trace_reescrita(pergunta, resposta, usuario="user-anonimo"):
    """
    Registra uma chamada de reescrita no Langfuse.

    Args:
        pergunta (str): Texto da pergunta original.
        resposta (str): Texto da resposta do chatbot.
        usuario (str): ID do usuário (padrão: user-anonimo).
    """
    try:
        trace = langfuse.trace(name="qa_rewriter", user_id=usuario)
        trace.step("pergunta_reescrita").generation(prompt=pergunta, completion=resposta)
    except Exception as e:
        print(f"[LangfuseLogger] ⚠️ Falha ao registrar no Langfuse: {e}")
