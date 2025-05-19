"""
Módulo para registrar tokens e custos de chamadas de LLMs usando callback do LangChain.
Gera logs detalhados com uso de tokens, custo e nome do modelo.
"""

import os
import datetime
import logging
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

logger = logging.getLogger(__name__)

class ModelLogger:
    """
    Classe para monitoramento de sessões de uso de modelos LLM via callback handler.

    Atributos:
        model_name (str): Nome do modelo LLM utilizado.
        log_path (str): Caminho para salvar os logs da sessão.
    """

    def __init__(self, model_name, log_path="logs"):
        """
        Inicializa o logger com um modelo específico.

        Parâmetros:
            model_name (str): Nome do modelo (ex: gpt-3.5-turbo).
            log_path (str): Diretório onde os logs serão salvos.
        """
        self.model_name = model_name
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.callback = OpenAICallbackHandler()

    def get_callback(self):
        """
        Retorna o callback handler usado na sessão LLM.

        Retorna:
            OpenAICallbackHandler: Handler para logging automático do LangChain.
        """
        return self.callback

    def log_session(self):
        """
        Salva um log da sessão com total de tokens usados e custo da chamada.

        Gera um arquivo .txt com informações de uso no diretório de logs.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_path, f"log_{timestamp}.txt")

        with open(log_file, "w") as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Total Tokens: {self.callback.total_tokens}\n")
            f.write(f"Prompt Tokens: {self.callback.prompt_tokens}\n")
            f.write(f"Completion Tokens: {self.callback.completion_tokens}\n")
            f.write(f"Total Cost (USD): ${self.callback.total_cost:.6f}\n")

        logger.info(f"🧾 Log salvo em {log_file}")
