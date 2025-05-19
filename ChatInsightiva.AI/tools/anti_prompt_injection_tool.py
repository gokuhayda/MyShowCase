"""
Ferramenta CrewAI que verifica se uma pergunta contém sinais de tentativa de injeção de prompt,
usando palavras-chave carregadas da configuração do sistema.
"""

from crewai import BaseTool
from typing import Any
from utils_tools.config_loader import load_config

class AntiPromptInjectionTool(BaseTool):
    """
    Tool responsável por bloquear ou validar perguntas que tentam manipular o comportamento do agente.

    Usa lista de palavras-chave definidas em config.yaml para detectar possíveis injeções.
    """
    name: str = "anti_prompt_injection"
    description: str = "Verifica se a entrada contém tentativa de injeção de prompt e bloqueia se necessário."

    def __init__(self):
        super().__init__()
        self._config = load_config()
        self._injetores = [w.lower() for w in self._config.get("injetores", [])]

    def _run(self, prompt: str) -> str:
        """
        Executa a verificação de injeção de prompt.

        Parâmetros:
            prompt (str): Texto fornecido pelo usuário.

        Retorna:
            str: Mensagem informando se a entrada é segura ou bloqueada.
        """
        return self._check_prompt(prompt)

    async def _arun(self, prompt: str) -> str:
        """
        Executa a verificação de injeção de prompt de forma assíncrona.

        Parâmetros:
            prompt (str): Texto fornecido pelo usuário.

        Retorna:
            str: Mensagem informando se a entrada é segura ou bloqueada.
        """
        return self._check_prompt(prompt)

    def _check_prompt(self, prompt: str) -> str:
        """
        Verifica a presença de termos perigosos usados em ataques de prompt injection.

        Parâmetros:
            prompt (str): Texto da entrada.

        Retorna:
            str: Mensagem apropriada de acordo com a segurança da entrada.
        """
        prompt_lower = prompt.lower()

        if any(palavra in prompt_lower for palavra in self._injetores):
            return "🚫 Entrada bloqueada: tentativa de manipulação detectada."

        return "✅ Pergunta validada com segurança."
