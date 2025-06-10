"""
Módulo responsável por roteamento de perguntas com base em palavras-chave.
Define qual agente da CrewAI deve responder conforme regras em config/index.yaml.
"""

import yaml

def load_index(filepath="config/index.yaml"):
    """
    Carrega o arquivo index.yaml contendo as regras de roteamento por palavra-chave.

    Parâmetros:
        filepath (str): Caminho para o arquivo de configuração.

    Retorna:
        dict: Configuração carregada.
    """
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f)

def escolher_agente(pergunta, index_config):
    """
    Escolhe o agente mais adequado com base nas palavras-chave presentes na pergunta do usuário.

    Parâmetros:
        pergunta (str): Pergunta recebida do usuário.
        index_config (dict): Dicionário com as regras de roteamento.

    Retorna:
        str: Nome do agente escolhido.
    """
    pergunta = pergunta.lower()
    for regra in index_config["routing_rules"]:
        if any(palavra in pergunta for palavra in regra["keywords"]):
            return regra["agent"]
    return index_config.get("default_agent", "Diagnóstico Insightiva")
