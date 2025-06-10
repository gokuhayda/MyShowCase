"""
Módulo responsável por carregar as configurações do projeto a partir de um arquivo YAML.
"""

import os
import yaml
# import logging

# logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "parameters.yaml")

def load_config():
    """
    Carrega as configurações do projeto a partir de um arquivo YAML.

    Retorna:
        dict: Configurações carregadas.
    
    Levanta:
        FileNotFoundError: Se o arquivo de configuração não for encontrado.
        yaml.YAMLError: Se houver erro na leitura do YAML.
    """
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        # logger.error(f"Arquivo de configuração não encontrado: {CONFIG_PATH}")
        raise FileNotFoundError(f"Arquivo não encontrado: {CONFIG_PATH}") from e
    except yaml.YAMLError as e:
        # logger.error("Erro ao interpretar o YAML de configuração.")
        raise yaml.YAMLError("Erro ao interpretar o YAML.") from e
