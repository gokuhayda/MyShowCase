
"""
Módulo de inicialização de modelos de embedding e LLMs com suporte a OpenAI, HuggingFace e Ollama.
"""

import os
import torch
import logging
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from langchain_ollama import OllamaEmbeddings

load_dotenv()
logger = logging.getLogger(__name__)

def initialize_embedding_model(config):
    """
    Inicializa o modelo de embedding com base na configuração.

    Parâmetros:
        config (dict): Dicionário de configurações do sistema.

    Retorna:
        Instância de embedding (OpenAI, HuggingFace ou Ollama).
    """
    embedding_type = config.get("embedding_type", "huggingface")

    if embedding_type == "openai":
        embedding_cfg = config.get("gpt", {}).get("embedding_model", {})
        logger.info(f"[EMBED] Inicializando OpenAIEmbedding: {embedding_cfg.get('type')}")
        return OpenAIEmbedding(model=embedding_cfg.get("type", "text-embedding-3-small"))

    elif embedding_type == "ollama":
        embedding_cfg = config.get("ollama", {}).get("embedding_model", {})
        logger.info(f"[EMBED] Inicializando OllamaEmbedding: {embedding_cfg.get('type')}")
        return OllamaEmbeddings(model=embedding_cfg.get("type", "llama"))

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[EMBED] Inicializando HuggingFaceEmbedding no dispositivo: {device}")
        return HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device=device
        )

def initialize_llm_model(config):
    """
    Inicializa o modelo LLM com base na configuração.

    Parâmetros:
        config (dict): Dicionário de configurações.

    Retorna:
        Instância de modelo de linguagem (OpenAI ou Ollama).
    """
    logger.info("🤖 [LLM] Configurando LLM...")

    default_model_key = config.get("default_model", "gpt").lower()
    if default_model_key not in ["gpt", "ollama"]:
        raise ValueError(f"[LLM] Modelo default inválido: {default_model_key}. Deve ser 'gpt' ou 'ollama'.")

    logger.info(f"[LLM] Modelo escolhido: {default_model_key}")

    if default_model_key == "gpt":
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OPENAI_API_KEY não está configurada no .env.")

            gpt_cfg = config.get("gpt", {})
            logger.info("🧠 [LLM] Inicializando modelo GPT com OpenAI...")
            return OpenAI(
                model=gpt_cfg["type"],
                api_key=openai_key,
                max_tokens=gpt_cfg.get("max_tokens", 2048),
                temperature=gpt_cfg.get("temperature", 0.3)
            )
        except Exception as e:
            logger.warning(f"[LLM] Falha na inicialização do GPT: {e}")
            logger.info("[LLM] Tentando fallback para modelo local Ollama...")

    if default_model_key == "ollama":
        try:
            ollama_cfg = config.get("ollama", {})
            logger.info(f"🧠 [LLM] Inicializando modelo via Ollama: {ollama_cfg['type']}")
            return Ollama(
                model=ollama_cfg["type"],
                request_timeout=60.0,
                temperature=ollama_cfg.get("temperature", 0.7),
                additional_kwargs={"num_predict": ollama_cfg.get("additional_kwargs", {}).get("num_predict", 100)}
            )
        except Exception as e:
            raise RuntimeError(f"[LLM] Falha ao inicializar Ollama: {e}")

    raise RuntimeError("[LLM] Tipo de modelo não reconhecido ou mal configurado.")

