import os
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import torch

# Carregar variáveis do arquivo .env
load_dotenv()

def initialize_embedding_model(config):
    """
    Inicializa o modelo de embeddings com base na configuração.
    """
    
    embedding_config = config["embedding_model"]

    try:
        if embedding_config["type"].startswith("text-embedding"):
            print(f"Inicializando modelo de embeddings OpenAI: {embedding_config['type']}")
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("A variável OPENAI_API_KEY não está configurada no .env.")
            return OpenAIEmbedding(
                model_name=embedding_config["type"],
                api_key=openai_key,
                max_tokens=embedding_config["max_tokens"]
            )
        else:
            print(f"Inicializando modelo de embeddings HuggingFace: {embedding_config['type']}")
            return HuggingFaceEmbedding(
                model_name=embedding_config["type"]
            )
    except Exception as e:
        raise RuntimeError(f"Erro ao inicializar o modelo de embeddings: {e}")
        
def initialize_llm_model(config):
    """
    Inicializa os modelos de LLM para uso em um banco vetorial.
    """
    llm = None

    model_name = config.get("default_model", "gpt4")
    device = "cuda" if torch.cuda.is_available() and config[model_name].get("device") == "cuda" else "cpu"

    if model_name == "gpt4":
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                print(f"Inicializando modelo GPT-4 para chat.")
                llm = OpenAI(
                    model=config["gpt4"]["type"],
                    api_key=openai_key,
                    max_tokens=config["gpt4"]["max_tokens"]
                )
            except Exception as e:
                print(f"Erro ao inicializar GPT-4: {e}")
    elif model_name == "llama":
        huggingface_key = os.getenv("HUGGINGFACE_API_KEY")
        if huggingface_key:
            try:
                print(f"Inicializando modelo LLaMA para chat no dispositivo {device}.")
                llm = HuggingFaceLLM(
                    model_name=config["llama"]["type"],
                    device=device,
                    temperature=config["llama"]["temperature"],
                    top_p=config["llama"]["top_p"]
                )
            except Exception as e:
                print(f"Erro ao inicializar LLaMA: {e}")

    # Validar modelo de chat inicializado
    if not llm:
        raise RuntimeError("Nenhum modelo de chat foi inicializado com sucesso.")

    print("LLM criado com sucesso.")
    return llm
