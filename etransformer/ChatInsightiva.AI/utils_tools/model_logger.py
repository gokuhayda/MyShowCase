import time
import json
from collections import defaultdict

# Tabela de preços por mil tokens
MODEL_PRICING = {
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
    "llama3": {"prompt": 0.0004, "completion": 0.0005},
    "mistral": {"prompt": 0.0002, "completion": 0.0003},
}

# Lista para armazenar logs
CALL_LOG = []

def log_model_call(model_name, prompt_tokens=0, completion_tokens=0, timestamp=None):
    """Registra uma chamada de modelo e calcula o custo"""
    timestamp = timestamp or time.time()

    if model_name not in MODEL_PRICING:
        print(f"[LOGGER WARNING] Modelo {model_name} não está no dicionário de preços. Registrando com custo zero.")
        cost = 0.0
    else:
        pricing = MODEL_PRICING[model_name]
        cost = (prompt_tokens / 1000) * pricing["prompt"] + (completion_tokens / 1000) * pricing["completion"]

    CALL_LOG.append({
        "timestamp": timestamp,
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": round(cost, 6)
    })

def log_from_response(response, model_name=None):
    """
    Registra uma chamada de modelo usando a resposta da API.
    Compatível com dicionários ou objetos. Tolera ausência de usage.
    """
    try:
        if hasattr(response, "usage"):
            usage = response.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
        elif isinstance(response, dict) and "usage" in response:
            usage = response["usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            print("[LOGGER INFO] Nenhuma informação de usage na resposta. Registrando chamada básica.")
            model = model_name or getattr(response, "model", None) or (
                response.get("model") if isinstance(response, dict) else "unknown_model"
            )
            log_model_call(model)
            return

        model = model_name or getattr(response, "model", None) or (
            response.get("model") if isinstance(response, dict) else "unknown_model"
        )

        log_model_call(model, prompt_tokens, completion_tokens)

    except Exception as e:
        print(f"[LOGGER ERROR] Falha ao registrar chamada: {e}")

def log_error(message):
    """Registra uma mensagem de erro no log com timestamp"""
    timestamp = time.time()
    CALL_LOG.append({
        "timestamp": timestamp,
        "error_message": message
    })
    print(f"[LOGGER ERROR] {message}")

def get_stats():
    """Retorna estatísticas agregadas"""
    stats = defaultdict(lambda: {"calls": 0, "total_tokens": 0, "total_cost": 0})

    for call in CALL_LOG:
        model = call["model"]
        stats[model]["calls"] += 1
        stats[model]["total_tokens"] += call["total_tokens"]
        stats[model]["total_cost"] += call["cost_usd"]

    total_cost = sum(model_stat["total_cost"] for model_stat in stats.values())
    total_calls = sum(model_stat["calls"] for model_stat in stats.values())

    return {
        "per_model": dict(stats),
        "total_calls": total_calls,
        "total_cost_usd": round(total_cost, 6)
    }

def export_log_to_json(path="model_calls_log.json"):
    """Exporta o log em formato JSON"""
    with open(path, "w") as f:
        json.dump(CALL_LOG, f, indent=2)

# Logger padrão para importação
import logging

logger = logging.getLogger("chatInsightiva")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


import os

# Cria a pasta de logs se não existir
os.makedirs("logs", exist_ok=True)

# Configura log em arquivo
fh = logging.FileHandler("logs/tools.log", encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Adiciona handler de arquivo se ainda não estiver presente
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(fh)
