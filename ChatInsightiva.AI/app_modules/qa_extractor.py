"""
Extrai pares QA de um JSON e gera respostas reais usando a pipeline do projeto (MultiIndexFAQRouter).
"""

import json
from datasets import Dataset
from utils_tools.config_loader import load_config
from core.query_router import MultiIndexFAQRouter

# Carrega config
config = load_config()
json_path = config["faq_json_path"]

# Inicializa router como feito no chatbot.py
router = MultiIndexFAQRouter(
    salutations=config["salutations"],
    welcome_messages=config["welcome_messages"],
    notification_warning=config["notification_warning_bot"]
)

def extrair_dados_do_json():
    """
    Extrai perguntas do JSON e gera as respostas reais via pipeline do projeto (router.responder).

    Retorna:
        Dataset: Dataset formatado com colunas [question, answer, reference, contexts].
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    registros = []
    for item in data:
        pergunta = item.get("metadata", {}).get("pergunta", "").strip()
        resposta_gerada = ""
        try:
            resposta_obj = router.responder(pergunta)
            if hasattr(resposta_obj, "resposta"):
                resposta_gerada = resposta_obj.resposta or resposta_obj.comentario
            elif hasattr(resposta_obj, "final_output"):
                resposta_gerada = resposta_obj.final_output
            else:
                resposta_gerada = str(resposta_obj)
        except Exception as e:
            resposta_gerada = f"[Erro ao gerar resposta] {str(e)}"

        registros.append({
            "question": pergunta,
            "answer": resposta_gerada,
            "reference": resposta_gerada,
            "contexts": [resposta_gerada],
        })

    return Dataset.from_list(registros)
