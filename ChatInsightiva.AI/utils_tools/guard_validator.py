from guardrails import Guard
from guardrails.hub import GroundedAIHallucination, RestrictToTopic
from utils_tools.config_loader import load_config

# Carrega a configuração do sistema
config = load_config()

# 🔄 Carregar tópicos válidos e inválidos do parameters.yaml
valid_topics = config.get("guardrails", {}).get("valid_topics", [
    "diagnóstico cultural", "onboarding", "valores organizacionais",
    "cultura organizacional", "liderança", "clima organizacional"
])

invalid_topics = config.get("guardrails", {}).get("invalid_topics", [
    "religião", "páscoa", "natal", "ovo", "animal", "mitologia"
])

def criar_guard(documentos_faq: list[str]):
    """
    Cria um objeto Guard com os dois validadores:
    - GroundedAIHallucination com base nos documentos FAQ
    - RestrictToTopic com tópicos do YAML
    """
    return Guard().use_many(
        GroundedAIHallucination(quant="\n\n".join(documentos_faq), disable_model=True),
        RestrictToTopic(
            valid_topics=valid_topics,
            invalid_topics=invalid_topics,
            model="facebook/bart-large-mnli",
            zero_shot_threshold=0.5,
            llm_threshold=3,
            disable_llm=True,
            disable_classifier=False,
            device=-1
        )
    )

def validar_guardrails(resposta: str, documentos_faq: list[str], context: str = ""):
    """
    Valida uma resposta com base nos guardrails configurados.
    
    Parâmetros:
    - resposta: Saída do modelo a ser validada.
    - documentos_faq: Lista de strings com os documentos que fundamentam a verificação de GroundedAIHallucination.
    - context: Contexto adicional usado na validação (ex: backstory dos agentes).

    Retorna:
    - Uma tupla (bool, dict) indicando se a validação foi bem-sucedida e os detalhes do resultado.
    """
    guard = criar_guard(documentos_faq)
    result = guard.validate(llm_output=resposta, context=context)
    return not result.get("failed_validation"), result
