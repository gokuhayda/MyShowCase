from typing import Union
from query_index import rag_pipeline, RespostaRAG
from utils_tools.config_loader import load_config
from core.crew_manager import executar_fallback_concierge, create_agents_from_yaml
from core.graph_loader import get_composable_graph
from utils_tools.model_logger import log_error
import logging

logger = logging.getLogger(__name__)


class MultiIndexFAQRouter:
    """
    Roteador responsável por consultar múltiplos índices FAQ com fallback para agentes CrewAI,
    utilizando um grafo de conhecimento.
    """

    def __init__(self, salutations=None, welcome_messages=None, notification_warning=None):
        logger.info("🔄 Inicializando MultiIndexFAQRouter...")
        self.salutations = salutations or []
        self.welcome_messages = welcome_messages or []
        self.notification_warning = notification_warning or []
        self.config = load_config()
        logger.debug("🔧 Configuração carregada com sucesso.")

        self.grafo = get_composable_graph()
        logger.info("🕸️ Grafo unificado criado com sucesso.")

        self.agents = create_agents_from_yaml("config/agents.yaml", [])

    def resposta_aprovada(self, resposta, config: dict) -> bool:
        """
        Verifica se a resposta passou nos critérios de validação.

        Parâmetros:
            resposta: Objeto de resposta retornado pelo RAG.
            config (dict): Configurações que incluem limiar de confiança.

        Retorna:
            bool: True se aprovada, False se falhar na validação ou confiança.
        """
        min_conf = config.get("confidence", {}).get("min_confidence", 0.75)
        result = resposta.passou_validacao and resposta.confianca >= min_conf
        logger.debug(
            f"📊 Validação da resposta — limiar: {min_conf}, passou: {resposta.passou_validacao}, confiança: {resposta.confianca}"
        )
        return result

    def responder(self, pergunta: str, session_history: list = None) -> RespostaRAG:
        """
        Processa a pergunta utilizando o pipeline RAG com fallback inteligente.

        Parâmetros:
            pergunta (str): Pergunta do usuário.
            session_history (list): Histórico de conversas da sessão.

        Retorna:
            RespostaRAG: Objeto com resposta, fontes, confiança e status de validação.
        """
        logger.info(f"🚀 Processando pergunta: {pergunta}")

        # Montar prompt com histórico, se houver
        if session_history:
            contexto = "\n".join([
                f"Usuário: {msg['user_message']}\nAssistente: {msg['bot_response']}"
                for msg in session_history if msg.get("user_message") and msg.get("bot_response")
            ])
            prompt_completo = (
                f"Este é o histórico da conversa até agora:\n"
                f"{contexto}\n\n"
                f"Agora responda à nova pergunta do usuário com base nesse histórico:\n"
                f"Usuário: {pergunta.strip()}"
            )
        else:
            prompt_completo = pergunta

        try:
            resposta = rag_pipeline(self.grafo, prompt_completo)
            logger.debug("📥 Resposta recebida com sucesso do rag_pipeline.")
        except Exception as e:
            log_error(f"Erro ao processar a consulta: {str(e)}")
            logger.exception("❌ Erro ao executar rag_pipeline.")
            return RespostaRAG(
                resposta="Houve um erro ao processar sua consulta.",
                fontes=[],
                confianca=0.0,
                comentario="Erro de execução no rag_pipeline.",
                tipo="erro",
                passou_validacao=False
            )

        if not self.resposta_aprovada(resposta, self.config):
            logger.warning("⚠️ Resposta com baixa confiança. Delegando para fallback.")
            resposta_fallback = executar_fallback_concierge(
                pergunta,
                resposta.resposta,
                self.agents
            )
            return RespostaRAG(
                resposta=resposta_fallback,
                fontes=[],
                confianca=resposta.confianca,
                comentario="Resposta gerada via fallback Concierge.",
                tipo="agente",
                passou_validacao=False
            )

        logger.info("✅ Resposta válida encontrada.")
        resposta.comentario = "Resposta validada via RAG com sucesso."
        return resposta
