"""
Tool CrewAI para validação semântica de perguntas com base em conteúdo vetorizado
(global, sem segmentação por categoria), usando ComposableGraph.
"""

import os
from typing import Any, Dict, Type
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools.base_tool import BaseTool
from utils_tools.config_loader import load_config
from llama_index.core.composability import ComposableGraph
from model_loader import initialize_embedding_model
import numpy as np
#from utils_tools.guard_validator import validar_guardrails  # ✅ substitui RestrictToTopic direto

class ToolInputSchema(BaseModel):
    description: str = Field(..., description="Pergunta ou texto para validação semântica.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados opcionais adicionais.")

class SemanticGuardTool(BaseTool):
    """
    Ferramenta que verifica se uma pergunta é semanticamente próxima de conteúdos vetorizados do FAQ,
    utilizando um grafo ComposableGraph. Se a confiança for inferior ao limiar, retorna mensagem de fallback.
    """
    name: str = "semantic_guard_tool"
    description: str = "Verifica se a pergunta está semanticamente relacionada ao FAQ via ComposableGraph"
    args_schema: Type[BaseModel] = ToolInputSchema

    _config: dict = PrivateAttr()
    _threshold: float = PrivateAttr()
    _top_k: int = PrivateAttr()
    _fallback: str = PrivateAttr()
    _query_engine: Any = PrivateAttr()

    def __init__(self, composable_graph: ComposableGraph):
        super().__init__()
        self._config = load_config()
        self._threshold = self._config.get("confidence", {}).get("semantic_threshold", 0.75)
        self._top_k = self._config.get("top_k", 13)
        self._fallback = self._config.get(
            "fallback_message",
            "Essa pergunta está fora do escopo do FAQ. Por favor, pergunte algo relacionado."
        )

        embed_model = initialize_embedding_model(self._config)
        self._query_engine = composable_graph.as_query_engine(
            similarity_top_k=self._top_k,
            embed_model=embed_model
        )

    def _run(self, description: str, metadata: Dict[str, Any]) -> Any:
        try:
            response = self._query_engine.query(description)
            score = response.metadata.get("score", 0.0)
            print(f"semantic_guard_tool — score retornado: {score} (limiar: {self._threshold})")

            if score < self._threshold:
                return self._fallback

            return f"A pergunta é relevante. Resposta:\n\n{response.response.strip()}"

        except Exception as e:
            return f"Erro ao verificar similaridade semântica: {str(e)}"


def validar_semantica_antes_da_crew(pergunta: str) -> tuple[bool, str]:
    try:
        from core.graph_loader import get_composable_graph
        graph = get_composable_graph()
        ferramenta = SemanticGuardTool(graph)
        resultado = ferramenta._run(pergunta, metadata={})

        if resultado.startswith("Essa pergunta está fora do escopo"):
            return False, resultado
        return True, resultado

    except Exception as e:
        return False, f"Erro ao validar semanticamente: {str(e)}"


#def verificar_tema_invalido(texto: str) -> tuple[bool, str]:
#    """
#    Verifica se o texto contém temas inválidos com base nos tópicos proibidos definidos no YAML,
#    utilizando a função centralizada `validar_guardrails`, que já aplica os guardrails com segurança.
#    Em caso de erro, assume que o tema não é inválido.
#    """
#    try:
#        documentos_faq = ["Placeholder para validação de tópico com RestrictToTopic"]
#        validou, resultado = validar_guardrails(resposta=texto, documentos_faq=documentos_faq)
#
#        if not validou:
#            return True, resultado.get("error", "tema inválido detectado")
#        return False, resultado.get("topic", None)
#
#    except Exception as e:
#        import logging
#        logging.warning(f"[verificar_tema_invalido] Ignorando erro durante verificação: {e}")
#        return False, None
