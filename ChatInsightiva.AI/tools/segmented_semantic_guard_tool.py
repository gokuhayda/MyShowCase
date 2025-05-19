"""
Tool CrewAI personalizada para verificar se uma pergunta é semanticamente relevante dentro
de uma categoria específica, usando um ComposableGraph com filtros e rerank opcional.
"""
import os
from typing import Any, Dict, Type
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools.base_tool import BaseTool
from utils_tools.config_loader import load_config
from llama_index.core.composability import ComposableGraph
from model_loader import initialize_embedding_model
from query_index import verificar_relevancia, verificar_confiança, rerank_with_cohere, get_confidence

class SegmentedToolInputSchema(BaseModel):
    description: str = Field(..., description="Pergunta ou texto para validação semântica.")
    categoria: str = Field(..., description="Categoria específica dentro do ComposableGraph.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados opcionais adicionais.")

class SegmentedSemanticGuardTool(BaseTool):
    """
    Ferramenta CrewAI que valida perguntas com base em conteúdo vetorizado segmentado por categoria.

    Utiliza ComposableGraph + rerank opcional com Cohere + filtros e confiança mínima.
    """
    name: str = "semantic_segmented_guard_tool"
    description: str = "Verifica se a pergunta está semanticamente relacionada ao conteúdo vetorizado dentro de uma categoria específica via ComposableGraph."
    args_schema: Type[BaseModel] = SegmentedToolInputSchema

    _config: dict = PrivateAttr()
    _threshold: float = PrivateAttr()
    _top_k: int = PrivateAttr()
    _fallback: str = PrivateAttr()
    _graph: ComposableGraph = PrivateAttr()
    _embed_model: Any = PrivateAttr()

    def __init__(self, composable_graph: ComposableGraph):
        super().__init__()
        self._config = load_config()
        self._threshold = self._config.get("confidence", {}).get("semantic_threshold", 0.75)
        self._top_k = self._config.get("top_k", 13)
        self._fallback = self._config.get(
            "fallback_message",
            "Essa pergunta está fora do escopo do FAQ. Por favor, pergunte algo relacionado."
        )
        self._graph = composable_graph
        self._embed_model = initialize_embedding_model(self._config)

    def _run(self, description: str, categoria: str, metadata: Dict[str, Any]) -> str:
        """
        Executa o processo de verificação semântica para uma categoria específica.

        Parâmetros:
            description (str): Pergunta a ser validada.
            categoria (str): Categoria a ser consultada no grafo.
            metadata (dict): Dados adicionais opcionais.

        Retorna:
            str: Resposta validada ou mensagem de fallback.
        """
        try:
            print(f"🔍 [SegmentedTool] Executando pipeline segmentada para a categoria: {categoria}")
            query_engine = self._graph.as_query_engine(
                similarity_top_k=self._top_k,
                embed_model=self._embed_model,
                filters={"categoria": categoria}
            )
            response = query_engine.query(description)
            print("📥 [SegmentedTool] Resposta bruta recebida.")

            resposta_bruta = response.response.strip()
            source_nodes = getattr(response, "source_nodes", [])
            if not resposta_bruta or not source_nodes:
                print("❌ [SegmentedTool] Nenhum conteúdo retornado.")
                return self._fallback

            documentos_faq = [node.text for node in source_nodes]
            if not verificar_relevancia(description, documentos_faq, para_query=True):
                print("❌ [SegmentedTool] Pergunta sem relevância semântica.")
                return self._fallback

            confianca_inicial = get_confidence(source_nodes)
            print(f"🔢 [SegmentedTool] Confiança inicial: {confianca_inicial} (limiar: {self._threshold})")

            if self._config.get("usar_rerank", False):
                print("🔁 [SegmentedTool] Aplicando rerank com Cohere...")
                source_nodes = rerank_with_cohere(source_nodes, description)

            confianca_final = get_confidence(source_nodes)
            is_semantic_valid = verificar_relevancia(resposta_bruta, documentos_faq)

            if not is_semantic_valid or confianca_final < self._threshold:
                print("⚠️ [SegmentedTool] Resposta fora dos critérios.")
                return self._fallback

            print("✅ [SegmentedTool] Resposta validada com sucesso.")
            return f"A pergunta é relevante na categoria '{categoria}'. Resposta:\n\n{resposta_bruta}"

        except Exception as e:
            print(f"💥 [SegmentedTool] Erro na execução: {e}")
            return f"Erro ao verificar similaridade segmentada: {str(e)}"
