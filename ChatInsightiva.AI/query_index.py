import logging
logger = logging.getLogger(__name__)
import os
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from utils_tools.config_loader import load_config
from dataclasses import dataclass
from typing import List
import cohere
from model_loader import initialize_embedding_model
from llama_index.core.composability import ComposableGraph
from llama_index.llms.openai import OpenAI
# 🔻 REMOVIDO: from utils_tools.guard_validator import validar_guardrails

logger.debug("🔄 [Início] Carregando configuração do sistema...")
config = load_config()
logger.debug("🔧 Configuração carregada:")

confidence_config = config["confidence"]
vector_store_path = config["storage"]["vector_store_path"]


@dataclass
class RespostaRAG:
    resposta: str
    fontes: List[str]
    confianca: float
    comentario: str
    tipo: str
    passou_validacao: bool


def get_confidence(source_nodes):
    if source_nodes and hasattr(source_nodes[0], "score"):
        return source_nodes[0].score
    return 0.0


def verificar_relevancia(texto: str, documentos: list, para_query: bool = False) -> bool:
    embed_model = initialize_embedding_model(config)
    try:
        texto_emb = embed_model._get_text_embedding(texto)
        docs_embs = embed_model._get_text_embeddings(documentos)
        similaridade = cosine_similarity([texto_emb], docs_embs)
        limiar = confidence_config.get("semantic_threshold", 0.80 if para_query else 0.7)
        return similaridade.max() >= limiar
    except Exception as e:
        logger.debug(f"Erro na verificação semântica: {e}")
        return False


def verificar_confiança(confianca, confianca_threshold=None):
    confianca_threshold = confianca_threshold or confidence_config.get("min_confidence", 0.75)
    return confianca >= confianca_threshold


def rerank_with_cohere(docs, query):
    try:
        co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        passages = [doc.text for doc in docs]
        results = co.rerank(
            query=query,
            documents=passages,
            model=confidence_config.get("cohere_model", "rerank-multilingual-v2.0"),
            top_n=confidence_config.get("similarity_top_k", 15)
        ).results
        min_score = confidence_config.get("rerank_min_score", 0.5)
        return [docs[r.index] for r in results if r.relevance_score > min_score]
    except Exception as e:
        logger.debug(f"Erro no rerank com Cohere: {e}")
        return docs


def get_query_engine(index, config):
    gpt_cfg = config["gpt"]
    similarity_top_k = confidence_config["similarity_top_k"]
    embed_model = initialize_embedding_model(config)

    llm = OpenAI(
        model=gpt_cfg.get("type", "gpt-3.5-turbo"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=gpt_cfg.get("temperature", 0.3),
        top_p=gpt_cfg.get("top_p", 0.8),
        max_tokens=gpt_cfg.get("max_tokens", 500),
        max_input_size=gpt_cfg.get("max_input_size", 4096),
        frequency_penalty=gpt_cfg.get("frequency_penalty", 0.2),
        presence_penalty=gpt_cfg.get("presence_penalty", 0.1),
        system_message=gpt_cfg.get("system_message", "").strip()
    )

    return index.as_query_engine(
        llm=llm,
        similarity_top_k=similarity_top_k,
        embed_model=embed_model
    )


def rag_pipeline(index_or_graph, pergunta):
    logger.debug("Iniciando pipeline RAG...")
    usar_traducao = config.get("translate", True)
    usar_rerank = config.get("usar_rerank", True)

    embed_model = initialize_embedding_model(config)

    if isinstance(index_or_graph, ComposableGraph):
        query_engine = index_or_graph.as_query_engine(embed_model=embed_model)
    else:
        query_engine = get_query_engine(index_or_graph, config)

    response = query_engine.query(pergunta)
    resposta_bruta = response.response.strip()
    source_nodes = getattr(response, "source_nodes", [])

    if not resposta_bruta or not source_nodes:
        return RespostaRAG(
            resposta="Essa pergunta está fora do escopo do FAQ. Por favor, pergunte algo relacionado ao nosso conteúdo.",
            fontes=[],
            confianca=0.0,
            comentario="Nenhum conteúdo recuperado.",
            tipo="agente",
            passou_validacao=False
        )

    documentos_faq = [node.text for node in source_nodes]

    if not verificar_relevancia(pergunta, documentos_faq, para_query=True):
        return RespostaRAG(
            resposta="Desculpe, sua pergunta parece estar fora do escopo do FAQ. Poderia reformular?",
            fontes=[],
            confianca=0.0,
            comentario="Pergunta semanticamente fora do contexto.",
            tipo="agente",
            passou_validacao=False
        )

    confianca_inicial = get_confidence(source_nodes)

    if usar_rerank:
        source_nodes = rerank_with_cohere(source_nodes, pergunta)

    confianca_final = get_confidence(source_nodes)
    is_semantic_valid = verificar_relevancia(resposta_bruta, documentos_faq)

    if not is_semantic_valid or not verificar_confiança(confianca_final):
        return RespostaRAG(
            resposta="Desculpe, não encontrei essa informação, mas posso encaminhar para o nosso agente tentar ajudar!",
            fontes=[],
            confianca=confianca_final,
            comentario="Resposta de baixa confiança ou fora do escopo semântico.",
            tipo="agente",
            passou_validacao=False
        )

    fontes_formatadas = [f"📚 {node.text.strip().split('\\n')[0][:100]}" for node in source_nodes]

    if usar_traducao:
        try:
            resposta_bruta = GoogleTranslator(source='auto', target='pt').translate(resposta_bruta)
        except Exception as e:
            logger.debug(f"Erro na tradução: {e}")

    return RespostaRAG(
        resposta=resposta_bruta,
        fontes=fontes_formatadas,
        confianca=confianca_final,
        comentario="Resposta validada com sucesso.",
        tipo="faq",
        passou_validacao=True
    )
