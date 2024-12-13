from sentence_transformers import SentenceTransformer, util
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from agent_split_question import split_into_subquestions
#from sentence_transformers import CrossEncoder

def query_index_relevant_choice(index, question, num_response, config):
 
    def query_index_with_subquestions(index, question):
        subquestions = split_into_subquestions(question)
        results = []
    
        for sub in subquestions:
            result = index.query(sub)  # Supondo que index.query() realiza a busca
            results.append({"question": sub, "result": result})
        
        return results

    def rerank_results(results, original_question, config):
        model = SentenceTransformer(top_n=config["retrieval"]["top_k"], 
                                    model=config["retrieval"]["reranker_model"]
                                    )
        
        # Combine subquestions e respostas em pares para reranking
        pairs = [(original_question, res["result"]) for res in results]
        scores = model.predict(pairs)
    
        # Combine os resultados com os scores e ordene
        for i, res in enumerate(results):
            res["score"] = scores[i]
        
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        return sorted_results
    
    # 1. Dividir a consulta em subconsultas e buscar respostas
    results = query_index_with_subquestions(index, question)
    
    # 2. Reordenar as respostas com base na relevância
    reranked_results = rerank_results(results, question, config)
    
    # Retornar apenas as `num_responses` mais relevantes
    return reranked_results[:num_response]
    
def query_index_rerank(index, question, config):
    """
    Reordena os resultados da consulta ao índice usando um modelo de reranking.
    
    Args:
        index: Objeto de índice inicializado.
        question: Pergunta do usuário.
        top_k: Número máximo de resultados similares.
        model_conf: Configuração ou modelo para reranking.
    
    Returns:
        Resposta processada pelo engine.
    """
    try:
        # Configurar o reranker
        reranker = SentenceTransformerRerank(
            top_n=config["retrieval"]["top_k"], model=config["retrieval"]["reranker_model"]
        )
        query_engine = index.as_query_engine(similarity_top_k=config["retrieval"]["top_k"], postprocessor=reranker)
        
        response = query_engine.query(question)
        return response.response
    except Exception as e:
        print(f"Erro ao reordenar consulta: {e}")
        return None


def query_index_subquestions(index, question, config):

    query_engine_tool = QueryEngineTool(
        query_engine=index.as_query_engine(similarity_top_k=config['deep_memory']['similarity_top_k']),
        metadata=ToolMetadata(name="pdf_query_engine", description="Consulta baseada em PDFs"),
    )

    sub_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[query_engine_tool],
        use_async=True
    )
    response = sub_query_engine.query(question)
    return response.response