import os
import streamlit as st
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.legacy.readers.file.base import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import SentenceTransformerRerank
from app.utils import process_pdfs

def create_index_with_rerankers(pdf_directory):
    text_dir = process_pdfs(pdf_directory)
    
    # Verificar se há arquivos processados
    processed_files = [f for f in os.listdir(text_dir) if f.endswith(".txt")]
    if not processed_files:
        raise FileNotFoundError(f"Nenhum arquivo de texto encontrado em {text_dir}. Certifique-se de que os PDFs foram processados corretamente.")
    
    documents = SimpleDirectoryReader(text_dir).load_data()
    documents = [Document(text=doc.text, metadata=doc.metadata) for doc in documents]

    node_parser = SimpleFileNodeParser.from_defaults()
    nodes = node_parser.get_nodes_from_documents(documents)

    dataset_path = os.path.join(pdf_directory, "deep_lake_db")
    vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    llm = OpenAI(model="gpt-4")
    embed_model = OpenAIEmbedding()

    vector_index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        llm=llm,
        embedding_model=embed_model,
        show_progress=True
    )
    return vector_index

# Função para consulta com suporte a Rerankers
def query_index_with_rerankers(question):
    if "vector_index" not in st.session_state:
        raise ValueError("O índice vetorial não foi criado.")
    
    vector_index = st.session_state.vector_index

    # Configuração do reranker
    reranker = SentenceTransformerRerank(
        top_n=5, model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    # Alternativamente, use:
    # reranker = LLMRerank(choice_batch_size=4, top_n=5)

    query_engine = vector_index.as_query_engine(
        similarity_top_k=10,
        postprocessor=reranker
    )
    response = query_engine.query(question)
    return response.response

