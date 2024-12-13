import yaml
from docs_storage_setup import prepare_documents_storage
from llama_index.core import VectorStoreIndex
from model_loader import initialize_embedding_model, initialize_llm_model

# Função para criar o índice vetorial
def create_index(config):
    documents, storage_context = prepare_documents_storage(config)
    embedding_model = initialize_embedding_model(config)
    llm = initialize_llm_model(config)
        
    # Criar índice vetorial
    vector_index = VectorStoreIndex(
                documents, 
                storage_context=storage_context,
                llm=llm,
                embedding_model=embedding_model,
            )
    return vector_index