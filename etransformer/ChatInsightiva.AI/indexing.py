"""
Módulo responsável por construir o índice vetorial com base em documentos processados.
Utiliza modelo de embedding configurado e armazena o índice no DeepLake.
"""

import os
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Document
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from utils_tools.config_loader import load_config
from model_loader import initialize_embedding_model

config = load_config()

def create_index(config=config):
    """
    Cria um índice vetorial a partir de documentos de texto processados.

    Parâmetros:
        config (dict): Configuração contendo caminhos e modelo de embedding.

    Retorna:
        VectorStoreIndex: Índice vetorial gerado e armazenado no DeepLake.
    """
    # Carregar documentos brutos
    raw_data_path = config["storage"]["processed_texts_directory"]
    print(f"📂 Lendo documentos de: {raw_data_path}")
    documents = SimpleDirectoryReader(raw_data_path, recursive=True).load_data()

    # Inicializar modelo de embedding
    embed_model = initialize_embedding_model(config)
    test_dim = len(embed_model.get_text_embedding("teste"))
    print(f"🔢 Dimensão dos vetores de embedding: {test_dim}")

    # Definir caminho do DeepLake
    deeplake_path = config["storage"]["vector_store_path"]
    print(f"💾 Inicializando armazenamento em: {deeplake_path}")

    # Criar o armazenamento com overwrite=True
    vector_store = DeepLakeVectorStore(dataset_path=deeplake_path, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Criar o índice
    print("📈 Criando índice com embeddings...")
    index = VectorStoreIndex.from_documents(
        [Document(text=doc.text) for doc in documents],
        storage_context=storage_context,
        embed_model=embed_model
    )

    print("✅ Índice criado com sucesso.")
    return index
