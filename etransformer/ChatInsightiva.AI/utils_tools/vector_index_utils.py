
from llama_index.core.embeddings.base import BaseEmbedding

def get_embedding_dimension(embedding_model: BaseEmbedding) -> int:
    test = embedding_model.get_text_embedding("test string")
    return len(test)

     
import os
import logging
import os
from pathlib import Path
from indexing import create_index
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from model_loader import initialize_embedding_model
from utils_tools.config_loader import load_config
config = load_config()
def load_embeddings():
    """
    Tenta carregar embeddings locais (HuggingFace). Se falhar, tenta OpenAI.
    """
    logging.info("🔠 [EMBED] Carregando HuggingFaceEmbeddings...")
    return initialize_embedding_model(config)
#    try:
#        logging.info("🔠 [EMBED] Tentando carregar HuggingFaceEmbeddings...")
#        embeddings = initialize_embedding_model(config)
#        logging.info("✅ [EMBED] HuggingFace carregado com sucesso")
#        return embeddings
#    except Exception as e_hf:
#        logging.warning(f"⚠️ [EMBED] HuggingFace falhou: {e_hf}")
#        try:
#            logging.info("🔁 [EMBED] Fallback para OpenAIEmbeddings...")
#            embeddings = initialize_embedding_model(config)
#            logging.info("✅ [EMBED] OpenAI carregado com sucesso")
#            return embeddings
#        except Exception as e_openai:
#            logging.error(f"❌ [EMBED] OpenAI também falhou: {e_openai}")
#            raise RuntimeError("🚨 Nenhum provedor de embeddings disponível. Verifique suas dependências e configuração.")

def load_combined_index(config):

    # 1. Carregar documentos FAQ locais
    print("📄 Carregando documentos do FAQ local...")

    def expand_faq_sources(sources):
        expanded = []
        for path in sources:
            if os.path.isdir(path):
                for fname in os.listdir(path):
                    if fname.endswith(".txt"):
                        full_path = os.path.join(path, fname)
                        expanded.append(full_path)
            elif os.path.isfile(path):
                expanded.append(path)
            else:
                print(f"⚠️ Caminho não encontrado ou inválido: {path}")
        return expanded

    sources = expand_faq_sources(config["faq_documents"])

    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["indexing"]["chunk_size"],
        chunk_overlap=config["indexing"]["overlap"]
    )

    for path in sources:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        docs_split = text_splitter.split_documents(docs)
        documents.extend(docs_split)

    # 2. Criar índice local com a lógica centralizada no create_index()
    print("🧠 Criando índice local para FAQ...")
    faq_index = create_index(config)

    # 3. Criar um índice no DeepLake com os mesmos documentos
    print("📦 Criando índice DeepLake...")
    deeplake_path = config["storage"]["vector_store_path"]

    # Se force_reindex estiver ativado, apagar índice antigo
    if config.get("force_reindex", False):
        import shutil
        if os.path.exists(deeplake_path):
            print(f"🧹 Limpando índice anterior em: {deeplake_path}")
            shutil.rmtree(deeplake_path)
        else:
            print(f"📁 Nenhum índice anterior encontrado para remover.")

    store = DeepLakeVectorStore(dataset_path=deeplake_path, read_only=False, overwrite=True)

    embed_model = initialize_embedding_model(config)

    # Verificar compatibilidade entre dimensão do embedding atual e do DeepLake
    current_dim = get_embedding_dimension(embed_model)
    try:
        sample_vector = store.client.tensor("embedding").numpy()[0]
        stored_dim = sample_vector.shape[0]
        if current_dim != stored_dim:
            raise ValueError(f"Incompatibilidade de dimensão dos embeddings! "
                             f"Modelo atual: {current_dim}, armazenado: {stored_dim}.")
        print(f"✅ Dimensão compatível: {current_dim}")
    except Exception as e:
        print(f"⚠️ Não foi possível verificar a dimensão dos embeddings salvos: {e}")

    storage_context = StorageContext.from_defaults(vector_store=store)
    deeplake_index = VectorStoreIndex.from_documents(
        [Document(text=doc.page_content) for doc in documents],
        storage_context=storage_context,
        embed_model=embed_model)

    # 4. Retornar ambos os índices
    print("🔗 Unificando os dois índices...")
    return [faq_index, deeplake_index]
