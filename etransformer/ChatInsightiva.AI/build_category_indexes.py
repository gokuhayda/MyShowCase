"""
Módulo para criar índices vetoriais por categoria a partir de um JSON estruturado
e unificá-los em um ComposableGraph usando a LlamaIndex e DeepLake.
"""

import os
import json
import shutil
import unicodedata
from pathlib import Path
from datetime import datetime
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.composability import ComposableGraph
import time
from model_loader import initialize_llm_model, initialize_embedding_model
from utils_tools.config_loader import load_config

def normalize_nome_categoria(nome):
    """
    Normaliza o nome da categoria removendo acentos e caracteres especiais.

    Parâmetros:
        nome (str): Nome original da categoria.

    Retorna:
        str: Nome normalizado para uso em diretórios e identificadores.
    """
    return unicodedata.normalize("NFKD", nome).encode("ASCII", "ignore").decode("utf-8").strip().lower().replace(" ", "_")

def create_unified_graph():
    """
    Cria índices vetoriais separados por categoria e os unifica em um ComposableGraph.

    Retorna:
        ComposableGraph: Grafo unificado com todos os índices categorizados.
    """
    EXECUTION_TAG = str(int(datetime.now().timestamp()))
    config = load_config()
    DEEPLAKE_BASE = Path(config["storage"]["vector_store_path"])
    FAQ_JSON = Path(config["faq_json_path"])

    embedding_model = initialize_embedding_model(config)
    print(f"[DEBUG] Tipo de embedding_model usado: {type(embedding_model)}")
    llm = initialize_llm_model(config)

    with open(FAQ_JSON, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    docs_by_cat = {}
    for item in faqs:
        metadata = item["metadata"]
        cat = normalize_nome_categoria(metadata.get("categoria", "sem_categoria"))
        doc = Document(
            text=item["text"],
            metadata={
                "pergunta": metadata.get("pergunta", ""),
                "categoria": cat,
                "id": metadata.get("id", item["id"])
            }
        )
        docs_by_cat.setdefault(cat, []).append(doc)

    print("📦 Criando índices DeepLake segmentados por categoria...")
    indices = []
    summaries = []

    for cat, docs in docs_by_cat.items():
        dataset_path = str(DEEPLAKE_BASE / f"{cat}_{EXECUTION_TAG}")

        if os.path.exists(dataset_path):
            print(f"🧽 Removendo índice antigo em {dataset_path}...")
            shutil.rmtree(dataset_path)
#        time.sleep(2)
        test_vector = embedding_model.get_text_embedding(docs[0].text)
        expected_dim = len(embedding_model.get_text_embedding("teste"))
        if len(test_vector) != expected_dim:
            print(f"⚠️ Categoria '{cat}' ignorada: dimensão incorreta.")
            continue

        print(f"💾 Persistindo índice da categoria: {cat}")

        vector_store = DeepLakeVectorStore(
            dataset_path=dataset_path,
            read_only=False,
            overwrite=True
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        storage_context.persist(persist_dir=dataset_path)

        index = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context,
            embed_model=embedding_model,
            llm=llm,
            show_progress=True
        )

        indices.append(index)
        summaries.append(f"Base de conhecimento da categoria '{cat}'.")

    print("🔗 Unificando os índices no ComposableGraph...")
    filtered_indices = []
    filtered_summaries = []
    for idx, summary in zip(indices, summaries):
        vs = getattr(idx, "_vector_store", None)
        if hasattr(vs, "dataset_path") and EXECUTION_TAG in vs.dataset_path:
            filtered_indices.append(idx)
            filtered_summaries.append(summary)

    graph = ComposableGraph.from_indices(
         root_index_cls=VectorStoreIndex,
         children_indices=filtered_indices,
         index_summaries=filtered_summaries,
         embed_model=embedding_model,
         llm=llm
    )

    print("✅ Grafo unificado criado com sucesso.")
    return graph
