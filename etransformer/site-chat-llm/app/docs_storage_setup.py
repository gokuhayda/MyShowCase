
from llama_index.core import Document, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
import os

def split_large_document(document, max_size=5000):
    """
    Divide um documento grande em partes menores.

    Args:
        document (Document): Documento original.
        max_size (int): Tamanho máximo permitido para cada parte.

    Returns:
        list: Lista de novos documentos fragmentados.
    """
    text = document.text
    chunks = [text[i:i + max_size] for i in range(0, len(text), max_size)]
    return [Document(text=chunk, metadata=document.metadata) for chunk in chunks]

def prepare_documents_storage(config):
    processed_texts_dir = config["storage"]["processed_texts_directory"]
    if not os.path.exists(processed_texts_dir):
        raise FileNotFoundError(f"Diretório {processed_texts_dir} não encontrado.")

    documents = SimpleDirectoryReader(processed_texts_dir).load_data()
    documents = [Document(text=doc.text, metadata=doc.metadata) for doc in documents]

    fragmented_documents = []
    for doc in documents:
        if len(doc.text) > config['setup_docs']['max_document_size']:
            print(f"Dividindo documento grande: {len(doc.text)} caracteres.")
            fragmented_documents.extend(split_large_document(doc, max_size=config['setup_docs']['max_document_size']))
        else:
            fragmented_documents.append(doc)

    documents = fragmented_documents

    deeplake_path = config["storage"]["vector_store_path"]
    print(f"Inicializando DeepLake em {deeplake_path}...")

    vector_store = DeepLakeVectorStore(
        dataset_path=deeplake_path,
        overwrite=True,
        tensor_config={"text": {"max_shape": (500, 768)}}  # Ajustar dimensões conforme necessário
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return documents, storage_context