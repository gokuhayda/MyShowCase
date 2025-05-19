
from llama_index.core import Document, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from utils_tools.vector_store_helpers import should_rebuild_index, clear_deeplake_vector_store
import os

def split_large_document(document, max_size=5000):
    text = document.text
    chunks = [text[i:i + max_size] for i in range(0, len(text), max_size)]
    return [Document(text=chunk, metadata=document.metadata) for chunk in chunks]

def prepare_documents_storage(config):
    """
    Lê documentos processados, fragmenta quando necessário e armazena vetorialmente no DeepLake.

    Parâmetros:
        config (dict): Configurações com caminhos e limites.

    Retorna:
        tuple: (lista de documentos, contexto de armazenamento)
    """
    from pathlib import Path
    import json

    processed_texts_dir = config["storage"]["processed_texts_directory"]
    if not os.path.exists(processed_texts_dir):
        raise FileNotFoundError(f"❌ Diretório {processed_texts_dir} não encontrado.")

    if should_rebuild_index(config):
        clear_deeplake_vector_store(config["storage"]["vector_store_path"])

    print("📥 Lendo documentos da pasta de textos processados...")

    documents = []
    for filepath in Path(processed_texts_dir).glob("*"):
        ext = filepath.suffix.lower()
        if ext == ".txt":
            text = filepath.read_text(encoding="utf-8")
            documents.append(Document(text=text, metadata={"filename": filepath.name}))
        elif ext == ".json":
            try:
                try:
                    content = filepath.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    content = filepath.read_text(encoding="utf-8-sig")

                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        pergunta = item.get("pergunta", "")
                        resposta = item.get("resposta", "")
                        contexto = item.get("contexto", "")
                        categoria = item.get("categoria", "")
                        text = f"📌 Categoria: {categoria}\n📚 Contexto: {contexto}\n\n❓ {pergunta}\n💬 {resposta}"
                        documents.append(Document(
                            text=text,
                            metadata={
                                "filename": filepath.name,
                                "id": item.get("id", ""),
                                "categoria": categoria
                            }
                        ))
                else:
                    documents.append(Document(text=str(data), metadata={"filename": filepath.name}))
            except Exception as e:
                print(f"⚠️ Erro ao ler {filepath.name}: {e}")
        else:
            try:
                docs = SimpleDirectoryReader(input_files=[str(filepath)]).load_data()
                documents.extend([Document(text=doc.text, metadata=doc.metadata) for doc in docs])
            except Exception as e:
                print(f"⚠️ Ignorado {filepath.name}: erro ao carregar: {e}")

    print(f"📄 {len(documents)} documentos carregados.")

    fragmented_documents = []
    for doc in documents:
        if len(doc.text) > config['setup_docs']['max_document_size']:
            print(f"✂️ Dividindo documento grande: {len(doc.text)} caracteres.")
            fragmented_documents.extend(
                split_large_document(doc, max_size=config['setup_docs']['max_document_size'])
            )
        else:
            fragmented_documents.append(doc)

    documents = fragmented_documents
    print(f"🔍 Total de fragmentos para indexação: {len(documents)}")

    deeplake_path = config["storage"]["vector_store_path"]
    print(f"💾 Inicializando DeepLake em {deeplake_path}...")

    vector_store = DeepLakeVectorStore(
        dataset_path=deeplake_path,
        overwrite=True,
        tensor_config={"text": {"max_shape": (500, 768)}}
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store).persist(
        persist_dir=deeplake_path
    )

    print("✅ Armazenamento inicializado com sucesso.")
    return documents, storage_context

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

def process_documents(config):
    """
    Lê e fragmenta documentos da pasta de textos processados e armazena no DeepLake.

    Parâmetros:
        config (dict): Configurações com diretórios e limites.

    Retorna:
        tuple: (lista de documentos, contexto de armazenamento)
    """
    processed_texts_dir = config["storage"]["processed_texts_directory"]
    if not os.path.exists(processed_texts_dir):
        raise FileNotFoundError(f"❌ Diretório {processed_texts_dir} não encontrado.")

    if should_rebuild_index(config):
        clear_deeplake_vector_store(config["storage"]["vector_store_path"])

    print("📥 Lendo documentos da pasta de textos processados...")
    documents = SimpleDirectoryReader(processed_texts_dir).load_data()
    documents = [Document(text=doc.text, metadata=doc.metadata) for doc in documents]

    print(f"📄 {len(documents)} documentos carregados.")

    fragmented_documents = []
    for doc in documents:
        if len(doc.text) > config['setup_docs']['max_document_size']:
            print(f"✂️ Dividindo documento grande: {len(doc.text)} caracteres.")
            fragmented_documents.extend(
                split_large_document(doc, max_size=config['setup_docs']['max_document_size'])
            )
        else:
            fragmented_documents.append(doc)

    documents = fragmented_documents
    print(f"🔍 Total de fragmentos para indexação: {len(documents)}")

    deeplake_path = config["storage"]["vector_store_path"]
    print(f"💾 Inicializando DeepLake em {deeplake_path}...")

    vector_store = DeepLakeVectorStore(
        dataset_path=deeplake_path,
        overwrite=True,
        tensor_config={"text": {"max_shape": (500, 768)}}
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store).persist(
        persist_dir=deeplake_path
    )

    print("✅ Armazenamento inicializado com sucesso.")
    return documents, storage_context