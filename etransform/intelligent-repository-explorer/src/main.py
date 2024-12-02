import re
import textwrap
import yaml
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core import SimpleDirectoryReader
from llama_index.core.vector_stores import SimpleVectorStore
import openai
from openai import OpenAIError, BadRequestError
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import nest_asyncio
import asyncio
import os 
import json

# Aplicar nest_asyncio para evitar conflitos de loop
nest_asyncio.apply()

# Carregar variáveis de ambiente
load_dotenv()

# Validar tokens
github_token = os.getenv("GITHUB_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not github_token or not openai_api_key:
    raise EnvironmentError("Certifique-se de que as variáveis GITHUB_TOKEN e OPENAI_API_KEY estão configuradas corretamente.")

# Configurar a chave de API do OpenAI
openai.api_key = openai_api_key

# Função para carregar configurações do YAML
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Função para analisar a URL do repositório
def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)

# Garantir que o diretório de persistência exista
import json

def ensure_directory_exists(path):
    """
    Garante que o diretório especificado e os arquivos necessários existam.
    """
    if not os.path.exists(path):
        print(f"[INFO] Diretório {path} não encontrado. Criando...")
        os.makedirs(path, exist_ok=True)

    # Verificar e criar arquivos vazios, se necessário
    files_to_check = ['index_store.json', 'docstore.json']
    for file_name in files_to_check:
        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            print(f"[INFO] Arquivo {file_path} não encontrado. Criando arquivo vazio...")
            with open(file_path, 'w') as f:
                json.dump({}, f)
    print("[DEBUG] Verificação de diretório e arquivos concluída.")

# Inicializar o leitor do GitHub
def initialize_github_client():
    return GithubClient(github_token)

def initialize_query_engine(config, docs):
    """
    Inicializa o mecanismo de consulta com base na configuração YAML.
    Cria automaticamente o índice e os arquivos de persistência necessários.
    """
    storage_path = config["storage"]["path"]
    persist = config["storage"]["persist"]
    overwrite = config["storage"]["overwrite"]

    # Garantir que o diretório e arquivos necessários existam
    ensure_directory_exists(storage_path)

    print(f"[DEBUG] Caminho de armazenamento: {storage_path}")
    print(f"[DEBUG] Persistência habilitada: {persist}, Sobrescrever: {overwrite}")

    try:
        # Tentar carregar o índice existente
        if persist and not overwrite:
            print("[INFO] Tentando carregar índice persistente existente...")
            storage_context = StorageContext.from_defaults(persist_dir=storage_path)
            index = load_index_from_storage(storage_context)
            print("[INFO] Índice carregado com sucesso!")
        else:
            print("[INFO] Criando um novo índice devido a sobrescrição ou ausência de índice...")
            raise FileNotFoundError
    except (FileNotFoundError, ValueError) as e:
        print(f"[AVISO] Arquivo de persistência não encontrado ou inválido: {e}")
        print("[INFO] Criando um novo índice e arquivos de persistência...")

        # Criação do StorageContext com vector_store padrão
        storage_context = StorageContext.from_defaults(
            persist_dir=storage_path,
            vector_store=SimpleVectorStore()  # Define o armazenamento vetorial padrão
        )
        index = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context,
        )

        if persist:
            index.storage_context.persist()
            print("[INFO] Novo índice criado e persistido com sucesso!")

    return index.as_query_engine()


async def query_llm(query, config):
    """
    Realiza uma consulta ao modelo LLM configurado (OpenAI GPT ou Llama) usando LangChain.
    """
    if config["llm"]["use_openai"]:
        print(f"[LOG] Usando modelo OpenAI com LangChain: {config['llm']['openai_model']}")
        try:
            # Inicializar o modelo de chat OpenAI através do LangChain
            chat = ChatOpenAI(
                model_name=config["llm"]["openai_model"],
                temperature=0.7,
                max_tokens=500,
                openai_api_key=openai_api_key
            )

            # Definir as mensagens do chat
            messages = [
                SystemMessage(content="Você é um assistente especializado em analisar este repositório. Responda apenas com base nas informações nele contidas. Se a pergunta estiver fora do contexto do repositório, informe que não pode responder."),
                HumanMessage(content=query)
            ]

            # Realizar a chamada ao modelo
            response = chat(messages)

            # Obter o conteúdo da resposta
            answer = response.content
            return answer.strip()
        except BadRequestError as e:
            print(f"[ERRO] Bad request to OpenAI GPT: {e}")
            return "Houve um erro ao processar sua consulta com o OpenAI GPT-4 devido a uma requisição inválida."
        except OpenAIError as e:
            print(f"[ERRO] General OpenAI API error: {e}")
            return "Houve um erro ao processar sua consulta com o OpenAI GPT-4."
        except Exception as e:
            print(f"[ERRO] Unexpected error: {e}")
            return "Houve um erro inesperado ao processar sua consulta."
    else:
        # Usar modelo Llama com LangChain
        try:
            print(f"[LOG] Usando modelo Llama com LangChain: {config['llm']['llama_model_name']}")
            tokenizer = AutoTokenizer.from_pretrained(config["llm"]["llama_model_name"])
            model = AutoModelForCausalLM.from_pretrained(config["llm"]["llama_model_name"])

            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))

            # Criar um pipeline de geração de texto
            pipe = pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id
            )

            # Envolver o pipeline com o LangChain
            llm = HuggingFacePipeline(pipeline=pipe)

            # Adicionar uma mensagem inicial ao prompt
            system_message = "Você é um assistente especializado em analisar este repositório. Responda apenas com base nas informações nele contidas. Se a pergunta estiver fora do contexto do repositório, informe que não pode responder..\n\n"
            full_query = system_message + query

            # Utilizar o LangChain para gerar a resposta
            response = llm(full_query)
            return response.strip()
        except Exception as e:
            print(f"[ERRO] Erro ao usar modelo Llama com LangChain: {e}")
            return "Houve um erro ao processar sua consulta com o modelo Llama via LangChain."

async def main():
    # Carregar configurações do YAML
    config = load_config()

    github_client = initialize_github_client()

    # Ler URL do repositório do arquivo YAML ou entrada do usuário
    github_url = config["github"]["repo_url"] or input("Insira a URL do repositório do GitHub: ")
    owner, repo = parse_github_url(github_url)

    if not owner or not repo:
        print("URL inválida. Tente novamente.")
        return

    # Carregar dados do repositório
    loader = GithubRepositoryReader(
        github_client,
        owner=owner,
        repo=repo,
        filter_file_extensions=(
            config["github"]["extensions"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        verbose=False,
        concurrent_requests=config["github"]["concurrent_requests"],
    )

    print(f"Carregando o repositório {repo} do usuário {owner}...")
    docs = loader.load_data(branch=config["github"]["branch"])

    if not docs:
        print("Nenhum documento encontrado no repositório.")
        return

    print("Documentos carregados:")
    for doc in docs:
        print(doc.metadata)

    query_engine = initialize_query_engine(config, docs)

    intro_question = config["llm"]["test_question"]
    print(f"Pergunta de teste: {intro_question}")
    print("=" * 50)
    answer = query_engine.query(intro_question)
    print(f"Resposta do índice: {textwrap.fill(str(answer), 100)} \n")

    while True:
        user_question = input("Insira sua pergunta (ou digite 'sair' para encerrar): ")
        if user_question.lower() == "sair":
            print("Encerrando. Obrigado!")
            break

        print(f"Sua pergunta: {user_question}")
        print("=" * 50)

        # Utilize o query_engine para processar a pergunta
        answer = query_engine.query(user_question)
        print(f"Resposta do índice: {textwrap.fill(str(answer), 100)} \n")


if __name__ == "__main__":
    asyncio.run(main())

