from langchain.tools import tool
from duckduckgo_search import ddg
from pymongo import MongoClient
from datetime import datetime
from geopy.geocoders import Nominatim
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Função para obter a hora atual
@tool
def get_current_time() -> str:
    """Retorna a hora atual no formato HH:MM:SS."""
    now = datetime.now()
    return now.strftime("%H:%M:%S")

# Função para obter a data atual
@tool
def get_current_date() -> str:
    """Retorna a data atual no formato YYYY-MM-DD."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d")

# Função para realizar buscas na internet usando DuckDuckGo
@tool
def duck_search(query: str) -> str:
    """Realiza uma busca no DuckDuckGo e retorna os resultados mais relevantes."""
    try:
        results = ddg(query, max_results=5)
        if not results:
            return "Nenhum resultado encontrado."
        response = "\n".join([f"{r['title']}: {r['href']}" for r in results])
        return response
    except Exception as e:
        return f"Erro ao realizar a busca: {str(e)}"

# Função para realizar buscas em um banco de dados MongoDB
def get_mongo_collection():
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    db = client[os.getenv("MONGO_DB", "meu_banco")]
    collection = db[os.getenv("MONGO_COLLECTION", "minha_colecao")]
    return collection

@tool
def search_mongo(query: str) -> str:
    """Realiza uma busca no MongoDB e retorna os resultados mais relevantes."""
    try:
        collection = get_mongo_collection()
        results = collection.find({"$text": {"$search": query}}).limit(5)
        response = []
        for result in results:
            response.append(f"ID: {result.get('_id')}, Dados: {result}")
        return "\n".join(response) if response else "Nenhum resultado encontrado."
    except Exception as e:
        return f"Erro ao consultar o MongoDB: {str(e)}"

# Função para obter localização baseada no IP (exemplo básico com geopy)
@tool
def get_location() -> str:
    """Retorna a localização aproximada com base no IP."""
    try:
        geolocator = Nominatim(user_agent="geoapi")
        location = geolocator.geocode("Your IP Address")
        return f"Você está próximo de {location.address}" if location else "Não foi possível determinar a localização."
    except Exception as e:
        return f"Erro ao determinar a localização: {str(e)}"

    