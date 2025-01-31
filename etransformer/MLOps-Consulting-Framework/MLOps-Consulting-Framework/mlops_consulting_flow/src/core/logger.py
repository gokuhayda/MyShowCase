import logging
import os

# Diretório para salvar os logs
log_directory = os.path.join(os.getcwd(), "..", "..", "logs")
os.makedirs(log_directory, exist_ok=True)  # Certifique-se de que o diretório existe

# Configuração do arquivo de log
log_file_path = os.path.join(log_directory, "mlops_consulting.log")
logging.basicConfig(
    level=logging.INFO,  # Ou DEBUG para mais detalhes
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()  # Também exibe no console, opcional
    ]
)

logger = logging.getLogger(__name__)
