# poem_crew.py

import os
import yaml
import logging
from typing import Any, Dict, List, Callable
from crewai import Crew, Agent, Task
from crewai.project import CrewBase, crew
from crewai import Process
from core.utils import load_yaml_config, load_project_description 
from core.crew_manager import create_agents_from_yaml, create_tasks_from_yaml
from tools.human_input_tool import HumanInputTool
from tools.yaml_validator import YamlValidatorTool
from crewai_tools import tools
from langchain_community.chat_models import ChatOpenAI
from logging.handlers import RotatingFileHandler
from llama_index.core.text_splitter import SentenceSplitter
import tiktoken
# poem_crew.py - Adicionar no início do arquivo
import litellm
litellm.set_verbose = True  # Habilita logging detalhado

from dotenv import load_dotenv
load_dotenv()


# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# # Configurações do ambiente
# hf_token = os.getenv("HUGGINGFACE_TOKEN")
# device = os.getenv("DEVICE", "auto")  # Default para "auto"

# # Carrega modelo e tokenizer
# model_name = os.getenv("HUGGINGFACE_MODEL")
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     token=hf_token,
#     device_map=device,
#     torch_dtype="auto"
# )

# llama_chat = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     temperature=0.7,
#     max_new_tokens=1024
# )

available_tools = {
    "human_tool": HumanInputTool(),
    "code_docs_search_tool": tools.CodeDocsSearchTool(),
    "code_interpreter_tool": tools.CodeInterpreterTool(),
    "csv_search_tool": tools.CSVSearchTool(),
    "dalle_tool": tools.DallETool(),
    "directory_read_tool": tools.DirectoryReadTool(result_as_answer=True),
    "directory_search_tool": tools.DirectorySearchTool(),
    "docx_search_tool": tools.DOCXSearchTool(),
    "exa_tools": tools.EXASearchTool(),
    "file_read_tool": tools.FileReadTool(),
    "file_writer_tool": tools.FileWriterTool(),
    "json_search_tool": tools.JSONSearchTool(),
    "semantic_search_tool": tools.MDXSearchTool(),
    "rag_tool": tools.RagTool(),
    "scrape_element_from_website_tool": tools.ScrapeElementFromWebsiteTool(),
    "scrape_tool": tools.ScrapeWebsiteTool(),
    "selenium_scraping_tool": tools.SeleniumScrapingTool(),
    "search_tool": tools.SerperDevTool(),
    "txt_search_tool": tools.TXTSearchTool(),
    'yaml_validator_tool': YamlValidatorTool(),
}

class MLOpsCrew:
    """Crew para o fluxo de MLOps"""

    def __init__(self, config: dict):
        self.llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.7) 
        self.agents_config: List[Dict[str, Any]] = []
        self.tasks_config: Dict[str, Any] = {}
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        try:
            self.agents_config = load_yaml_config(os.path.join(BASE_DIR, "config", "agents.yaml"))
            self.tasks_config = load_yaml_config(os.path.join(BASE_DIR, "config", "tasks.yaml"))
            print("Configurações carregadas com sucesso (agents.yaml e tasks.yaml).")
        except (FileNotFoundError, ValueError) as e:
            print(f"Erro ao carregar configurações de YAML: {e}")
            raise

        # Carrega configurações globais do YAML
        globals_config = config.get("globals", {})
        file_paths = globals_config.get("file_paths", {})

        self.project_description_path = file_paths.get("project_description", "")
        self.report_discovery_path = file_paths.get("file_report_discovery", "")
        self.report_assessment_path = file_paths.get("file_report_assessment", "")
        self.report_pipeline_design_path = file_paths.get("file_report_pipeline_design", "")
        self.report_deployment_path = file_paths.get("file_report_deployment", "")
        self.report_model_development_path = file_paths.get("file_report_model_development", "")

        self.log_discovery_path = file_paths.get("file_log_discovery", "")
        self.log_assessment_path = file_paths.get("file_log_assessment", "")
        self.log_pipeline_design_path = file_paths.get("file_log_pipeline_design", "")
        self.log_deployment_path = file_paths.get("file_log_deployment", "")
        self.log_model_development_path = file_paths.get("file_log_model_development", "")

        self.sources_path_general = file_paths.get("file_path_sources", "")
        self.log_path = file_paths.get("log_path", "/home/goku/Documentos/mlops_consulting/mlops_consulting_flow/logs")

        # Garante que todos os diretórios existam
        self.ensure_directories_exist(file_paths)

        # Configuração do Logging com Rotação para cada fase
        self.setup_logging()

        # Criação dos agentes com logging e gerenciamento de input
        self.agents = create_agents_from_yaml(
            self.agents_config, 
            available_tools, 
        )
        self.tasks = create_tasks_from_yaml(
            self.tasks_config, 
            self.agents, 
        )

    def ensure_directories_exist(self, file_paths: Dict[str, str]):
        """Garante que todos os diretórios especificados existam."""
        dirs = set()
        for path in file_paths.values():
            # Verifica se o caminho é um arquivo ou diretório
            if os.path.isfile(path):
                dir_path = os.path.dirname(path)
            else:
                dir_path = path
            dirs.add(dir_path)
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Diretório garantido: {dir_path}")

    def setup_logging(self):
        """Configura o sistema de logging com rotação de logs para cada fase."""
        # Configura logger principal
        self.logger = logging.getLogger("MLOpsCrew")
        self.logger.setLevel(logging.INFO)
        
        # Evita a adição múltipla de handlers
        if not self.logger.handlers:
            # Handler geral
            general_log_path = os.path.join(self.log_path, "general.log")
            handler = RotatingFileHandler(
                general_log_path,
                maxBytes=5*1024*1024,  # 5 MB
                backupCount=5
            )
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.info("Sistema de logging configurado com rotação para logs gerais.")

        # Configura loggers específicos para cada fase
        self.loggers = {}
        phase_logs = {
            "discovery_phase": self.log_discovery_path,
            "assessment_phase": self.log_assessment_path,
            "pipeline_design_phase": self.log_pipeline_design_path,
            "model_development_phase": self.log_model_development_path,
            "deployment_phase": self.log_deployment_path,
        }

        for phase, log_dir in phase_logs.items():
            log_file = os.path.join(log_dir, f"{phase}.log")
            phase_logger = logging.getLogger(phase)
            phase_logger.setLevel(logging.INFO)
            if not phase_logger.handlers:
                handler = RotatingFileHandler(
                    log_file,
                    maxBytes=5*1024*1024,  # 5 MB
                    backupCount=5
                )
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                phase_logger.addHandler(handler)
                phase_logger.info(f"Sistema de logging configurado com rotação para {phase}.")
            self.loggers[phase] = phase_logger

    def get_logger_for_phase(self, phase: str) -> logging.Logger:
        """Retorna o logger específico para a fase."""
        return self.loggers.get(phase, self.logger)

    def manage_input_size(self, input_data: str) -> (List[str], bool):
        """
        Verifica se o input excede o tamanho máximo permitido.
        Se exceder, trunca o input e retorna uma flag indicando truncamento.
        """
        max_input_size = 4096  # Este valor pode ser parametrizado
        if len(input_data) > max_input_size:
            truncated_input = input_data[:max_input_size]
            self.logger.warning(f"Input truncado de {len(input_data)} para {max_input_size} caracteres.")
            return [truncated_input], True
        return [input_data], False



    def chunk_text_semantic(self, text: str, max_tokens: int = 12000, chunk_size: int = 3500) -> list:
        """
        Segmenta o texto de forma semântica, garantindo que cada chunk não ultrapasse 'max_tokens'.
        
        - Usa SentenceSplitter para criar chunks significativos.
        - Ajusta o tamanho final garantindo que o total de tokens não exceda 'max_tokens'.
        """

        def count_tokens(text: str, model: str = "gpt-4") -> int:
            """
            Conta o número de tokens em um texto usando o modelo especificado.
            """
            encoder = tiktoken.encoding_for_model(model)
            return len(encoder.encode(text))
        
        if not text:
            return []

        # Divide o texto em chunks semânticos
        text_splitter = SentenceSplitter(chunk_size=chunk_size)
        semantic_chunks = text_splitter.split_text(text)

        final_chunks = []
        current_chunk = []
        current_count = 0

        for sentence in semantic_chunks:
            sentence_tokens = count_tokens(sentence)

            # Se adicionar essa frase ultrapassar o limite, armazena o chunk atual e inicia um novo
            if current_count + sentence_tokens > max_tokens:
                final_chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_count = 0

            current_chunk.append(sentence)
            current_count += sentence_tokens

        # Adiciona o último chunk restante, se houver
        if current_chunk:
            final_chunks.append(' '.join(current_chunk))

        return final_chunks


    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,  
            tasks=self.tasks,
            manager_llm=self.llm,
            process=Process.hierarchical,
            memory=False,
            verbose=True
        )
