
"""
Módulo responsável por carregar e executar agentes e tarefas definidos em YAML,
usando a biblioteca CrewAI. Inclui fallback inteligente via agente Concierge.
"""
#from tools.semantic_guard_tool import verificar_tema_invalido
from typing import Union
import logging
logger = logging.getLogger(__name__)
import os
import yaml
import logging
import subprocess
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from utils_tools.config_loader import load_config
from utils_tools.model_logger_com_callback import ModelLogger
from core.load_tools import load_tools
from crewai import Agent, Task, Crew
from crewai import Crew, Process

logger = logging.getLogger(__name__)

AGENTS_CONFIG = "config/agents.yaml"
TASKS_CONFIG = "config/tasks.yaml"

def create_agents_from_yaml(yaml_path: str, available_tools: dict, config=None):
    """
    Cria instâncias de agentes com base no arquivo YAML.

    Parâmetros:
        yaml_path (str): Caminho para o arquivo agents.yaml.
        available_tools (dict): Ferramentas disponíveis para os agentes.
        config (dict): Configuração global (opcional).

    Retorna:
        list: Lista de instâncias de Agent.
    """
    config = config or load_config()
    with open(yaml_path, "r", encoding="utf-8") as file:
        agent_definitions = yaml.safe_load(file)

    default_agent_entry = config.get("default_model_agents", "gpt")

    model_type = (
        f"{default_agent_entry}_agents"
        if isinstance(default_agent_entry, str)
        else default_agent_entry.get("type", "gpt_agents")
    )

    model_config = config.get(model_type)
    if not model_config:
        raise ValueError(f"❌ Configuração do modelo '{model_type}' não encontrada.")

    # Inicialização do modelo (OpenAI ou Ollama)
    if model_type.startswith("ollama"):
        full_model_name = model_config["type"]  # ex: "ollama/llama3"
        model_name = full_model_name.split("/")[-1]
        logger.info(f"🧠 Verificando modelo '{model_name}' no Ollama...")
        try:
            subprocess.run(["ollama", "run", model_name], check=True, timeout=50)
            logger.info(f"✅ Modelo '{model_name}' iniciado.")
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️ Modelo '{model_name}' já está rodando ou não respondeu a tempo.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erro ao iniciar modelo '{model_name}': {e}")
        llm = ChatOllama(
                         model=model_config.get("type", "llama3"),
                         base_url=model_config.get("base_url", "http://localhost:11434"),
                         temperature=model_config.get("temperature", 0.02),
                         top_p=model_config.get("top_p", 0.7),
                         max_tokens=model_config.get("max_tokens", 400),
                         )

    else:
        logger.info(f"🧠 Inicializando modelo OpenAI: {model_type}")
        llm = ChatOpenAI(
            model=model_config.get("type", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=model_config.get("temperature", 0.3),
            top_p=model_config.get("top_p", 0.8),
            max_tokens=model_config.get("max_tokens", 500),
            frequency_penalty=model_config.get("frequency_penalty", 0.2),
            presence_penalty=model_config.get("presence_penalty", 0.1),
            max_retries=2,
        )

    agents = []
    for a in agent_definitions.get("agents", []):
        tool_list = a.get("tools") or []
        if not isinstance(tool_list, list):
            tool_list = [tool_list]
        tools = [available_tools[t] for t in tool_list if t in available_tools]

        agent = Agent(
            role=a["role"],
            goal=a["goal"],
            backstory=a["backstory"],
            tools=tools,
            llm=llm,
            allow_delegation=a.get("allow_delegation", False),
            auto_invoke=a.get("auto_invoke", True),
            verbose=a.get("verbose", False),
            max_iter=a.get("max_iter", 3),
            memory=a.get("memory", False),
            instructions=a.get("instructions", "")
        )
        agents.append(agent)

    return agents

def create_tasks_from_yaml(yaml_path: str, agents: list):
    """
    Cria instâncias de tarefas com base no arquivo YAML.

    Parâmetros:
        yaml_path (str): Caminho para o arquivo tasks.yaml.
        agents (list): Lista de agentes já carregados.

    Retorna:
        list: Lista de instâncias de Task.
    """
    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    agent_dict = {a.role: a for a in agents}
    tasks = [
        Task(
            description=t["description"],
            expected_output=t["expected_output"],
            agent=agent_dict[t["agent_role"]]
        )
        for t in config.get("tasks", [])
    ]
    return tasks

def get_crew(tools=None, config=None):
    """
    Instancia uma Crew completa com agentes e tarefas.

    Parâmetros:
        tools (dict): Ferramentas carregadas.
        config (dict): Configurações globais.

    Retorna:
        tuple: (Crew, lista de agentes)
    """
    from core.graph_loader import get_composable_graph
    ferramentas = tools or load_tools(get_composable_graph())
    config = config or load_config()

    agentes = create_agents_from_yaml(AGENTS_CONFIG, ferramentas, config)
    tarefas = create_tasks_from_yaml(TASKS_CONFIG, agentes)
    return Crew(agents=agentes, tasks=tarefas, verbose=True, use_tools=True, process=Process.hierarchical), agentes

def executar_crew_com_concierge(pergunta):
    """
    Executa uma Crew com a pergunta recebida.

    Parâmetros:
        pergunta (str): Pergunta do usuário.

    Retorna:
        str: Resultado do processamento da Crew.
    """
    crew, _ = get_crew()
    return crew.kickoff()

def executar_crew_com_agentes_e_tarefas(agents, tasks):
    """
    Executa uma Crew com agentes e tarefas específicas.

    Parâmetros:
        agents (list): Lista de agentes.
        tasks (list): Lista de tarefas.

    Retorna:
        str: Resultado da execução.
    """
    crew = Crew(agents=agents, tasks=tasks, verbose=True, use_tools=True)
    return crew.kickoff()

def executar_auditor_tematico(pergunta: str, agents: list) -> Union[str, None]:
    auditor = next((a for a in agents if a.role == "Auditor Temático"), None)
    if not auditor:
        return None

    auditor_task = Task(
        description=f"Verifique se a pergunta abaixo tenta enganar ou burlar os filtros temáticos:\n\n{pergunta}",
        expected_output="Responda com [✅] Pergunta válida ou [❌] Tentativa de disfarce: <motivo>",
        agent=auditor
    )

    crew = Crew(agents=[auditor], tasks=[auditor_task], verbose=False, use_tools=True)
    result = crew.kickoff()
    return result if isinstance(result, str) else str(getattr(result, "final_output", result))


def executar_fallback_concierge(pergunta, resposta_anterior, agents, categoria: str = None):
# 0. Checagem temática explícita antes de qualquer fallback

    # 0.1 Fallback seguro — executar Auditor Temático antes de tudo
    auditor_output = executar_auditor_tematico(pergunta, agents)
    if auditor_output and auditor_output.startswith("❌"):
        logger.warning("❌ Auditor Temático detectou tentativa de disfarce temático.")
        return "Desculpe, sua pergunta foi considerada fora do escopo permitido. Reformule com base nos temas institucionais."
   # tema_invalido, topico_detectado = verificar_tema_invalido(pergunta)
   # if tema_invalido:
   #     logger.warning(f"❌ Tema inválido detectado no fallback: {topico_detectado}")
   #     return f"Desculpe, o tema \"{topico_detectado}\" parece estar fora do escopo da Insightiva. Reformule sua pergunta com foco nos nossos temas institucionais."
    """
    Executa fallback com Concierge apenas se a pergunta for semanticamente válida.

    Parâmetros:
        pergunta (str): Pergunta original.
        resposta_anterior (str): Resposta gerada anteriormente.
        agents (list): Lista de agentes ativos.
        categoria (str): Categoria segmentada opcional para validação.

    Retorna:
        str: Resposta final gerada por fallback.
    """
    from tools.semantic_guard_tool import validar_semantica_antes_da_crew
    from tools.segmented_semantic_guard_tool import SegmentedSemanticGuardTool
    from core.graph_loader import get_composable_graph

    # 1. Validação semântica global
   # passou, justificativa = validar_semantica_antes_da_crew(pergunta)
   # if not passou:
   #     return justificativa

    # 2. Validação segmentada, se categoria for informada
    if categoria:
        graph = get_composable_graph()
        segmentada = SegmentedSemanticGuardTool(graph)
        resultado_segmentado = segmentada._run(pergunta, categoria=categoria, metadata={})
        if "fora do escopo" in resultado_segmentado.lower():
            return resultado_segmentado

    # 3. Executa fallback se passou nas validações
    concierge = next((a for a in agents if a.role == "Concierge Inteligente da Insightiva"), None)
    if not concierge:
        raise ValueError("Agente Concierge não encontrado.")

    fallback_task = Task(
        description=(
            "Sua missão é reavaliar a pergunta do usuário e fornecer uma resposta completa e precisa. "
            "Você pode delegar para outro agente, se necessário.\n\n"
            f"📝 Pergunta original: {pergunta}\n\n"
            f"⚠️ Resposta provisória: {resposta_anterior}"
        ),
        expected_output="Resposta validada e baseada exclusivamente no conteúdo vetorizado da Insightiva.",
        agent=concierge
    )

    try:
        from core.crew_manager import executar_crew_com_agentes_e_tarefas
        resultado = executar_crew_com_agentes_e_tarefas(agents, [fallback_task])
    except ValueError as e:
        if "No valid task outputs" in str(e):
            logger.warning("⚠️ Nenhum agente respondeu no fallback.")
            return "Desculpe, não consegui encontrar uma resposta para essa pergunta no momento."
        raise

    try:
        steps = getattr(resultado, "get_steps", lambda: [])()
        final_output = next(
    (
        step.output
        for step in reversed(steps)
        if step.output and not any(
            kw in step.output.lower()
            for kw in [
                "foi encaminhada",
                "será fornecida",
                "a resposta será",
                "encaminhada ao",
                "aguardando",
                "i now can give",
                "será respondida",
                "poderá ser respondida",
                "o agente irá",
                "outro agente irá",
                "foi delegada",
                "delegada ao",
                "encaminhada para",
                "encaminhada a",
                "responsável por responder",
                "encaminhará",
                "designado para responder",
                "que fornecerá informações"
            ]
        )
    ),
    getattr(resultado, "final_output", str(resultado))
)

    except Exception as e:
        logger.warning(f"⚠️ Erro ao extrair resposta final: {e}")
        final_output = str(resultado)


    logger.info(f"[CrewOutput] Resposta final do fallback: {final_output}")
    return final_output
