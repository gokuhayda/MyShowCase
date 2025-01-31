from typing import List, Dict
from crewai import Agent, Task
from . import logger
from crewai.tools.base_tool import BaseTool

from typing import List, Dict
from crewai import Agent
from . import logger
from crewai.tools.base_tool import BaseTool

def create_agents_from_yaml(config: dict, available_tools: Dict[str, BaseTool]) -> List[Agent]:
    """
    Gera instâncias de Agent a partir de um arquivo de configuração YAML.
    - Valida se as ferramentas definidas em cada agente realmente existem no dicionário 'available_tools'.
    - Lança exceções caso não encontre ferramentas ou dados inconsistentes.
    """
    agents = []
    for agent_data in config.get('agents', []):
        try:
            tools_from_yaml = agent_data.get('tools', [])
            print(f"Ferramentas especificadas para '{agent_data['role']}': {tools_from_yaml}")

            # Filtra ferramentas válidas
            assigned_tools = [available_tools[tool] for tool in tools_from_yaml if tool in available_tools]
            missing_tools = [tool for tool in tools_from_yaml if tool not in available_tools]
            
            if missing_tools:
                print(f"Ferramentas NÃO encontradas para '{agent_data['role']}': {missing_tools}")
            
            if not assigned_tools:
                raise ValueError(
                    f"Agente '{agent_data['role']}' não possui ferramentas válidas no YAML: {tools_from_yaml}"
                )
            
            groq_llm = "groq/llama-3.1-70b-versatile"
            agent_args = {
                "role": agent_data['role'],
                "goal": agent_data['goal'],
                "backstory": agent_data['backstory'],
                "verbose": agent_data.get('verbose', False),
                "allow_delegation": agent_data.get('allow_delegation', True),
                "tools": assigned_tools,
                "human_input":[False]
                #"llm":[groq_llm] #agent com difrentes LLM
            }

            if "output_pydantic" in agent_data:
                agent_args["output_pydantic"] = agent_data["output_pydantic"]

            if "human_input" in agent_data:
                agent_args["human_input"] = agent_data["human_input"]

            if "allow_code_execution" in agent_data:
                agent_args["allow_code_execution"] = agent_data["allow_code_execution"]

            agent = Agent(**agent_args)
            agents.append(agent)
        except Exception as e:
            print(f"Erro ao criar agente '{agent_data.get('role', 'DESCONHECIDO')}': {e}")
    return agents


def create_tasks_from_yaml(config: dict, agents: List[Agent]) -> List[Task]:
    """
    Gera instâncias de Task a partir de um arquivo de configuração YAML.
    - Encontra o agente correspondente ao 'agent_role' definido em cada task.
    - Ordena as tarefas por prioridade (Alta, Média, Baixa).
    """
    if not isinstance(config, dict):
        raise TypeError("Config deve ser um dicionário carregado do YAML, mas não é.")

    tasks = []
    task_priorities = {}

    for task_name, task_data in config.get('tasks', {}).items():
        if not isinstance(task_data, dict):
            raise TypeError(
                f"Cada tarefa em 'tasks' deve ser um dicionário. Encontrado: {type(task_data)} na chave {task_name}"
            )

        agent_role = task_data.get('agent_role')
        if not agent_role:
            print(f"A tarefa '{task_name}' não possui 'agent_role'. Será ignorada.")
            continue

        # Acha o agente correspondente
        agent = next((a for a in agents if a.role == agent_role), None)
        if not agent:
            raise ValueError(
                f"Agente com a role '{agent_role}' não foi encontrado. "
                f"Roles disponíveis: {[a.role for a in agents]}"
            )

        # Cria a tarefa
        task = Task(
            description=task_data['description'],
            expected_output=task_data['expected_output'],
            agent=agent
        )

        if 'context' in task_data:
            task["context"] = task["context"]
        tasks.append(task)
        # Armazena a prioridade (padrão: Média)
        task_priorities[task] = task_data.get('priority', 'Média')

    # Ordena pela prioridade
    priority_order = {"Alta": 1, "Média": 2, "Baixa": 3}
    tasks.sort(key=lambda t: priority_order.get(task_priorities[t], 2))

    return tasks
