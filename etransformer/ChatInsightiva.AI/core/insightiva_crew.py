from core.crew_manager import create_agents_from_yaml, create_tasks_from_yaml
from utils_tools.config_loader import load_config
from crewai_tools import tools

class InsightivaCrew:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.file_paths = self.config.get("globals", {}).get("file_paths", {})

        self.available_tools = {
            "txt_search_tool": tools.TXTSearchTool(),
            "json_search_tool": tools.JSONSearchTool(),
        }

        self.agents = create_agents_from_yaml("config/agents_fit.yaml", self.available_tools, config=self.config)
        self.tasks = create_tasks_from_yaml("config/tasks_fit.yaml", self.agents)

    def crew(self):
        from crewai import Crew, Process
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.hierarchical, verbose=True)
