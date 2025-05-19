import yaml
import asyncio
from crewai.flow.flow import Flow, listen, start
from core.insightiva_crew import InsightivaCrew
from core.utils import load_project_description

class InsightivaConsultingFlow(Flow):
    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.crew_instance = InsightivaCrew(config_path=config_path)

    @start()
    def initialize_diagnosis(self):
        description_path = self.crew_instance.file_paths.get("project_description", "")
        content, parsed_input = load_project_description(description_path)
        return {
            "context": content,
            "parsed_input": parsed_input,
            "log_path": self.crew_instance.file_paths.get("file_log_discovery", ""),
            "report_path": self.crew_instance.file_paths.get("file_report_discovery", "")
        }

    @listen(initialize_diagnosis)
    async def analyze_values_phase(self, inputs: dict):
        print("[Fase] Análise de Valores")
        results = await self.crew_instance.crew().kickoff_async(inputs={
            "context": inputs["context"],
            "phase": "analyze_values",
            "log_path": self.crew_instance.file_paths.get("file_log_assessment", ""),
            "report_path": self.crew_instance.file_paths.get("file_report_assessment", "")
        })
        return {**inputs, "values_analysis": results}

    @listen(analyze_values_phase)
    async def historical_culture_phase(self, inputs: dict):
        print("[Fase] História e Cultura Organizacional")
        results = await self.crew_instance.crew().kickoff_async(inputs={
            "context": inputs["context"],
            "phase": "historical_culture",
            "log_path": self.crew_instance.file_paths.get("file_log_pipeline_design", ""),
            "report_path": self.crew_instance.file_paths.get("file_report_pipeline_design", "")
        })
        return {**inputs, "historical_analysis": results}

    @listen(historical_culture_phase)
    async def culture_map_phase(self, inputs: dict):
        print("[Fase] Mapa de Cultura e Alinhamento")
        results = await self.crew_instance.crew().kickoff_async(inputs={
            "context": inputs["context"],
            "phase": "culture_map",
            "log_path": self.crew_instance.file_paths.get("file_log_model_development", ""),
            "report_path": self.crew_instance.file_paths.get("file_report_model_development", "")
        })
        return {**inputs, "culture_map": results}

    @listen(culture_map_phase)
    async def recommendations_phase(self, inputs: dict):
        print("[Fase] Recomendações e Plano de Ação")
        results = await self.crew_instance.crew().kickoff_async(inputs={
            "context": inputs["context"],
            "phase": "recommendations",
            "log_path": self.crew_instance.file_paths.get("file_log_deployment", ""),
            "report_path": self.crew_instance.file_paths.get("file_report_deployment", "")
        })
        return {**inputs, "recommendations": results}

    @listen(recommendations_phase)
    async def final_report_phase(self, inputs: dict):
        print("[Fase] Síntese Final do Diagnóstico Cultural")
        results = await self.crew_instance.crew().kickoff_async(inputs={
            "context": inputs["context"],
            "phase": "final_report",
            "log_path": self.crew_instance.file_paths.get("file_log_deployment", ""),
            "report_path": self.crew_instance.file_paths.get("file_report_deployment", "")
        })
        return {**inputs, "final_report": results}

    async def run(self):
        """
        Executa todas as fases da consultoria.
        """
        step1 = self.initialize_diagnosis()
        step2 = await self.analyze_values_phase(step1)
        step3 = await self.historical_culture_phase(step2)
        step4 = await self.culture_map_phase(step3)
        step5 = await self.recommendations_phase(step4)
        step6 = await self.final_report_phase(step5)

        return step6
