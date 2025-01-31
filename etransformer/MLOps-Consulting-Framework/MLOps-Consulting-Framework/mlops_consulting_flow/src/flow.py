import yaml
import logging
import asyncio
from crewai.flow.flow import Flow, listen, start
from crews.poem_crew.poem_crew import MLOpsCrew
from core.utils import load_project_description

class MLOpsConsultingFlow(Flow):
    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Instancia a Crew de MLOps
        self.mlops_crew = MLOpsCrew(config=self.config)

    @start()
    def initialize_discovery(self):
        """Inicializa a fase de descoberta."""
        md_content, parsed_input = load_project_description(self.mlops_crew.project_description_path)
        chunked_description = self.mlops_crew.chunk_text_semantic(md_content)

        file_log_path = self.mlops_crew.log_discovery_path
        print(f"DEBUG - Discovery Log Path: {file_log_path}")

        return {
            "project_type": parsed_input.get("project_type", ""),
            "industry": parsed_input.get("industry", ""),
            "objectives": parsed_input.get("objectives", ""),
            "team_members": parsed_input.get("team_members", ""),
            "project_requirements": parsed_input.get("requirements", ""),
            "project_description_chunks": chunked_description,
        }

    @listen(initialize_discovery)
    async def discovery_phase(self, inputs: dict):
        """Executa a fase de descoberta do projeto."""
        if "file_log_path" not in inputs:
            inputs["file_log_path"] = self.mlops_crew.log_discovery_path
            inputs["file_path_report"] = self.mlops_crew.report_discovery_path
            inputs["file_path_sources"] = self.mlops_crew.sources_path_general

        print(f"DEBUG - Running Discovery Phase with Log Path: {inputs['file_log_path']}")

        results = await asyncio.gather(*[
            self.mlops_crew.crew().kickoff_async(inputs={"project_description": chunk, "file_log_path": inputs["file_log_path"],
                                                                                       "file_path_report": inputs["file_path_report"],
                                                                                       "file_path_sources": inputs["file_path_sources"],
                                                                                       "project_requirements": inputs["project_requirements"],
                                                                                       "project_type": inputs["project_type"],
                                                                                       "industry": inputs["industry"],
                                                                                       "team_members": inputs["team_members"]})
            for chunk in inputs["project_description_chunks"]
        ])

        chunked_results = self.mlops_crew.chunk_text_semantic(str(results))

        return {
            **inputs,
            "discovery_results": chunked_results,
        }

    @listen(discovery_phase)
    async def assessment_phase(self, discovery_results: dict):
        """Executa a fase de avaliação do projeto."""
        if "file_log_path" not in discovery_results:
            discovery_results["file_log_path"] = self.mlops_crew.log_assessment_path
            discovery_results["file_path_report"] = self.mlops_crew.report_assessment_path
            discovery_results["file_path_sources"] = self.mlops_crew.sources_path_general, 

        print(f"DEBUG - Running Assessment Phase with Log Path: {discovery_results['file_log_path']}")

        chunked_results = self.mlops_crew.chunk_text_semantic(str(discovery_results))
        results = await asyncio.gather(*[
            self.mlops_crew.crew().kickoff_async(inputs={"assessment_input": chunk, "file_log_path": discovery_results["file_log_path"],
                                                                                    "file_path_report": discovery_results["file_path_report"],
                                                                                    "file_path_sources": discovery_results["file_path_sources"],
                                                                                    "project_requirements": discovery_results["project_requirements"],
                                                                                    "project_type": discovery_results["project_type"],
                                                                                    "industry": discovery_results["industry"],
                                                                                    "team_members": discovery_results["team_members"]})
            for chunk in chunked_results
        ])

        return {
            **discovery_results,
            "assessment_results": results,
        }

    @listen(assessment_phase)
    async def pipeline_design_phase(self, assessment_results: dict):
        """Executa a fase de design do pipeline."""
        if "file_log_path" not in assessment_results:
            assessment_results["file_log_path"] = self.mlops_crew.log_pipeline_design_path
            assessment_results["file_path_report"] = self.mlops_crew.report_pipeline_design_path
            assessment_results["file_path_sources"] = self.mlops_crew.sources_path_general

        print(f"DEBUG - Running Pipeline Design Phase with Log Path: {assessment_results['file_log_path']}")

        chunked_results = self.mlops_crew.chunk_text_semantic(str(assessment_results))
        results = await asyncio.gather(*[
            self.mlops_crew.crew().kickoff_async(inputs={"pipeline_input": chunk, "file_log_path": assessment_results["file_log_path"],
                                                                                "file_path_report": assessment_results["file_path_report"],
                                                                                "file_path_sources": assessment_results["file_path_sources"],
                                                                                "project_requirements": assessment_results["project_requirements"],
                                                                                "project_type": assessment_results["project_type"],
                                                                                "industry": assessment_results["industry"],
                                                                                "team_members": assessment_results["team_members"]})
            for chunk in chunked_results
        ])

        return {
            **assessment_results,
            "pipeline_results": results,
        }

    @listen(pipeline_design_phase)
    async def model_development_phase(self, pipeline_results: dict):
        """Executa a fase de desenvolvimento do modelo."""
        if "file_log_path" not in pipeline_results:
            pipeline_results["file_log_path"] = self.mlops_crew.log_model_path
            pipeline_results["file_path_report"] = self.mlops_crew.log_model_path
            pipeline_results["file_path_sources"] = self.mlops_crew.sources_path_general

        print(f"DEBUG - Running Model Development Phase with Log Path: {pipeline_results['file_log_path']}")

        chunked_results = self.mlops_crew.chunk_text_semantic(str(pipeline_results))
        results = await asyncio.gather(*[
            self.mlops_crew.crew().kickoff_async(inputs={"model_input": chunk, "file_log_path": pipeline_results["file_log_path"],
                                                                                "file_path_report": pipeline_results["file_path_report"],
                                                                                "file_path_sources": pipeline_results["file_path_sources"],
                                                                                "project_requirements": pipeline_results["project_requirements"],
                                                                                "project_type": pipeline_results["project_type"],
                                                                                "industry": pipeline_results["industry"],
                                                                                "team_members": pipeline_results["team_members"]})
            for chunk in chunked_results
        ])

        return {
            **pipeline_results,
            "model_results": results,
        }

    @listen(model_development_phase)
    async def deployment_phase(self, model_results: dict):
        """Executa a fase de implantação do modelo."""
        if "file_log_path" not in model_results:
            model_results["file_log_path"] = self.mlops_crew.log_deployment_path
            model_results["file_path_report"] = self.mlops_crew.log_deployment_path
            model_results["file_path_sources"] = self.mlops_crew.log_deployment_path
                       
        print(f"DEBUG - Running Deployment Phase with Log Path: {model_results['file_log_path']}")

        chunked_results = self.mlops_crew.chunk_text_semantic(str(model_results))
        results = await asyncio.gather(*[
            self.mlops_crew.crew().kickoff_async(inputs={"deployment_input": chunk, "file_log_path": model_results["file_log_path"],
                                                                                    "file_path_report": model_results["file_path_report"],
                                                                                    "file_path_sources": model_results["file_path_sources"],
                                                                                    "project_requirements": model_results["project_requirements"],
                                                                                    "project_type": model_results["project_type"],
                                                                                    "industry": model_results["industry"],
                                                                                    "team_members": model_results["team_members"]})
            for chunk in chunked_results
        ])

        return {
            **model_results,
            "deployment_results": results,
        }

    @listen(deployment_phase)
    async def final_report_phase(self, deployment_results: dict):
        """Executa a fase final de geração de relatório."""
        if "file_log_path" not in deployment_results:
            deployment_results["file_log_path"] = self.mlops_crew.log_final_flux_path
            deployment_results["file_log_file_path_reportpath"] = self.mlops_crew.log_final_flux_path
            deployment_results["file_path_sources"] = self.mlops_crew.log_final_flux_path

        print(f"DEBUG - Running Final Report Phase with Log Path: {deployment_results['file_log_path']}")

        chunked_results = self.mlops_crew.chunk_text_semantic(str(deployment_results))
        results = await asyncio.gather(*[
            self.mlops_crew.crew().kickoff_async(inputs={"report_input": chunk, "file_log_path": deployment_results["file_log_path"],
                                                                                "file_path_report": deployment_results["file_path_report"],
                                                                                "file_path_sources": deployment_results["file_path_sources"],
                                                                                "project_requirements": deployment_results["project_requirements"],
                                                                                "project_type": deployment_results["project_type"],
                                                                                "industry": deployment_results["industry"],
                                                                                "team_members": deployment_results["team_members"]})
        ])

        return {
            **deployment_results,
            "final_report_results": results,
        }

    async def run(self):
        """Executa todo o fluxo de consultoria de MLOps."""

        discovery_inputs = self.initialize_discovery()
        discovery_results = await self.discovery_phase(discovery_inputs)

        assessment_results = await self.assessment_phase(discovery_results)
        pipeline_results = await self.pipeline_design_phase(assessment_results)
        model_results = await self.model_development_phase(pipeline_results)
        deployment_results = await self.deployment_phase(model_results)
        final_report_results = await self.final_report_phase(deployment_results)

        print("Fluxo concluído com sucesso!")
        return final_report_results
