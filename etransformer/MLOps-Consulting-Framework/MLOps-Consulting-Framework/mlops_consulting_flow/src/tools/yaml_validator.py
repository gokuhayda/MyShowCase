import yaml
from jsonschema import validate, ValidationError, SchemaError
from typing import Any, Dict
from crewai.tools.base_tool import BaseTool
from pydantic import Field

class YamlValidatorTool(BaseTool):
    """
    Ferramenta para validar arquivos YAML contra um esquema JSON predefinido.
    Pode ser chamada por um agente para verificar a conformidade de um arquivo YAML.
    """
    name: str = "YAML Validator"
    description: str = "Valida um arquivo YAML fornecido contra um esquema JSON específico."
    
    schema: Dict[str, Any] = Field(default_factory=dict, description="Esquema JSON para validação do YAML")

    def _run(self, yaml_content: str) -> Dict[str, Any]:
        """
        Executa a validação do YAML.

        :param yaml_content: Conteúdo do arquivo YAML como string.
        :return: Dicionário com o resultado da validação.
        """
        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as exc:
            return {
                "success": False,
                "error": f"Erro ao analisar o YAML: {exc}"
            }
        
        try:
            validate(instance=data, schema=self.schema)
            return {
                "success": True,
                "message": "Validação bem-sucedida: O arquivo YAML está em conformidade com o esquema."
            }
        except ValidationError as ve:
            return {
                "success": False,
                "error": f"Erro de validação:\nMensagem: {ve.message}\nLocalização: {' -> '.join([str(elem) for elem in ve.path])}"
            }
        except SchemaError as se:
            return {
                "success": False,
                "error": f"Erro no esquema JSON:\nMensagem: {se.message}"
            }

    async def _arun(self, yaml_content: str) -> Dict[str, Any]:
        """
        Executa a validação do YAML de forma assíncrona.

        :param yaml_content: Conteúdo do arquivo YAML como string.
        :return: Dicionário com o resultado da validação.
        """
        return self._run(yaml_content)
