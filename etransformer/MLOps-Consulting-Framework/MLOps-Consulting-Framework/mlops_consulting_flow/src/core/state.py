from pydantic import BaseModel, Field
from typing import Dict, Any, List

# ==============================================================================
# Classes de Estado para cada Fase do Projeto MLOps
# ==============================================================================

class DescobertaEstado(BaseModel):
    fase_atual: str = Field("Descoberta", description="Fase inicial de descoberta.")
    perguntas_respostas: Dict[str, Any] = Field(
        default_factory=dict,
        description="Coleta de perguntas e respostas do cliente na fase de discovery."
    )

class EntendimentoNegocioEstado(BaseModel):
    fase_atual: str = Field("Entendimento de Negócio", description="Fase de business understanding.")
    kpis: Dict[str, float] = Field(
        default_factory=dict,
        description="KPIs definidos (ex.: ROI, precisão mínima, SLA de latência)."
    )
    restricoes_de_negocio: Dict[str, str] = Field(
        default_factory=dict,
        description="Limitações de negócio, prazos, orçamentos e compliance."
    )

class DesignPipelinesEstado(BaseModel):
    fase_atual: str = Field("Design de Pipelines", description="Definição de arquitetura e pipelines.")
    arquitetura_proposta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Estrutura proposta (ex.: Airflow + Spark, etc.)."
    )
    fontes_de_dados: List[str] = Field(
        default_factory=list,
        description="Fontes de dados internas, externas, data lakes, etc."
    )

class DesenvolvimentoModelosEstado(BaseModel):
    fase_atual: str = Field("Desenvolvimento de Modelos", description="Fase de treinamento e validação de modelos.")
    parametros_modelo: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hiperparâmetros e configurações de cada modelo."
    )
    metricas_treinamento: Dict[str, float] = Field(
        default_factory=dict,
        description="Métricas coletadas durante o treinamento."
    )
    resultados_validacao: Dict[str, float] = Field(
        default_factory=dict,
        description="Métricas de validação em dataset de teste."
    )

class ImplantacaoEstado(BaseModel):
    fase_atual: str = Field("Implantação", description="Fase de deployment.")
    status_implantacao: str = Field("Não Iniciado", description="Status: Em Progresso, Staging, Produção, etc.")
    configuracoes_monitoramento: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuração de monitoramento (Prometheus, logs, etc.)."
    )
    pipelines_ci_cd: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detalhes de CI/CD."
    )

class RelatorioFinalEstado(BaseModel):
    fase_atual: str = Field("Geração de Relatório", description="Fase de consolidação de relatório final.")
    secoes_relatorio: Dict[str, Any] = Field(
        default_factory=dict,
        description="Seções do relatório final, ex.: Resumo, Riscos, Recomendações."
    )
    sumario_geral: str = Field("", description="Síntese global dos resultados alcançados.")


class EstadoMLOps(BaseModel):
    """
    Estado geral do pipeline/consultoria em MLOps, unindo todas as fases do fluxo.
    """
    descoberta: DescobertaEstado = Field(default_factory=DescobertaEstado)
    entendimento_negocio: EntendimentoNegocioEstado = Field(default_factory=EntendimentoNegocioEstado)
    design_pipelines: DesignPipelinesEstado = Field(default_factory=DesignPipelinesEstado)
    desenvolvimento_modelos: DesenvolvimentoModelosEstado = Field(default_factory=DesenvolvimentoModelosEstado)
    implantacao: ImplantacaoEstado = Field(default_factory=ImplantacaoEstado)
    relatorio_final: RelatorioFinalEstado = Field(default_factory=RelatorioFinalEstado)

    caminho_arquivos_entrada: str = Field("", description="Local dos dados de entrada.")
    caminho_relatorio_final: str = Field("", description="Local para salvamento do relatório final.")