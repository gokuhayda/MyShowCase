"""
Módulo responsável por registrar e retornar as ferramentas disponíveis para os agentes da CrewAI.
Inclui ferramentas de proteção semântica, busca em arquivos e defesa contra injeção de prompt.
"""

def log_tool_call(tool_func):
    """
    Decorador que registra a chamada de uma ferramenta nos logs para rastreamento.

    Parâmetros:
        tool_func (callable): Função ou instância da ferramenta.

    Retorna:
        callable: Função decorada com logging.
    """
    def wrapper(*args, **kwargs):
        from utils_tools.model_logger import logger
        logger.info(f"[TOOL USADA] {tool_func.__class__.__name__} chamada com args: {args}, kwargs: {kwargs}")
        return tool_func(*args, **kwargs)
    return wrapper

def load_tools(composable_graph=None):
    """
    Carrega as ferramentas disponíveis para os agentes CrewAI, incluindo:
    - Ferramentas de busca em arquivos
    - Validação semântica
    - Defesa contra prompt injection

    Parâmetros:
        composable_graph (ComposableGraph): Grafo para ferramentas que exigem contexto vetorial.

    Retorna:
        dict: Dicionário com as ferramentas mapeadas por chave.
    """
    from crewai_tools import tools
    from tools.anti_prompt_injection_tool import AntiPromptInjectionTool
#    from tools.semantic_guard_tool import SemanticGuardTool
    from tools.segmented_semantic_guard_tool import SegmentedSemanticGuardTool

#    semantic_guard = log_tool_call(SemanticGuardTool(composable_graph))
    semantic_segmented_guard = log_tool_call(SegmentedSemanticGuardTool(composable_graph))
    anti_injection = log_tool_call(AntiPromptInjectionTool())

    available_tools_action = {
        "csv_search_tool": tools.CSVSearchTool(),
        "directory_read_tool": tools.DirectoryReadTool(result_as_answer=True),
        "docx_search_tool": tools.DOCXSearchTool(),
        "txt_search_tool": tools.TXTSearchTool(),
        "semantic_search_tool": tools.MDXSearchTool(),
        # "semantic_guard_tool": semantic_guard,
        # "semantic_segmented_guard_tool": semantic_segmented_guard,
        # "anti_prompt_injection": anti_injection,
    }

    return available_tools_action
