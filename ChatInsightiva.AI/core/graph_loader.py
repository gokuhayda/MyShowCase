"""
Módulo responsável por fornecer um único ComposableGraph compartilhado entre componentes do sistema.
"""

import logging
from build_category_indexes import create_unified_graph
from llama_index.core.composability import ComposableGraph

logger = logging.getLogger(__name__)
_shared_graph = None

def get_composable_graph() -> ComposableGraph:
    """
    Retorna uma instância única (singleton) de ComposableGraph,
    criando-a caso ainda não tenha sido inicializada.

    Retorna:
        ComposableGraph: Instância unificada do grafo de conhecimento.
    """
    global _shared_graph
    if _shared_graph is None:
        logger.debug("🧠 Criando ComposableGraph (único)...")
        _shared_graph = create_unified_graph()
    return _shared_graph
