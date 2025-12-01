"""
Protocol (Interface) para Callbacks.

Protocols em Python são como interfaces em outras linguagens,
mas usando duck typing. Qualquer classe que implementar os
métodos especificados pode ser usada como Callback.
"""

from typing import Protocol, Dict, Any


class Callback(Protocol):
    """
    Interface que define o contrato de um Callback.
    
    Callbacks são notificados em momentos específicos do treino
    e podem influenciar o fluxo (ex: early stopping).
    """
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        """
        Chamado uma vez no início do treino.
        
        Args:
            logs: Dicionário com informações iniciais
                  (ex: total de épocas, tamanho do dataset)
        """
        ...
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool:
        """
        Chamado ao final de cada época.
        
        Args:
            epoch: Número da época (0-indexed)
            logs: Métricas da época (ex: loss, accuracy)
        
        Returns:
            True se o treino deve ser interrompido, False caso contrário
        """
        ...
