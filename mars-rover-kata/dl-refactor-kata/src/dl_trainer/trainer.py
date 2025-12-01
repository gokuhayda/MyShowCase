"""
Motor de treino genérico seguindo o Trainer Pattern.

O Trainer executa o loop de treino mas NÃO sabe:
- Como logar métricas
- Quando parar o treino
- Como salvar checkpoints

Essas responsabilidades são delegadas para Callbacks.
"""

from typing import Any, List, Dict
from .protocols import Callback


class Trainer:
    """
    Executor genérico de loops de treino.
    
    Segue o Open/Closed Principle:
    - Aberto para extensão (adicionar callbacks)
    - Fechado para modificação (lógica do loop não muda)
    
    Examples:
        >>> trainer = Trainer(
        ...     model=MyModel(),
        ...     optimizer=Adam(),
        ...     loss_fn=MSELoss(),
        ...     callbacks=[EarlyStopping(patience=3)]
        ... )
        >>> trainer.fit(dataloader, epochs=10)
    """
    
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_fn: Any,
        callbacks: List[Callback] = None
    ):
        """
        Inicializa o Trainer.
        
        Args:
            model: Modelo a ser treinado (ex: torch.nn.Module)
            optimizer: Otimizador (ex: torch.optim.Adam)
            loss_fn: Função de perda (ex: torch.nn.MSELoss)
            callbacks: Lista de callbacks a serem notificados
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.callbacks = callbacks or []
        self.stop_training = False
    
    def fit(self, dataloader: Any, epochs: int) -> None:
        """
        Executa o loop de treino.
        
        Args:
            dataloader: Iterável com batches de treino
            epochs: Número de épocas
        """
        # Hook 1: Notificar início do treino
        initial_logs = {
            "total_epochs": epochs,
            "num_callbacks": len(self.callbacks)
        }
        self._notify_callbacks("on_train_begin", initial_logs)
        print(f"🚀 Training started with {len(self.callbacks)} callbacks")
        
        # Loop principal
        for epoch in range(epochs):
            if self.stop_training:
                print(f"⏹️  Training stopped at epoch {epoch}")
                break
            
            # Simular época de treino
            epoch_loss = self._train_one_epoch(dataloader)
            
            # Hook 2: Notificar fim da época
            logs = {
                "epoch": epoch,
                "loss": epoch_loss
            }
            print(f"Epoch {epoch} | Loss: {epoch_loss:.2f}")
            
            should_stop = self._notify_callbacks("on_epoch_end", epoch, logs)
            if should_stop:
                self.stop_training = True
        
        print("✅ Training completed")
    
    def _train_one_epoch(self, dataloader: Any) -> float:
        """
        Simula o treino de uma época.
        
        Na implementação real, seria:
```python
        total_loss = 0.0
        for batch in dataloader:
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)
```
        
        Args:
            dataloader: Batches de dados
        
        Returns:
            Loss médio da época
        """
        # Simulação: loss começa em 1.0 e diminui gradualmente
        import random
        return random.uniform(0.1, 0.5)
    
    def _notify_callbacks(
        self, 
        method_name: str, 
        *args: Any
    ) -> bool:
        """
        Notifica todos os callbacks sobre um evento.
        
        Args:
            method_name: Nome do método a chamar (ex: "on_epoch_end")
            *args: Argumentos a passar para o método
        
        Returns:
            True se qualquer callback solicitar parada do treino
        """
        should_stop_any = False
        
        for callback in self.callbacks:
            # Duck typing: verifica se o callback tem o método
            method = getattr(callback, method_name, None)
            if method and callable(method):
                result = method(*args)
                
                # Se o callback retornar True, marca para parar
                if result is True:
                    should_stop_any = True
        
        return should_stop_any
