"""
Implementações concretas de Callbacks.

Cada callback encapsula UMA responsabilidade específica.
"""

from typing import Dict, Any


class EarlyStopping:
    """
    Para o treino se a métrica não melhorar por N épocas.
    
    Strategy Pattern: Encapsula a lógica de "quando parar".
    """
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        """
        Args:
            patience: Número de épocas para esperar melhoria
            min_delta: Melhoria mínima considerada significativa
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.bad_epochs = 0
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        """Reseta contadores no início do treino."""
        self.best_loss = float('inf')
        self.bad_epochs = 0
        print(f"📊 Early stopping: patience={self.patience}")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool:
        """
        Verifica se houve melhoria na loss.
        
        Returns:
            True se deve parar o treino
        """
        current_loss = logs.get("loss", float('inf'))
        
        # Verifica se houve melhoria significativa
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.bad_epochs = 0
            print(f"✨ New best loss: {self.best_loss:.4f}")
        else:
            self.bad_epochs += 1
            print(
                f"🛑 Early stopping: "
                f"{self.bad_epochs}/{self.patience} bad epochs"
            )
        
        # Para se atingiu o limite de paciência
        if self.bad_epochs >= self.patience:
            print(
                f"⏹️  Early stopping triggered at epoch {epoch} "
                f"(patience: {self.patience})"
            )
            return True
        
        return False


class ModelCheckpoint:
    """
    Salva o modelo periodicamente ou quando melhora.
    
    Separa a lógica de I/O da lógica matemática.
    """
    
    def __init__(
        self, 
        filepath: str = "model_checkpoint.pth",
        save_best_only: bool = True
    ):
        """
        Args:
            filepath: Caminho para salvar o modelo
            save_best_only: Se True, só salva quando melhora
        """
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.best_loss = float('inf')
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        """Inicializa o melhor loss."""
        self.best_loss = float('inf')
        print(f"💾 Checkpoint: saving to {self.filepath}")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool:
        """
        Salva o modelo se for o melhor até agora.
        
        Na implementação real:
```python
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': current_loss,
        }, self.filepath)
```
        """
        current_loss = logs.get("loss", float('inf'))
        
        should_save = (
            not self.save_best_only or 
            current_loss < self.best_loss
        )
        
        if should_save:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                print(
                    f"💾 Checkpoint: Saving best model "
                    f"(loss: {current_loss:.4f})"
                )
            else:
                print(f"💾 Checkpoint: Saving model at epoch {epoch}")
            
            # Simulação de salvamento
            # torch.save(model.state_dict(), self.filepath)
        
        return False


class MetricsLogger:
    """
    Loga métricas em formato estruturado.
    
    Pode ser extendido para enviar para W&B, MLflow, TensorBoard.
    """
    
    def __init__(self, log_every_n_epochs: int = 1):
        """
        Args:
            log_every_n_epochs: Frequência de logging
        """
        self.log_every_n_epochs = log_every_n_epochs
        self.history = []
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        """Inicializa histórico."""
        self.history = []
        print("📈 Metrics logger initialized")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool:
        """Registra métricas."""
        self.history.append({"epoch": epoch, **logs})
        
        if epoch % self.log_every_n_epochs == 0:
            print(f"📊 Metrics: {logs}")
        
        return False
    
    def get_history(self) -> list:
        """Retorna histórico de métricas."""
        return self.history


class LearningRateScheduler:
    """
    Ajusta learning rate ao longo do treino.
    
    Strategy Pattern: Encapsula política de scheduling.
    """
    
    def __init__(self, initial_lr: float = 0.001, decay_factor: float = 0.9):
        """
        Args:
            initial_lr: Learning rate inicial
            decay_factor: Fator de decaimento por época
        """
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.current_lr = initial_lr
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        """Reseta learning rate."""
        self.current_lr = self.initial_lr
        print(f"📉 LR Scheduler: initial_lr={self.initial_lr}")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool:
        """Decai o learning rate."""
        self.current_lr *= self.decay_factor
        print(f"📉 LR: {self.current_lr:.6f}")
        
        # Na implementação real:
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = self.current_lr
        
        return False
