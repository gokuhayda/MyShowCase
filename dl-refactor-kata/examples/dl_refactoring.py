from typing import List, Protocol, Any, Dict

# Simulação de tipos do PyTorch para não precisar da lib instalada
class Module: ...
class Optimizer: ...
class DataLoader: ...

# 1. O CONTRATO DE CALLBACK (Strategy / Observer)
class Callback(Protocol):
    def on_train_begin(self, logs: Dict[str, Any]): ...
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool: ...
    # Retorna True se deve parar o treino (Early Stopping)

# 2. O TRAINER (Closed for Modification)
class Trainer:
    """
    Motor de treino genérico.
    Não sabe como logar, não sabe quando salvar.
    Delega isso para os Callbacks.
    """
    def __init__(self, 
                 model: Any, 
                 optimizer: Any, 
                 loss_fn: Any,
                 callbacks: List[Callback] = []):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.callbacks = callbacks
        self.stop_training = False

    def fit(self, dataloader: list, epochs: int):
        # Hook: Início
        self._notify_callbacks("on_train_begin", {})
        
        for epoch in range(epochs):
            if self.stop_training: break
            
            epoch_loss = 0.0
            # Simulação do Loop de Treino
            for batch in dataloader:
                # self.optimizer.zero_grad()
                # output = self.model(batch)
                # loss = self.loss_fn(output, target)
                # loss.backward()
                # self.optimizer.step()
                epoch_loss += 0.1 # Simulação
            
            # Hook: Fim da Época
            logs = {"loss": epoch_loss, "epoch": epoch}
            should_stop = self._notify_callbacks("on_epoch_end", epoch, logs)
            if should_stop:
                self.stop_training = True

    def _notify_callbacks(self, method_name: str, *args) -> bool:
        """Dispara eventos para todos os callbacks"""
        should_stop_any = False
        for cb in self.callbacks:
            method = getattr(cb, method_name, None)
            if method:
                result = method(*args)
                if result is True: should_stop_any = True
        return should_stop_any

# 3. IMPLEMENTAÇÕES DE CALLBACK (Extensibilidade)

class EarlyStopping(Callback):
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.bad_epochs = 0
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool:
        # Lógica simplificada: se loss > 0.5, é ruim
        loss = logs.get("loss", 0)
        if loss > 0.5:
            self.bad_epochs += 1
            print(f"🛑 EarlyStopping: Bad epoch {self.bad_epochs}/{self.patience}")
        else:
            self.bad_epochs = 0
            
        return self.bad_epochs >= self.patience

class ModelCheckpoint(Callback):
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool:
        print(f"💾 Salvando modelo no final da época {epoch}...")
        return False

# ==========================================
# USO
# ==========================================
if __name__ == "__main__":
    print("--- Refatoração de Deep Learning ---")
    
    # Configuração via Composição
    trainer = Trainer(
        model="MyNet",
        optimizer="Adam",
        loss_fn="MSE",
        callbacks=[
            EarlyStopping(patience=2),
            ModelCheckpoint()
        ]
    )
    
    # Dados Fake
    fake_loader = [1, 2, 3] 
    
    # Rodar
    trainer.fit(fake_loader, epochs=5)
