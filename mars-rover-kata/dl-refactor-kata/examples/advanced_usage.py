"""
Exemplo avançado: Múltiplos callbacks personalizados.
"""

from src.dl_trainer import (
    Trainer,
    EarlyStopping,
    ModelCheckpoint,
    MetricsLogger,
    LearningRateScheduler
)


class CustomSlackNotifier:
    """
    Exemplo de callback personalizado: notificação no Slack.
    
    Demonstra como é fácil estender sem modificar Trainer.
    """
    
    def on_train_begin(self, logs):
        print("📢 [Slack] Treino iniciado!")
        # Na prática: slack_client.post_message(...)
    
    def on_epoch_end(self, epoch, logs):
        if epoch % 5 == 0:  # Notificar a cada 5 épocas
            loss = logs.get("loss", "N/A")
            print(f"📢 [Slack] Checkpoint: Época {epoch}, Loss: {loss}")
        return False


def main():
    print("=" * 60)
    print("Exemplo Avançado: Múltiplos Callbacks Personalizados")
    print("=" * 60)
    
    # Composição rica de callbacks
    trainer = Trainer(
        model="AdvancedCNN",
        optimizer="Adam",
        loss_fn="FocalLoss",
        callbacks=[
            EarlyStopping(patience=5, min_delta=0.01),
            ModelCheckpoint(filepath="checkpoints/model.pth"),
            MetricsLogger(log_every_n_epochs=2),
            LearningRateScheduler(initial_lr=0.01, decay_factor=0.95),
            CustomSlackNotifier()
        ]
    )
    
    # Simular treino com mais épocas
    fake_dataloader = list(range(10))
    trainer.fit(fake_dataloader, epochs=20)
    
    print("\n" + "=" * 60)
    print("✅ Exemplo avançado concluído!")
    print("=" * 60)
    
    # Mostrar que callbacks podem ser reutilizados
    print("\n--- Novo treino com mesmos callbacks ---")
    trainer2 = Trainer(
        model="AnotherModel",
        optimizer="SGD",
        loss_fn="MSE",
        callbacks=trainer.callbacks  # Reuso!
    )
    trainer2.fit(fake_dataloader, epochs=5)


if __name__ == "__main__":
    main()
