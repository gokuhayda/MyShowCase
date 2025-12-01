"""
Exemplo básico de uso do Trainer com callbacks.
"""

from src.dl_trainer import Trainer, EarlyStopping, ModelCheckpoint

def main():
    print("=" * 50)
    print("Exemplo Básico: Trainer com Callbacks")
    print("=" * 50)
    
    # Configurar Trainer com composição de callbacks
    trainer = Trainer(
        model="SimulatedResNet",
        optimizer="Adam(lr=0.001)",
        loss_fn="CrossEntropyLoss",
        callbacks=[
            EarlyStopping(patience=3),
            ModelCheckpoint(filepath="best_model.pth")
        ]
    )
    
    # Dados simulados (na prática seria DataLoader do PyTorch)
    fake_dataloader = [1, 2, 3]
    
    # Executar treino
    trainer.fit(fake_dataloader, epochs=10)
    
    print("\n" + "=" * 50)
    print("Treino concluído!")
    print("=" * 50)


if __name__ == "__main__":
    main()
