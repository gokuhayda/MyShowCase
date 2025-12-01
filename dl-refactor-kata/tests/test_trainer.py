"""
Testes para a classe Trainer.
"""

import pytest
from src.dl_trainer import Trainer
from src.dl_trainer.protocols import Callback


class MockCallback:
    """Callback mock para testes."""
    
    def __init__(self):
        self.train_begin_called = False
        self.epoch_end_calls = []
    
    def on_train_begin(self, logs):
        self.train_begin_called = True
    
    def on_epoch_end(self, epoch, logs):
        self.epoch_end_calls.append((epoch, logs))
        return False


class StoppingCallback:
    """Callback que para o treino imediatamente."""
    
    def on_epoch_end(self, epoch, logs):
        return True  # Para no primeiro época


def test_trainer_notifies_callbacks():
    """Trainer deve notificar callbacks sobre eventos."""
    callback = MockCallback()
    trainer = Trainer(
        model="mock",
        optimizer="mock",
        loss_fn="mock",
        callbacks=[callback]
    )
    
    trainer.fit(dataloader=[1, 2], epochs=3)
    
    # Verificar que on_train_begin foi chamado
    assert callback.train_begin_called
    
    # Verificar que on_epoch_end foi chamado 3 vezes
    assert len(callback.epoch_end_calls) == 3
    
    # Verificar estrutura dos logs
    epoch, logs = callback.epoch_end_calls[0]
    assert epoch == 0
    assert "loss" in logs


def test_trainer_stops_when_callback_requests():
    """Trainer deve parar se callback retornar True."""
    stopping_callback = StoppingCallback()
    trainer = Trainer(
        model="mock",
        optimizer="mock",
        loss_fn="mock",
        callbacks=[stopping_callback]
    )
    
    # Pedir 10 épocas mas deve parar na primeira
    trainer.fit(dataloader=[1], epochs=10)
    
    assert trainer.stop_training is True


def test_trainer_without_callbacks():
    """Trainer deve funcionar sem callbacks."""
    trainer = Trainer(
        model="mock",
        optimizer="mock",
        loss_fn="mock"
    )
    
    # Não deve lançar exceção
    trainer.fit(dataloader=[1], epochs=2)


def test_trainer_with_multiple_callbacks():
    """Trainer deve gerenciar múltiplos callbacks."""
    callback1 = MockCallback()
    callback2 = MockCallback()
    
    trainer = Trainer(
        model="mock",
        optimizer="mock",
        loss_fn="mock",
        callbacks=[callback1, callback2]
    )
    
    trainer.fit(dataloader=[1], epochs=2)
    
    # Ambos devem ser notificados
    assert callback1.train_begin_called
    assert callback2.train_begin_called
    assert len(callback1.epoch_end_calls) == 2
    assert len(callback2.epoch_end_calls) == 2
