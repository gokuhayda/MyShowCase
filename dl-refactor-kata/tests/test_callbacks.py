"""
Testes para callbacks concretos.
"""

import pytest
from src.dl_trainer.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    MetricsLogger,
    LearningRateScheduler
)


class TestEarlyStopping:
    """Testes para EarlyStopping callback."""
    
    def test_stops_after_patience_epochs(self):
        """Deve parar após N épocas sem melhoria."""
        callback = EarlyStopping(patience=2)
        callback.on_train_begin({})
        
        # Época 0: loss ruim
        should_stop = callback.on_epoch_end(0, {"loss": 1.0})
        assert should_stop is False
        
        # Época 1: loss ruim novamente
        should_stop = callback.on_epoch_end(1, {"loss": 1.0})
        assert should_stop is False
        
        # Época 2: loss ruim pela terceira vez
        should_stop = callback.on_epoch_end(2, {"loss": 1.0})
        assert should_stop is True  # Atingiu patience
    
    def test_resets_counter_on_improvement(self):
        """Deve resetar contador quando loss melhora."""
        callback = EarlyStopping(patience=2)
        callback.on_train_begin({})
        
        # Época 0: loss inicial
        callback.on_epoch_end(0, {"loss": 1.0})
        
        # Época 1: loss piora
        callback.on_epoch_end(1, {"loss": 1.1})
        
        # Época 2: loss melhora (reseta contador)
        callback.on_epoch_end(2, {"loss": 0.8})
        
        # Época 3: loss piora (contador em 1, não em 3)
        should_stop = callback.on_epoch_end(3, {"loss": 0.9})
        assert should_stop is False
        
        # Época 4: loss piora novamente (contador em 2)
        should_stop = callback.on_epoch_end(4, {"loss": 0.95})
        assert should_stop is True
    
    def test_considers_min_delta(self):
        """Deve considerar melhoria mínima (min_delta)."""
        callback = EarlyStopping(patience=1, min_delta=0.1)
        callback.on_train_begin({})
        
        # Época 0: loss inicial
        callback.on_epoch_end(0, {"loss": 1.0})
        
        # Época 1: melhoria pequena (< min_delta)
        should_stop = callback.on_epoch_end(1, {"loss": 0.95})
        assert should_stop is True  # Não considera melhoria


class TestModelCheckpoint:
    """Testes para ModelCheckpoint callback."""
    
    def test_saves_when_loss_improves(self):
        """Deve indicar salvamento quando loss melhora."""
        callback = ModelCheckpoint(save_best_only=True)
        callback.on_train_begin({})
        
        # Primeira época (qualquer loss é melhor que inf)
        callback.on_epoch_end(0, {"loss": 1.0})
        assert callback.best_loss == 1.0
        
        # Segunda época (melhora)
        callback.on_epoch_end(1, {"loss": 0.8})
        assert callback.best_loss == 0.8
        
        # Terceira época (não melhora)
        callback.on_epoch_end(2, {"loss": 0.9})
        assert callback.best_loss == 0.8  # Mantém o melhor
    
    def test_never_stops_training(self):
        """ModelCheckpoint nunca deve parar o treino."""
        callback = ModelCheckpoint()
        callback.on_train_begin({})
        
        should_stop = callback.on_epoch_end(0, {"loss": 1.0})
        assert should_stop is False


class TestMetricsLogger:
    """Testes para MetricsLogger callback."""
    
    def test_records_metrics_history(self):
        """Deve registrar histórico de métricas."""
        callback = MetricsLogger()
        callback.on_train_begin({})
        
        callback.on_epoch_end(0, {"loss": 1.0, "acc": 0.8})
        callback.on_epoch_end(1, {"loss": 0.8, "acc": 0.85})
        
        history = callback.get_history()
        
        assert len(history) == 2
        assert history[0]["epoch"] == 0
        assert history[0]["loss"] == 1.0
        assert history[1]["acc"] == 0.85


class TestLearningRateScheduler:
    """Testes para LearningRateScheduler callback."""
    
    def test_decays_learning_rate(self):
        """Deve decair o learning rate ao longo das épocas."""
        callback = LearningRateScheduler(
            initial_lr=0.1, 
            decay_factor=0.5
        )
        callback.on_train_begin({})
        
        assert callback.current_lr == 0.1
        
        # Após 1 época
        callback.on_epoch_end(0, {"loss": 1.0})
        assert callback.current_lr == pytest.approx(0.05)
        
        # Após 2 épocas
        callback.on_epoch_end(1, {"loss": 0.8})
        assert callback.current_lr == pytest.approx(0.025)
