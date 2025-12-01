"""
Inference Tests
Valida latência, formato, e robustez das predições
"""
import pytest
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.predict import WineQualityPredictor

@pytest.fixture
def predictor():
    """Inicializa preditor."""
    return WineQualityPredictor()

@pytest.fixture
def sample_wine():
    """Amostra válida de vinho."""
    return {
        'fixed_acidity': 7.4,
        'volatile_acidity': 0.7,
        'citric_acid': 0.0,
        'residual_sugar': 1.9,
        'chlorides': 0.076,
        'free_sulfur_dioxide': 11.0,
        'total_sulfur_dioxide': 34.0,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4
    }

def test_predict_returns_valid_class(predictor, sample_wine):
    """Test: Predição retorna classe válida (0 ou 1)."""
    prediction = predictor.predict(sample_wine)
    
    assert isinstance(prediction, np.ndarray)
    assert prediction[0] in [0, 1]

def test_predict_proba_sums_to_one(predictor, sample_wine):
    """Test: Probabilidades somam 1.0."""
    probas = predictor.predict_proba(sample_wine)
    
    assert probas.shape == (1, 2)
    assert np.isclose(probas.sum(), 1.0)

def test_latency_single_prediction(predictor, sample_wine):
    """Test: Latência < 100ms para single prediction."""
    max_latency_ms = 100
    
    start = time.time()
    predictor.predict(sample_wine)
    latency_ms = (time.time() - start) * 1000
    
    assert latency_ms < max_latency_ms, \
        f"Latency {latency_ms:.2f}ms exceeds {max_latency_ms}ms"

def test_latency_batch_prediction(predictor, sample_wine):
    """Test: Batch de 100 predições < 1 segundo."""
    batch_size = 100
    max_latency_s = 1.0
    
    batch = [sample_wine] * batch_size
    
    start = time.time()
    predictor.predict(batch)
    latency_s = time.time() - start
    
    assert latency_s < max_latency_s, \
        f"Batch latency {latency_s:.2f}s exceeds {max_latency_s}s"

def test_missing_features_raises_error(predictor):
    """Test: Features faltando gera erro."""
    incomplete_wine = {'fixed_acidity': 7.4}  # Só 1 feature
    
    with pytest.raises(ValueError, match="Missing features"):
        predictor.predict(incomplete_wine)

def test_deterministic_predictions(predictor, sample_wine):
    """Test: Mesma entrada = mesma saída (determinismo)."""
    pred1 = predictor.predict(sample_wine)
    pred2 = predictor.predict(sample_wine)
    
    assert np.array_equal(pred1, pred2)

def test_batch_consistency(predictor, sample_wine):
    """Test: Predição single = predição em batch."""
    # Single
    pred_single = predictor.predict(sample_wine)[0]
    
    # Batch com mesmo sample
    pred_batch = predictor.predict([sample_wine])[0]
    
    assert pred_single == pred_batch

def test_edge_case_extreme_values(predictor):
    """Test: Valores extremos (mas válidos) não quebram."""
    extreme_wine = {
        'fixed_acidity': 15.0,      # Alto
        'volatile_acidity': 0.1,    # Baixo
        'citric_acid': 1.0,         # Máximo
        'residual_sugar': 15.0,
        'chlorides': 0.01,
        'free_sulfur_dioxide': 90.0,
        'total_sulfur_dioxide': 250.0,
        'density': 1.005,
        'pH': 2.5,
        'sulphates': 2.0,
        'alcohol': 14.5
    }
    
    prediction = predictor.predict(extreme_wine)
    assert prediction[0] in [0, 1]

def test_predict_with_confidence_format(predictor, sample_wine):
    """Test: predict_with_confidence retorna formato esperado."""
    result = predictor.predict_with_confidence(sample_wine)
    
    # Verificar estrutura
    assert 'prediction' in result
    assert 'confidence' in result
    assert 'probabilities' in result
    assert 'interpretation' in result
    
    # Verificar tipos
    assert isinstance(result['prediction'], int)
    assert isinstance(result['confidence'], float)
    assert isinstance(result['probabilities'], dict)
    assert isinstance(result['interpretation'], str)
    
    # Verificar valores
    assert result['prediction'] in [0, 1]
    assert 0 <= result['confidence'] <= 1
    assert 0 in result['probabilities']
    assert 1 in result['probabilities']