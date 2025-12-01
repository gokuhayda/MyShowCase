"""
Model Performance Tests
Valida se métricas atendem quality gates
"""
import pytest
import json
import yaml
from pathlib import Path

@pytest.fixture
def metrics():
    """Carrega métricas do último treino."""
    metrics_path = "models/metrics.json"
    
    if not Path(metrics_path).exists():
        pytest.skip(f"Metrics file not found: {metrics_path}")
    
    with open(metrics_path, 'r') as f:
        return json.load(f)

@pytest.fixture
def quality_gates():
    """Carrega quality gates do params.yaml."""
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
    return params['metrics']

def test_accuracy_threshold(metrics, quality_gates):
    """Test: Acurácia no teste >= threshold."""
    accuracy = metrics['test_accuracy']
    threshold = quality_gates['min_accuracy']
    
    assert accuracy >= threshold, \
        f"Accuracy {accuracy:.4f} below threshold {threshold:.4f}"

def test_precision_threshold(metrics, quality_gates):
    """Test: Precisão no teste >= threshold."""
    precision = metrics['test_precision']
    threshold = quality_gates['min_precision']
    
    assert precision >= threshold, \
        f"Precision {precision:.4f} below threshold {threshold:.4f}"

def test_recall_threshold(metrics, quality_gates):
    """Test: Recall no teste >= threshold."""
    recall = metrics['test_recall']
    threshold = quality_gates['min_recall']
    
    assert recall >= threshold, \
        f"Recall {recall:.4f} below threshold {threshold:.4f}"

def test_f1_threshold(metrics, quality_gates):
    """Test: F1-score no teste >= threshold."""
    f1 = metrics['test_f1']
    threshold = quality_gates['min_f1']
    
    assert f1 >= threshold, \
        f"F1 {f1:.4f} below threshold {threshold:.4f}"

def test_overfitting_gap(metrics, quality_gates):
    """Test: Gap entre treino e teste <= threshold."""
    gap = metrics['accuracy_gap']
    max_gap = quality_gates['max_train_test_gap']
    
    assert gap <= max_gap, \
        f"Overfitting gap {gap:.4f} exceeds maximum {max_gap:.4f}"

def test_cross_validation_stability(metrics):
    """Test: Cross-validation std <= 0.05 (modelo estável)."""
    cv_std = metrics.get('cv_accuracy_std', 0)
    max_std = 0.05
    
    assert cv_std <= max_std, \
        f"CV std {cv_std:.4f} too high (max: {max_std:.4f}). Model unstable."

def test_auc_reasonable(metrics):
    """Test: AUC >= 0.70 (modelo discrimina classes)."""
    auc = metrics.get('test_auc', 0)
    min_auc = 0.70
    
    assert auc >= min_auc, \
        f"AUC {auc:.4f} too low (min: {min_auc:.4f}). Model can't discriminate classes."

def test_metrics_keys_exist(metrics):
    """Test: Todas métricas essenciais estão presentes."""
    required_keys = [
        'test_accuracy', 'test_precision', 'test_recall', 'test_f1',
        'train_accuracy', 'accuracy_gap', 'cv_accuracy_mean'
    ]
    
    missing = [key for key in required_keys if key not in metrics]
    
    assert not missing, f"Missing metrics: {missing}"