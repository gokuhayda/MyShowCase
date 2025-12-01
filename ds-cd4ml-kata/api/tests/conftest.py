"""
Pytest Fixtures para API Tests
================================
Fixtures compartilhadas entre testes da API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.main import app
from api.models import WineSample

# ============================================================================
# FIXTURES - Test Client
# ============================================================================

@pytest.fixture(scope="module")
def client():
    """
    TestClient do FastAPI.
    
    Scope: module - cliente é reutilizado em todos os testes do módulo.
    """
    with TestClient(app) as test_client:
        yield test_client

# ============================================================================
# FIXTURES - Sample Data
# ============================================================================

@pytest.fixture
def valid_wine_sample():
    """Amostra válida de vinho para testes."""
    return {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }

@pytest.fixture
def valid_wine_sample_2():
    """Segunda amostra válida (para batch tests)."""
    return {
        "fixed_acidity": 8.1,
        "volatile_acidity": 0.6,
        "citric_acid": 0.2,
        "residual_sugar": 2.1,
        "chlorides": 0.08,
        "free_sulfur_dioxide": 15.0,
        "total_sulfur_dioxide": 40.0,
        "density": 0.9980,
        "pH": 3.55,
        "sulphates": 0.62,
        "alcohol": 10.5
    }

@pytest.fixture
def invalid_wine_sample_missing_field():
    """Amostra inválida - campo faltando."""
    return {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        # citric_acid FALTANDO
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }

@pytest.fixture
def invalid_wine_sample_out_of_range():
    """Amostra inválida - valor fora do range."""
    return {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 1.0,  # INVÁLIDO: pH < 2.5
        "sulphates": 0.56,
        "alcohol": 9.4
    }

@pytest.fixture
def batch_request_valid(valid_wine_sample, valid_wine_sample_2):
    """Batch request válido."""
    return {
        "samples": [valid_wine_sample, valid_wine_sample_2]
    }

@pytest.fixture
def batch_request_too_large(valid_wine_sample):
    """Batch request muito grande (>100 samples)."""
    return {
        "samples": [valid_wine_sample] * 101  # 101 samples
    }

@pytest.fixture
def batch_request_empty():
    """Batch request vazio."""
    return {
        "samples": []
    }