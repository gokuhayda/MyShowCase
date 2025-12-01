"""
Validation Tests
================
Testes específicos de validação Pydantic.
"""

import pytest
from fastapi import status

# ============================================================================
# TESTES - Range Validation
# ============================================================================

@pytest.mark.parametrize("field,invalid_value", [
    ("fixed_acidity", -1.0),      # Deve ser >= 0
    ("fixed_acidity", 25.0),      # Deve ser <= 20
    ("volatile_acidity", -0.5),   # Deve ser >= 0
    ("volatile_acidity", 3.0),    # Deve ser <= 2
    ("pH", 2.0),                  # Deve ser >= 2.5
    ("pH", 5.0),                  # Deve ser <= 4.5
    ("alcohol", 5.0),             # Deve ser >= 8
    ("alcohol", 20.0),            # Deve ser <= 15
    ("density", 0.98),            # Deve ser >= 0.99
    ("density", 1.02),            # Deve ser <= 1.01
])
def test_field_range_validation(client, valid_wine_sample, field, invalid_value):
    """
    Test: Campos fora do range são rejeitados.
    
    Parametrizado para testar múltiplos campos e valores.
    """
    sample = valid_wine_sample.copy()
    sample[field] = invalid_value
    
    response = client.post("/predict", json=sample)
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# ============================================================================
# TESTES - Type Validation
# ============================================================================

@pytest.mark.parametrize("field,invalid_value", [
    ("fixed_acidity", "not_a_number"),
    ("volatile_acidity", None),
    ("pH", True),
    ("alcohol", [9.4]),
    ("density", {"value": 0.9978}),
])
def test_field_type_validation(client, valid_wine_sample, field, invalid_value):
    """
    Test: Tipos inválidos são rejeitados.
    """
    sample = valid_wine_sample.copy()
    sample[field] = invalid_value
    
    response = client.post("/predict", json=sample)
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# ============================================================================
# TESTES - Required Fields
# ============================================================================

@pytest.mark.parametrize("field_to_remove", [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
])
def test_missing_required_field(client, valid_wine_sample, field_to_remove):
    """
    Test: Todos os campos são obrigatórios.
    
    Testa cada campo individualmente.
    """
    sample = valid_wine_sample.copy()
    del sample[field_to_remove]
    
    response = client.post("/predict", json=sample)
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# ============================================================================
# TESTES - Extra Fields
# ============================================================================

def test_extra_fields_ignored(client, valid_wine_sample):
    """
    Test: Campos extras são ignorados (ou rejeitados, dependendo da config).
    """
    sample = valid_wine_sample.copy()
    sample["extra_field"] = "should_be_ignored"
    
    response = client.post("/predict", json=sample)
    
    # Pode ser 200 (ignorado) ou 422 (rejeitado)
    # Depende da config do Pydantic
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]

# ============================================================================
# TESTES - Edge Cases
# ============================================================================

def test_boundary_values_min(client, valid_wine_sample):
    """
    Test: Valores mínimos dos ranges são aceitos.
    """
    sample = valid_wine_sample.copy()
    sample["pH"] = 2.5  # Mínimo permitido
    sample["alcohol"] = 8.0  # Mínimo permitido
    sample["density"] = 0.99  # Mínimo permitido
    
    response = client.post("/predict", json=sample)
    
    assert response.status_code == status.HTTP_200_OK

def test_boundary_values_max(client, valid_wine_sample):
    """
    Test: Valores máximos dos ranges são aceitos.
    """
    sample = valid_wine_sample.copy()
    sample["pH"] = 4.5  # Máximo permitido
    sample["alcohol"] = 15.0  # Máximo permitido
    sample["density"] = 1.01  # Máximo permitido
    
    response = client.post("/predict", json=sample)
    
    assert response.status_code == status.HTTP_200_OK

def test_zero_values(client, valid_wine_sample):
    """
    Test: Valores zero são aceitos onde permitido.
    """
    sample = valid_wine_sample.copy()
    sample["citric_acid"] = 0.0
    sample["residual_sugar"] = 0.0
    
    response = client.post("/predict", json=sample)
    
    assert response.status_code == status.HTTP_200_OK

# ============================================================================
# TESTES - Batch Validation
# ============================================================================

def test_batch_max_size(client, valid_wine_sample):
    """
    Test: Batch de 100 samples (máximo) é aceito.
    """
    batch = {
        "samples": [valid_wine_sample] * 100
    }
    
    response = client.post("/predict/batch", json=batch)
    
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert data["total"] == 100

def test_batch_size_101_rejected(client, valid_wine_sample):
    """
    Test: Batch de 101 samples é rejeitado.
    """
    batch = {
        "samples": [valid_wine_sample] * 101
    }
    
    response = client.post("/predict/batch", json=batch)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_batch_with_one_invalid_sample(client, valid_wine_sample, invalid_wine_sample_out_of_range):
    """
    Test: Batch com uma amostra inválida é rejeitado.
    """
    batch = {
        "samples": [valid_wine_sample, invalid_wine_sample_out_of_range]
    }
    
    response = client.post("/predict/batch", json=batch)
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY