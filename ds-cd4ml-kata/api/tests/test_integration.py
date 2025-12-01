"""
Integration Tests
=================
Testes de integração end-to-end.
"""

import pytest
from fastapi import status

# ============================================================================
# TESTES - End-to-End Workflows
# ============================================================================

def test_complete_prediction_workflow(client, valid_wine_sample):
    """
    Test: Workflow completo - health check → predict → model info.
    """
    # Step 1: Verificar saúde
    health_response = client.get("/health")
    assert health_response.status_code == status.HTTP_200_OK
    assert health_response.json()["status"] == "healthy"
    
    # Step 2: Fazer predição
    pred_response = client.post("/predict", json=valid_wine_sample)
    assert pred_response.status_code == status.HTTP_200_OK
    prediction = pred_response.json()
    assert prediction["prediction"] in [0, 1]
    
    # Step 3: Verificar info do modelo
    info_response = client.get("/model/info")
    assert info_response.status_code == status.HTTP_200_OK
    info = info_response.json()
    assert "model_version" in info

def test_batch_workflow(client, valid_wine_sample, valid_wine_sample_2):
    """
    Test: Workflow de batch - criar batch → predict → verificar resultados.
    """
    # Criar batch
    batch = {
        "samples": [valid_wine_sample, valid_wine_sample_2]
    }
    
    # Fazer predições
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert data["total"] == 2
    assert len(data["predictions"]) == 2
    
    # Verificar que cada predição é válida
    for pred in data["predictions"]:
        assert pred["prediction"] in [0, 1]
        assert 0 <= pred["confidence"] <= 1

def test_compare_single_vs_batch_same_result(client, valid_wine_sample):
    """
    Test: Single prediction = batch prediction (mesma amostra).
    """
    # Single prediction
    single_response = client.post("/predict", json=valid_wine_sample)
    single_pred = single_response.json()
    
    # Batch prediction com mesma amostra
    batch = {"samples": [valid_wine_sample]}
    batch_response = client.post("/predict/batch", json=batch)
    batch_pred = batch_response.json()["predictions"][0]
    
    # Resultados devem ser idênticos
    assert single_pred["prediction"] == batch_pred["prediction"]
    assert single_pred["confidence"] == batch_pred["confidence"]
    assert single_pred["probabilities"] == batch_pred["probabilities"]

# ============================================================================
# TESTES - Consistency
# ============================================================================

def test_multiple_identical_requests(client, valid_wine_sample):
    """
    Test: Múltiplas requests idênticas retornam mesmo resultado.
    """
    responses = []
    
    for _ in range(5):
        response = client.post("/predict", json=valid_wine_sample)
        assert response.status_code == status.HTTP_200_OK
        responses.append(response.json())
    
    # Todas as predições devem ser idênticas
    first_pred = responses[0]["prediction"]
    for resp in responses:
        assert resp["prediction"] == first_pred

def test_different_samples_different_results(client, valid_wine_sample, valid_wine_sample_2):
    """
    Test: Amostras diferentes podem ter predições diferentes.
    
    Nota: Não exigimos que sejam diferentes, apenas verificamos estrutura.
    """
    response1 = client.post("/predict", json=valid_wine_sample)
    response2 = client.post("/predict", json=valid_wine_sample_2)
    
    assert response1.status_code == status.HTTP_200_OK
    assert response2.status_code == status.HTTP_200_OK
    
    # Ambas devem ter estrutura válida
    pred1 = response1.json()
    pred2 = response2.json()
    
    assert pred1["prediction"] in [0, 1]
    assert pred2["prediction"] in [0, 1]

# ============================================================================
# TESTES - Error Recovery
# ============================================================================

def test_recovery_after_error(client, valid_wine_sample, invalid_wine_sample_out_of_range):
    """
    Test: API se recupera após erro (continua funcionando).
    """
    # Causar erro
    error_response = client.post("/predict", json=invalid_wine_sample_out_of_range)
    assert error_response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    # API deve continuar funcionando normalmente
    success_response = client.post("/predict", json=valid_wine_sample)
    assert success_response.status_code == status.HTTP_200_OK

def test_concurrent_valid_and_invalid_requests(client, valid_wine_sample, invalid_wine_sample_missing_field):
    """
    Test: Requests válidas e inválidas podem ser intercaladas.
    """
    # Valid
    r1 = client.post("/predict", json=valid_wine_sample)
    assert r1.status_code == status.HTTP_200_OK
    
    # Invalid
    r2 = client.post("/predict", json=invalid_wine_sample_missing_field)
    assert r2.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    # Valid again
    r3 = client.post("/predict", json=valid_wine_sample)
    assert r3.status_code == status.HTTP_200_OK
    
    # Resultados válidos devem ser idênticos
    assert r1.json()["prediction"] == r3.json()["prediction"]