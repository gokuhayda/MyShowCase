"""
API Endpoints Tests
===================
Testes de todos os endpoints da API.
"""

import pytest
from fastapi import status

# ============================================================================
# TESTES - Root Endpoint
# ============================================================================

def test_root_endpoint(client):
    """
    Test: GET / retorna informações básicas da API.
    """
    response = client.get("/")
    
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "status" in data
    assert data["status"] == "healthy"

# ============================================================================
# TESTES - Health Check
# ============================================================================

def test_health_check_success(client):
    """
    Test: GET /health retorna status healthy.
    """
    response = client.get("/health")
    
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data
    assert "latency_ms" in data
    assert "timestamp" in data
    
    # Latência deve ser razoável (< 100ms)
    assert data["latency_ms"] < 100

def test_health_check_response_structure(client):
    """
    Test: Health check tem estrutura esperada.
    """
    response = client.get("/health")
    data = response.json()
    
    # Verificar todos os campos obrigatórios
    required_fields = ["status", "model_loaded", "model_version", "latency_ms", "timestamp"]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"

# ============================================================================
# TESTES - Single Prediction
# ============================================================================

def test_predict_valid_sample(client, valid_wine_sample):
    """
    Test: POST /predict com amostra válida retorna predição.
    """
    response = client.post("/predict", json=valid_wine_sample)
    
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    
    # Verificar estrutura da resposta
    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "interpretation" in data
    assert "latency_ms" in data
    assert "timestamp" in data
    
    # Verificar valores
    assert data["prediction"] in [0, 1]
    assert 0 <= data["confidence"] <= 1
    assert "0" in data["probabilities"]
    assert "1" in data["probabilities"]
    
    # Probabilidades devem somar ~1.0
    prob_sum = data["probabilities"]["0"] + data["probabilities"]["1"]
    assert 0.99 <= prob_sum <= 1.01

def test_predict_missing_field(client, invalid_wine_sample_missing_field):
    """
    Test: POST /predict com campo faltando retorna erro 422.
    """
    response = client.post("/predict", json=invalid_wine_sample_missing_field)
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    data = response.json()
    assert "detail" in data

def test_predict_out_of_range(client, invalid_wine_sample_out_of_range):
    """
    Test: POST /predict com valor fora do range retorna erro 422.
    """
    response = client.post("/predict", json=invalid_wine_sample_out_of_range)
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_predict_invalid_type(client, valid_wine_sample):
    """
    Test: POST /predict com tipo inválido retorna erro 422.
    """
    invalid_sample = valid_wine_sample.copy()
    invalid_sample["pH"] = "invalid_string"  # Deveria ser float
    
    response = client.post("/predict", json=invalid_sample)
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_predict_response_time(client, valid_wine_sample):
    """
    Test: Predição deve ser rápida (< 100ms).
    """
    response = client.post("/predict", json=valid_wine_sample)
    
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert data["latency_ms"] < 100, f"Prediction too slow: {data['latency_ms']}ms"

def test_predict_deterministic(client, valid_wine_sample):
    """
    Test: Mesma entrada = mesma saída (determinismo).
    """
    response1 = client.post("/predict", json=valid_wine_sample)
    response2 = client.post("/predict", json=valid_wine_sample)
    
    assert response1.status_code == status.HTTP_200_OK
    assert response2.status_code == status.HTTP_200_OK
    
    data1 = response1.json()
    data2 = response2.json()
    
    # Predições devem ser idênticas
    assert data1["prediction"] == data2["prediction"]
    assert data1["confidence"] == data2["confidence"]
    assert data1["probabilities"] == data2["probabilities"]

# ============================================================================
# TESTES - Batch Prediction
# ============================================================================

def test_predict_batch_valid(client, batch_request_valid):
    """
    Test: POST /predict/batch com batch válido retorna predições.
    """
    response = client.post("/predict/batch", json=batch_request_valid)
    
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    
    # Verificar estrutura
    assert "predictions" in data
    assert "total" in data
    assert "latency_ms" in data
    assert "avg_latency_ms" in data
    assert "timestamp" in data
    
    # Verificar valores
    assert data["total"] == 2
    assert len(data["predictions"]) == 2
    
    # Cada predição deve ter estrutura correta
    for pred in data["predictions"]:
        assert "prediction" in pred
        assert "confidence" in pred
        assert pred["prediction"] in [0, 1]

def test_predict_batch_too_large(client, batch_request_too_large):
    """
    Test: POST /predict/batch com >100 samples retorna erro 400.
    """
    response = client.post("/predict/batch", json=batch_request_too_large)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    data = response.json()
    assert "error" in data or "detail" in data

def test_predict_batch_empty(client, batch_request_empty):
    """
    Test: POST /predict/batch com batch vazio retorna erro 400.
    """
    response = client.post("/predict/batch", json=batch_request_empty)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_predict_batch_performance(client, valid_wine_sample):
    """
    Test: Batch de 10 amostras deve ser eficiente.
    """
    batch = {
        "samples": [valid_wine_sample] * 10
    }
    
    response = client.post("/predict/batch", json=batch)
    
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    
    # Latência média por amostra deve ser < 50ms
    assert data["avg_latency_ms"] < 50, f"Batch too slow: {data['avg_latency_ms']}ms per sample"

def test_predict_batch_consistency(client, valid_wine_sample):
    """
    Test: Batch com mesma amostra repetida deve ter predições idênticas.
    """
    batch = {
        "samples": [valid_wine_sample] * 3
    }
    
    response = client.post("/predict/batch", json=batch)
    
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    predictions = data["predictions"]
    
    # Todas as predições devem ser idênticas
    first_pred = predictions[0]["prediction"]
    for pred in predictions:
        assert pred["prediction"] == first_pred

# ============================================================================
# TESTES - Model Info
# ============================================================================

def test_model_info_success(client):
    """
    Test: GET /model/info retorna informações do modelo.
    """
    response = client.get("/model/info")
    
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    
    # Verificar estrutura
    assert "model_version" in data
    assert "model_type" in data
    assert "features" in data
    
    # Verificar valores
    assert isinstance(data["features"], list)
    assert len(data["features"]) == 11  # 11 features esperadas
    assert data["model_type"] == "RandomForestClassifier"

def test_model_info_features_complete(client):
    """
    Test: Model info contém todas as features esperadas.
    """
    response = client.get("/model/info")
    data = response.json()
    
    expected_features = [
        "fixed_acidity", "volatile_acidity", "citric_acid",
        "residual_sugar", "chlorides", "free_sulfur_dioxide",
        "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"
    ]
    
    for feature in expected_features:
        assert feature in data["features"], f"Missing feature: {feature}"

def test_model_info_has_metrics(client):
    """
    Test: Model info contém métricas (se disponíveis).
    """
    response = client.get("/model/info")
    data = response.json()
    
    # Métricas podem ser None se não disponíveis
    if data.get("metrics") is not None:
        assert isinstance(data["metrics"], dict)
        # Verificar algumas métricas esperadas
        expected_metrics = ["test_accuracy", "test_f1", "test_precision", "test_recall"]
        for metric in expected_metrics:
            if metric in data["metrics"]:
                assert 0 <= data["metrics"][metric] <= 1

# ============================================================================
# TESTES - Error Handling
# ============================================================================

def test_invalid_endpoint_404(client):
    """
    Test: Endpoint inválido retorna 404.
    """
    response = client.get("/invalid/endpoint")
    
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_invalid_method_405(client):
    """
    Test: Método HTTP inválido retorna 405.
    """
    # POST em endpoint que aceita apenas GET
    response = client.post("/health")
    
    assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

def test_predict_without_body_422(client):
    """
    Test: POST /predict sem body retorna 422.
    """
    response = client.post("/predict")
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# ============================================================================
# TESTES - CORS
# ============================================================================

def test_cors_headers_present(client, valid_wine_sample):
    """
    Test: Resposta contém headers CORS.
    """
    response = client.post("/predict", json=valid_wine_sample)
    
    assert "access-control-allow-origin" in response.headers

# ============================================================================
# TESTES - Content Type
# ============================================================================

def test_response_content_type(client):
    """
    Test: Resposta tem content-type application/json.
    """
    response = client.get("/health")
    
    assert response.headers["content-type"] == "application/json"

# ============================================================================
# TESTES - Timestamps
# ============================================================================

def test_response_has_valid_timestamp(client, valid_wine_sample):
    """
    Test: Resposta contém timestamp válido (ISO format).
    """
    response = client.post("/predict", json=valid_wine_sample)
    data = response.json()
    
    # Verificar que timestamp é uma string ISO válida
    from datetime import datetime
    try:
        datetime.fromisoformat(data["timestamp"])
    except ValueError:
        pytest.fail("Invalid timestamp format")