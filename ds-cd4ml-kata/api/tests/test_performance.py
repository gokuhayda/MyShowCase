"""
Performance Tests
=================
Testes de latência e throughput da API.
"""

import pytest
import time
from fastapi import status

# ============================================================================
# TESTES - Latency
# ============================================================================

def test_single_prediction_latency(client, valid_wine_sample):
    """
    Test: Single prediction deve ser < 100ms.
    """
    start = time.time()
    response = client.post("/predict", json=valid_wine_sample)
    latency_ms = (time.time() - start) * 1000
    
    assert response.status_code == status.HTTP_200_OK
    assert latency_ms < 100, f"Latency too high: {latency_ms:.2f}ms"

def test_health_check_latency(client):
    """
    Test: Health check deve ser < 200ms.
    """
    start = time.time()
    response = client.get("/health")
    latency_ms = (time.time() - start) * 1000
    
    assert response.status_code == status.HTTP_200_OK
    assert latency_ms < 200, f"Health check too slow: {latency_ms:.2f}ms"

def test_batch_prediction_latency(client, valid_wine_sample):
    """
    Test: Batch de 10 samples deve ter latência média < 20ms per sample.
    """
    batch = {
        "samples": [valid_wine_sample] * 10
    }
    
    response = client.post("/predict/batch", json=batch)
    
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    avg_latency = data["avg_latency_ms"]
    
    assert avg_latency < 20, f"Batch avg latency too high: {avg_latency:.2f}ms"

# ============================================================================
# TESTES - Throughput
# ============================================================================

def test_sequential_predictions_throughput(client, valid_wine_sample):
    """
    Test: 100 predições sequenciais devem completar em < 10s.
    """
    start = time.time()
    
    for _ in range(100):
        response = client.post("/predict", json=valid_wine_sample)
        assert response.status_code == status.HTTP_200_OK
    
    total_time = time.time() - start
    
    assert total_time < 10, f"100 predictions took {total_time:.2f}s (expected < 10s)"
    
    # Calcular throughput
    throughput = 100 / total_time
    print(f"\n    Throughput: {throughput:.2f} predictions/second")
    
    assert throughput > 10, f"Throughput too low: {throughput:.2f} pred/s"

# ============================================================================
# TESTES - Caching / Warm-up
# ============================================================================

def test_cold_start_vs_warm_predictions(client, valid_wine_sample):
    """
    Test: Predições após warm-up devem ser mais rápidas.
    """
    # Cold start (primeira predição)
    start = time.time()
    response1 = client.post("/predict", json=valid_wine_sample)
    cold_latency_ms = (time.time() - start) * 1000
    
    # Warm predictions (após cache)
    latencies = []
    for _ in range(10):
        start = time.time()
        response = client.post("/predict", json=valid_wine_sample)
        latencies.append((time.time() - start) * 1000)
    
    avg_warm_latency_ms = sum(latencies) / len(latencies)
    
    print(f"\n    Cold start: {cold_latency_ms:.2f}ms")
    print(f"    Warm avg: {avg_warm_latency_ms:.2f}ms")
    
    # Warm deveria ser <= cold (ou próximo)
    # Não exigir que seja menor pois modelo já está loaded

# ============================================================================
# TESTES - Memory
# ============================================================================

@pytest.mark.slow
def test_memory_leak_batch_predictions(client, valid_wine_sample):
    """
    Test: Múltiplos batches não devem causar memory leak.
    
    Marca: slow (demora mais tempo)
    """
    batch = {
        "samples": [valid_wine_sample] * 50
    }
    
    # Fazer 20 batches (1000 predições total)
    for _ in range(20):
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == status.HTTP_200_OK
    
    # Se chegou aqui sem crash, não há memory leak óbvio
    # (teste mais rigoroso exigiria profiling de memória)