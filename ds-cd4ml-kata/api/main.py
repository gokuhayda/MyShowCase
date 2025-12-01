"""
FastAPI - Wine Quality Prediction API
======================================
API REST para servir modelo de classificação de qualidade de vinho.

Endpoints:
- GET  /              - Health check e info da API
- GET  /health        - Health check detalhado
- POST /predict       - Predição single
- POST /predict/batch - Predição em lote
- GET  /model/info    - Informações do modelo

Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import (
    WineSample,
    WineBatchRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from api.predictor import PredictorService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI App Configuration
# ============================================================================

app = FastAPI(
    title="Wine Quality Prediction API",
    description="""
    🍷 API REST para classificação de qualidade de vinho.
    
    ## Features
    * **Single Prediction**: Predição para uma amostra
    * **Batch Prediction**: Predição para múltiplas amostras
    * **Model Info**: Informações sobre o modelo em produção
    * **Health Check**: Status da API e do modelo
    
    ## ML Pipeline
    - Model: RandomForest Classifier
    - Features: 11 physicochemical properties
    - Target: Binary classification (good wine vs bad wine)
    - Accuracy: ~86%
    
    ## Usage
```python
    import requests
    
    # Single prediction
    sample = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        # ... outras features
    }
    response = requests.post("http://localhost:8000/predict", json=sample)
    print(response.json())
```
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health", "description": "Health checks e status"},
        {"name": "Prediction", "description": "Endpoints de predição"},
        {"name": "Model", "description": "Informações do modelo"}
    ]
)

# ============================================================================
# CORS Middleware
# ============================================================================
# Permitir requisições de qualquer origem (ajustar em produção!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção: especificar domínios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State - Predictor Service
# ============================================================================
predictor_service = None

@app.on_event("startup")
async def startup_event():
    """Inicializa o serviço de predição na startup da API."""
    global predictor_service
    logger.info("🚀 Starting API...")
    
    try:
        predictor_service = PredictorService()
        logger.info("✅ Predictor service initialized successfully")
        logger.info(f"   Model loaded: {predictor_service.model_path}")
        logger.info(f"   Startup time: {datetime.now().isoformat()}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize predictor: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup na shutdown da API."""
    logger.info("🛑 Shutting down API...")
    # Cleanup se necessário (fechar conexões, etc)

# ============================================================================
# ENDPOINTS - Health & Info
# ============================================================================

@app.get(
    "/",
    response_model=dict,
    tags=["Health"],
    summary="Root endpoint"
)
async def root():
    """
    Root endpoint - informações básicas da API.
    """
    return {
        "message": "🍷 Wine Quality Prediction API",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs",
        "health": "/health"
    }

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check detalhado"
)
async def health_check():
    """
    Health check detalhado.
    
    Retorna:
    - Status da API
    - Status do modelo
    - Métricas de performance
    - Uptime
    """
    try:
        # Verificar se modelo está carregado
        is_model_loaded = predictor_service is not None and predictor_service.predictor is not None
        
        if not is_model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        # Teste rápido de predição (smoke test)
        test_sample = WineSample(
            fixed_acidity=7.4,
            volatile_acidity=0.7,
            citric_acid=0.0,
            residual_sugar=1.9,
            chlorides=0.076,
            free_sulfur_dioxide=11.0,
            total_sulfur_dioxide=34.0,
            density=0.9978,
            pH=3.51,
            sulphates=0.56,
            alcohol=9.4
        )
        
        import time
        start = time.time()
        _ = predictor_service.predict(test_sample)
        latency_ms = (time.time() - start) * 1000
        
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_version=predictor_service.get_model_version(),
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

# ============================================================================
# ENDPOINTS - Predictions
# ============================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predição single",
    status_code=status.HTTP_200_OK
)
async def predict(sample: WineSample):
    """
    Fazer predição para uma amostra de vinho.
    
    **Features necessárias (11):**
    - fixed_acidity: Acidez fixa (g/L)
    - volatile_acidity: Acidez volátil (g/L)
    - citric_acid: Ácido cítrico (g/L)
    - residual_sugar: Açúcar residual (g/L)
    - chlorides: Cloretos (g/L)
    - free_sulfur_dioxide: SO2 livre (mg/L)
    - total_sulfur_dioxide: SO2 total (mg/L)
    - density: Densidade (g/cm³)
    - pH: pH
    - sulphates: Sulfatos (g/L)
    - alcohol: Álcool (%vol)
    
    **Retorna:**
    - prediction: 0 (bad wine) ou 1 (good wine)
    - confidence: Probabilidade [0-1]
    - probabilities: {0: prob_bad, 1: prob_good}
    - interpretation: Explicação textual
    - latency_ms: Tempo de inferência
    
    **Exemplo:**
```json
    {
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
```
    """
    try:
        result = predictor_service.predict(sample)
        return result
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predição em lote",
    status_code=status.HTTP_200_OK
)
async def predict_batch(request: WineBatchRequest):
    """
    Fazer predições para múltiplas amostras de vinho.
    
    **Limite:** 100 amostras por request
    
    **Retorna:**
    - predictions: Lista de predições
    - total: Total de amostras
    - latency_ms: Tempo total de inferência
    - avg_latency_ms: Tempo médio por amostra
    
    **Exemplo:**
```json
    {
        "samples": [
            {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                ...
            },
            {
                "fixed_acidity": 8.1,
                "volatile_acidity": 0.6,
                ...
            }
        ]
    }
```
    """
    try:
        # Validar tamanho do batch
        if len(request.samples) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size exceeds maximum of 100 samples"
            )
        
        if len(request.samples) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch must contain at least 1 sample"
            )
        
        result = predictor_service.predict_batch(request.samples)
        return result
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

# ============================================================================
# ENDPOINTS - Model Info
# ============================================================================

@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Informações do modelo"
)
async def model_info():
    """
    Obter informações sobre o modelo em produção.
    
    Retorna:
    - model_version: Versão do modelo
    - model_type: Tipo de algoritmo
    - features: Lista de features esperadas
    - metrics: Métricas de performance
    - trained_at: Data/hora do treino
    """
    try:
        info = predictor_service.get_model_info()
        return info
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handler customizado para HTTPException."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handler genérico para exceções não tratadas."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================================
# Run Server (para desenvolvimento)
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload em desenvolvimento
        log_level="info"
    )