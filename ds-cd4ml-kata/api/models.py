"""
Pydantic Models - Schema Validation
====================================
Modelos Pydantic para validação de dados da API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from datetime import datetime

# ============================================================================
# REQUEST MODELS
# ============================================================================

class WineSample(BaseModel):
    """
    Schema para uma amostra de vinho.
    
    Todas as features são obrigatórias para predição.
    """
    fixed_acidity: float = Field(
        ...,
        ge=0,
        le=20,
        description="Acidez fixa (g/L tartaric acid)",
        example=7.4
    )
    volatile_acidity: float = Field(
        ...,
        ge=0,
        le=2,
        description="Acidez volátil (g/L acetic acid)",
        example=0.7
    )
    citric_acid: float = Field(
        ...,
        ge=0,
        le=2,
        description="Ácido cítrico (g/L)",
        example=0.0
    )
    residual_sugar: float = Field(
        ...,
        ge=0,
        le=20,
        description="Açúcar residual (g/L)",
        example=1.9
    )
    chlorides: float = Field(
        ...,
        ge=0,
        le=1,
        description="Cloretos (g/L sodium chloride)",
        example=0.076
    )
    free_sulfur_dioxide: float = Field(
        ...,
        ge=0,
        le=100,
        description="SO2 livre (mg/L)",
        example=11.0
    )
    total_sulfur_dioxide: float = Field(
        ...,
        ge=0,
        le=300,
        description="SO2 total (mg/L)",
        example=34.0
    )
    density: float = Field(
        ...,
        ge=0.99,
        le=1.01,
        description="Densidade (g/cm³)",
        example=0.9978
    )
    pH: float = Field(
        ...,
        ge=2.5,
        le=4.5,
        description="pH",
        example=3.51
    )
    sulphates: float = Field(
        ...,
        ge=0,
        le=2,
        description="Sulfatos (g/L potassium sulphate)",
        example=0.56
    )
    alcohol: float = Field(
        ...,
        ge=8,
        le=15,
        description="Álcool (% vol)",
        example=9.4
    )
    
    class Config:
        schema_extra = {
            "example": {
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
        }

class WineBatchRequest(BaseModel):
    """Schema para batch de amostras."""
    samples: List[WineSample] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="Lista de amostras (máx 100)"
    )
    
    @validator('samples')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 samples")
        return v

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class PredictionResponse(BaseModel):
    """Schema para resposta de predição single."""
    prediction: int = Field(..., description="Predição: 0 (bad wine) ou 1 (good wine)")
    confidence: float = Field(..., ge=0, le=1, description="Confiança da predição [0-1]")
    probabilities: Dict[str, float] = Field(..., description="Probabilidades de cada classe")
    interpretation: str = Field(..., description="Interpretação textual do resultado")
    latency_ms: float = Field(..., description="Tempo de inferência (ms)")
    timestamp: str = Field(..., description="Timestamp da predição (ISO format)")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "confidence": 0.87,
                "probabilities": {
                    "0": 0.13,
                    "1": 0.87
                },
                "interpretation": "Good Wine (High confidence: 87.0%)",
                "latency_ms": 5.23,
                "timestamp": "2024-12-01T10:30:00.123456"
            }
        }

class BatchPredictionResponse(BaseModel):
    """Schema para resposta de predição em lote."""
    predictions: List[PredictionResponse] = Field(..., description="Lista de predições")
    total: int = Field(..., description="Total de amostras processadas")
    latency_ms: float = Field(..., description="Tempo total de inferência (ms)")
    avg_latency_ms: float = Field(..., description="Tempo médio por amostra (ms)")
    timestamp: str = Field(..., description="Timestamp do batch (ISO format)")

class HealthResponse(BaseModel):
    """Schema para resposta de health check."""
    status: str = Field(..., description="Status da API: 'healthy' ou 'unhealthy'")
    model_loaded: bool = Field(..., description="Modelo está carregado?")
    model_version: str = Field(..., description="Versão do modelo")
    latency_ms: float = Field(..., description="Latência do smoke test (ms)")
    timestamp: str = Field(..., description="Timestamp do health check")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "v1.0.0",
                "latency_ms": 4.52,
                "timestamp": "2024-12-01T10:30:00.123456"
            }
        }

class ModelInfoResponse(BaseModel):
    """Schema para informações do modelo."""
    model_version: str = Field(..., description="Versão do modelo")
    model_type: str = Field(..., description="Tipo de algoritmo")
    features: List[str] = Field(..., description="Lista de features esperadas")
    metrics: Optional[Dict[str, float]] = Field(None, description="Métricas de performance")
    trained_at: Optional[str] = Field(None, description="Data/hora do treino")
    
    class Config:
        schema_extra = {
            "example": {
                "model_version": "v1.0.0",
                "model_type": "RandomForestClassifier",
                "features": [
                    "fixed_acidity", "volatile_acidity", "citric_acid",
                    "residual_sugar", "chlorides", "free_sulfur_dioxide",
                    "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"
                ],
                "metrics": {
                    "test_accuracy": 0.8656,
                    "test_f1": 0.8839,
                    "test_precision": 0.8571,
                    "test_recall": 0.9123
                },
                "trained_at": "2024-12-01T08:00:00"
            }
        }