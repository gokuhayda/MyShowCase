"""
Predictor Service
=================
Serviço que encapsula lógica de predição.
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.predict import WineQualityPredictor
from api.models import WineSample, PredictionResponse, BatchPredictionResponse, ModelInfoResponse

class PredictorService:
    """
    Serviço de predição com cache e logging.
    """
    
    def __init__(self, model_path: str = "models/model.pkl"):
        """
        Inicializa o serviço.
        
        Args:
            model_path: Caminho para o modelo
        """
        self.model_path = model_path
        self.predictor = WineQualityPredictor(model_path)
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict:
        """Carrega métricas do modelo."""
        metrics_path = "models/metrics.json"
        
        if Path(metrics_path).exists():
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return {}
    
    def predict(self, sample: WineSample) -> PredictionResponse:
        """
        Fazer predição para uma amostra.
        
        Args:
            sample: WineSample com as features
            
        Returns:
            PredictionResponse com resultado
        """
        # Converter Pydantic model para dict
        sample_dict = sample.dict()
        
        # Medir latência
        start_time = time.time()
        
        # Predição
        result = self.predictor.predict_with_confidence(sample_dict)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Formatar resposta
        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            probabilities={
                "0": result['probabilities'][0],
                "1": result['probabilities'][1]
            },
            interpretation=result['interpretation'],
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now().isoformat()
        )
    
    def predict_batch(self, samples: List[WineSample]) -> BatchPredictionResponse:
        """
        Fazer predições em lote.
        
        Args:
            samples: Lista de WineSample
            
        Returns:
            BatchPredictionResponse com resultados
        """
        start_time = time.time()
        
        predictions = []
        for sample in samples:
            pred = self.predict(sample)
            predictions.append(pred)
        
        total_latency_ms = (time.time() - start_time) * 1000
        avg_latency_ms = total_latency_ms / len(samples) if samples else 0
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(samples),
            latency_ms=round(total_latency_ms, 2),
            avg_latency_ms=round(avg_latency_ms, 2),
            timestamp=datetime.now().isoformat()
        )
    
    def get_model_version(self) -> str:
        """Retorna versão do modelo."""
        # TODO: Ler de metadata ou Git tag
        return "v1.0.0"
    
    def get_model_info(self) -> ModelInfoResponse:
        """
        Retorna informações sobre o modelo.
        """
        return ModelInfoResponse(
            model_version=self.get_model_version(),
            model_type="RandomForestClassifier",
            features=self.predictor.feature_names,
            metrics=self.metrics if self.metrics else None,
            trained_at=self.metrics.get('timestamp') if self.metrics else None
        )