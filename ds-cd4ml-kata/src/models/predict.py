"""
Model Inference Pipeline
Carrega modelo treinado e faz predições
"""
import pandas as pd
import pickle
import numpy as np
import json
import time
from pathlib import Path
from typing import Union, List, Dict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WineQualityPredictor:
    """
    Preditor de qualidade de vinho.
    
    Attributes:
        model: Modelo treinado (RandomForest)
        feature_names: Lista de features esperadas
        metrics: Métricas do modelo (do metrics.json)
    """
    
    def __init__(self, model_path: str = "models/model.pkl"):
        """
        Inicializa o preditor carregando o modelo.
        
        Args:
            model_path: Caminho para o modelo serializado (.pkl)
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.metrics = None
        
        self._load_model()
        self._load_metrics()
    
    def _load_model(self):
        """Carrega modelo do disco."""
        logger.info(f"📂 Loading model from {self.model_path}")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Extrair feature names (se disponível)
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = list(self.model.feature_names_in_)
        else:
            # Fallback: features esperadas do dataset
            self.feature_names = [
                'fixed_acidity', 'volatile_acidity', 'citric_acid',
                'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
            ]
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Features expected: {len(self.feature_names)}")
    
    def _load_metrics(self):
        """Carrega métricas do modelo."""
        metrics_path = "models/metrics.json"
        
        if Path(metrics_path).exists():
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
            logger.info(f"📊 Model metrics loaded:")
            logger.info(f"   Test Accuracy: {self.metrics.get('test_accuracy', 'N/A'):.4f}")
            logger.info(f"   Test F1: {self.metrics.get('test_f1', 'N/A'):.4f}")
    
    def predict(self, X: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        Faz predição(ões).
        
        Args:
            X: Features em formato:
               - DataFrame: múltiplas predições
               - Dict: single predição {feature: value}
               - List[Dict]: múltiplas predições
        
        Returns:
            Array com predições (0 ou 1)
        
        Example:
            >>> predictor = WineQualityPredictor()
            >>> sample = {'fixed_acidity': 7.4, 'volatile_acidity': 0.7, ...}
            >>> prediction = predictor.predict(sample)
            >>> print(prediction)  # [1] (good wine)
        """
        # Converter input para DataFrame
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        
        # Validar features
        self._validate_features(X)
        
        # Predição
        start_time = time.time()
        predictions = self.model.predict(X)
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Prediction completed in {latency_ms:.2f}ms")
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        Retorna probabilidades de cada classe.
        
        Returns:
            Array com shape (n_samples, 2)
            - [:, 0]: probabilidade de classe 0 (bad wine)
            - [:, 1]: probabilidade de classe 1 (good wine)
        """
        # Converter input
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        
        # Validar
        self._validate_features(X)
        
        # Probabilidades
        probas = self.model.predict_proba(X)
        
        return probas
    
    def predict_with_confidence(self, X: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Predição com confiança e explicação.
        
        Returns:
            Dict com:
            - prediction: classe predita (0 ou 1)
            - confidence: probabilidade da classe predita
            - probabilities: {0: prob_0, 1: prob_1}
            - interpretation: string explicativa
        """
        # Converter para DataFrame
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        # Predição e probabilidades
        pred = self.predict(X)[0]
        probas = self.predict_proba(X)[0]
        
        result = {
            'prediction': int(pred),
            'confidence': float(probas[pred]),
            'probabilities': {
                0: float(probas[0]),
                1: float(probas[1])
            },
            'interpretation': self._interpret(pred, probas[pred])
        }
        
        return result
    
    def _validate_features(self, X: pd.DataFrame):
        """Valida se features esperadas estão presentes."""
        missing = set(self.feature_names) - set(X.columns)
        
        if missing:
            raise ValueError(
                f"Missing features: {missing}\n"
                f"Expected: {self.feature_names}\n"
                f"Got: {list(X.columns)}"
            )
        
        # Reordenar colunas para match do treino
        X = X[self.feature_names]
    
    def _interpret(self, prediction: int, confidence: float) -> str:
        """Gera interpretação textual da predição."""
        quality = "Good Wine" if prediction == 1 else "Bad Wine"
        confidence_pct = confidence * 100
        
        if confidence >= 0.9:
            level = "Very High"
        elif confidence >= 0.75:
            level = "High"
        elif confidence >= 0.6:
            level = "Moderate"
        else:
            level = "Low"
        
        return f"{quality} ({level} confidence: {confidence_pct:.1f}%)"


def main():
    """Exemplo de uso do preditor."""
    logger.info("=" * 70)
    logger.info("🍷 Wine Quality Predictor - Example Usage")
    logger.info("=" * 70)
    
    # Inicializar preditor
    predictor = WineQualityPredictor()
    
    # Exemplo 1: Single prediction (dict)
    logger.info("\n📊 Example 1: Single prediction")
    sample_wine = {
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
    
    result = predictor.predict_with_confidence(sample_wine)
    logger.info(f"   Input: {sample_wine}")
    logger.info(f"   Prediction: {result['prediction']}")
    logger.info(f"   Confidence: {result['confidence']:.4f}")
    logger.info(f"   Interpretation: {result['interpretation']}")
    
    # Exemplo 2: Batch prediction
    logger.info("\n📊 Example 2: Batch prediction")
    batch_wines = [
        sample_wine,
        {**sample_wine, 'alcohol': 12.5, 'sulphates': 0.8},  # Melhor vinho
    ]
    
    predictions = predictor.predict(batch_wines)
    logger.info(f"   Batch size: {len(batch_wines)}")
    logger.info(f"   Predictions: {predictions}")
    
    # Exemplo 3: Probabilidades
    logger.info("\n📊 Example 3: Probabilities")
    probas = predictor.predict_proba(sample_wine)
    logger.info(f"   Prob(bad wine): {probas[0][0]:.4f}")
    logger.info(f"   Prob(good wine): {probas[0][1]:.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Prediction examples completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()