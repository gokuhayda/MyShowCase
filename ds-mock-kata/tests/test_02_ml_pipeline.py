import unittest
from unittest.mock import patch
import numpy as np
# Importação robusta
try:
    from katas.b02_ml_pipeline.model_trainer import ModelTrainer
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from katas.b02_ml_pipeline.model_trainer import ModelTrainer

class TestMLPipeline(unittest.TestCase):
    
    @patch('katas.b02_ml_pipeline.model_trainer.RandomForestClassifier')
    def test_train_and_evaluate_flow(self, mock_rf_class):
        # --- ARRANGE ---
        # Acesso direto ao return_value (mais idiomático)
        mock_instance = mock_rf_class.return_value
        mock_instance.predict.return_value = np.array([1, 0])
        
        trainer = ModelTrainer()
        X_train = np.array([[1, 1], [2, 2]])
        y_train = np.array([1, 0])
        X_test  = np.array([[3, 3], [4, 4]])
        y_test  = np.array([1, 0])
        
        # --- ACT ---
        accuracy = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        # --- ASSERT ---
        # Verifica o construtor 
        mock_rf_class.assert_called_once_with(n_estimators=100, random_state=42)
        
        # Verifica os métodos
        mock_instance.fit.assert_called_once_with(X_train, y_train)
        mock_instance.predict.assert_called_once_with(X_test)
        
        # Verifica o resultado
        self.assertEqual(accuracy, 1.0)
        
        print("\n✅ Kata 02: ML Training isolado com sucesso!")
