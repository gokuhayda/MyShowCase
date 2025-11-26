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
    
    # MOCK ONDE É USADO! Não em 'sklearn.ensemble.RandomForestClassifier'
    @patch('katas.b02_ml_pipeline.model_trainer.RandomForestClassifier')
    def test_train_and_evaluate_flow(self, mock_rf_class):
        # --- ARRANGE ---
        # 1. Configurar a Instância do Mock
        # Quando o código fizer `model = RandomForestClassifier()`, ele recebe este mock_instance
        mock_instance = mock_rf_class.return_value
        
        # 2. Configurar o retorno do .predict()
        # Simulamos que o modelo previu [1, 0] (para bater com y_test e dar 100% accuracy)
        mock_instance.predict.return_value = np.array([1, 0])
        
        trainer = ModelTrainer()
        
        # Dados Fakes (formato numpy)
        X_train = np.array([[1, 1], [2, 2]])
        y_train = np.array([1, 0])
        X_test  = np.array([[3, 3], [4, 4]])
        y_test  = np.array([1, 0]) # Gabarito igual à predição simulada
        
        # --- ACT ---
        accuracy = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        # --- ASSERT ---
        # 1. O modelo foi treinado? (Verificação de Comportamento)
        mock_instance.fit.assert_called_once_with(X_train, y_train)
        
        # 2. A predição foi feita?
        mock_instance.predict.assert_called_once_with(X_test)
        
        # 3. A lógica de acurácia funcionou? (Se previu [1,0] e gabarito é [1,0], acc deve ser 1.0)
        self.assertEqual(accuracy, 1.0)
        
        print("\n✅ Kata 02: ML Training isolado com sucesso!")
                        