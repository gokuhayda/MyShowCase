import unittest
import json
import os

class TestModelQuality(unittest.TestCase):
    
    def test_accuracy_threshold(self):
        """
        Quality Gate: O modelo DEVE ter acurácia > 80%.
        Se for menor, o CI/CD quebra e impede o deploy.
        """
        metrics_path = "metrics/scores.json"
        
        if not os.path.exists(metrics_path):
            self.fail("Arquivo de métricas não encontrado! Rode o treino antes.")
            
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
        accuracy = metrics.get("accuracy", 0)
        print(f"\n🔍 Verificando acurácia: {accuracy:.2f}")
        
        # O Threshold de Aceitação
        self.assertGreaterEqual(accuracy, 0.80, "❌ Modelo rejeitado: Acurácia abaixo de 80%")
