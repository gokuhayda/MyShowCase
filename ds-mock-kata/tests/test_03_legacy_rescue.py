import unittest
from unittest.mock import Mock
import sys
import os

# Setup de path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from katas.b03_legacy_rescue.refactored.scoring_logic import ScoringLogic
from katas.b03_legacy_rescue.refactored.orchestrator import CustomerScoreOrchestrator

class TestLegacyRescue(unittest.TestCase):
    
    # TESTE 1: Lógica Pura (Sem Mocks!)
    # Testamos a matemática exaustivamente aqui. É rápido e não falha por causa de rede.
    def test_scoring_logic_math(self):
        # Dado: Age 30, Income 1000, History 2
        # Esperado: (30*0.1) + (1000/1000) + (2*5) = 3 + 1 + 10 = 14
        base = ScoringLogic.calculate_base_score(30, 1000, 2)
        self.assertEqual(base, 14.0)
        
        # Final: 14 * 0.5 = 7.0
        final = ScoringLogic.calculate_final_score(14.0, 0.5)
        self.assertEqual(final, 7.0)
        print("✅ Kata 03: Lógica Pura testada sem mocks!")

    # TESTE 2: Orquestração (Com Mocks nos Boundaries)
    # Aqui testamos apenas se os componentes conversam entre si.
    def test_orchestrator_flow(self):
        # --- ARRANGE ---
        # Criamos mocks para as dependências externas
        mock_api = Mock()
        mock_model = Mock()
        
        # Configuramos o comportamento (Stubs)
        mock_api.get_customer_data.return_value = {
            "age": 30, "income": 1000, "history": 2
        }
        mock_model.predict_proba.return_value = [[0.99, 0.5]] # 50% prob
        
        # INJETAMOS os mocks no orquestrador
        orchestrator = CustomerScoreOrchestrator(mock_api, mock_model)
        
        # --- ACT ---
        score = orchestrator.generate_score(999)
        
        # --- ASSERT ---
        # Verificamos se o orquestrador chamou a API correta
        mock_api.get_customer_data.assert_called_once_with(999)
        
        # Verificamos se o modelo foi chamado
        mock_model.predict_proba.assert_called_once()
        
        # Verificamos o resultado final (Lógica: 14 * 0.5 = 7.0)
        self.assertEqual(score, 7.0)
        print("✅ Kata 03: Orquestração testada com DI!")