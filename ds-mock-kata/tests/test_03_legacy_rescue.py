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
        """
        Testa a lógica de scoring pura (Core Layer).
        Sem I/O, sem mocks - apenas matemática!
        """
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
        """
        Testa o fluxo de orquestração usando Dependency Injection.
        Mockamos as dependências externas (API e ML Model).
        """
        # --- ARRANGE ---
        # Criamos mocks para as dependências externas
        mock_api = Mock()
        mock_model = Mock()
        
        # Configuramos o comportamento (Stubs)
        mock_api.get_customer_data.return_value = {
            "age": 30, 
            "income": 1000, 
            "history": 2
        }
        
        # Modelo retorna [prob_classe_0, prob_classe_1]
        # O orchestrator usa o índice [1] (segunda posição)
        mock_model.predict_proba.return_value = [[0.5, 0.5]]  # 50% cada classe
        
        # INJETAMOS os mocks no orquestrador (Dependency Injection!)
        orchestrator = CustomerScoreOrchestrator(mock_api, mock_model)
        
        # --- ACT ---
        score = orchestrator.generate_score(999)
        
        # --- ASSERT ---
        # 1. Verificamos se o orquestrador chamou a API com o ID correto
        mock_api.get_customer_data.assert_called_once_with(999)
        
        # 2. Verificamos se o modelo foi chamado com as features corretas
        # Features esperadas: [[age, income, history]]
        mock_model.predict_proba.assert_called_once_with([[30, 1000, 2]])
        
        # 3. Verificamos o resultado final (Lógica: 14 * 0.5 = 7.0)
        self.assertEqual(score, 7.0)
        print("✅ Kata 03: Orquestração testada com DI!")
