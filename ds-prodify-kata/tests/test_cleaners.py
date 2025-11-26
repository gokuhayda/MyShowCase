"""
Testes unitários para o módulo cleaners.py

Este é o componente CRÍTICO mencionado na pergunta de entrevista:
"O modelo está treinando com dados sujos - onde você investiga?"
Resposta: SalesDataCleaner

Testa:
- Remoção de valores nulos
- Cálculo correto da coluna 'total'
- Edge cases e validações
- Contrato CleanerStrategy
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
from .cleaners import SalesDataCleaner, CleanerStrategy


class TestSalesDataCleaner(unittest.TestCase):
    """Testes para a classe SalesDataCleaner"""
    
    def setUp(self):
        """Setup executado antes de cada teste"""
        self.cleaner = SalesDataCleaner()
    
    def test_cleaner_implements_strategy_protocol(self):
        """Verifica se SalesDataCleaner implementa o protocolo CleanerStrategy"""
        # SalesDataCleaner deve ter o método clean
        self.assertTrue(hasattr(self.cleaner, 'clean'))
        self.assertTrue(callable(getattr(self.cleaner, 'clean')))
    
    def test_clean_returns_dataframe(self):
        """Verifica se clean() retorna um pandas DataFrame"""
        df = pd.DataFrame({'qtd': [1, 2], 'preco': [10.0, 20.0]})
        result = self.cleaner.clean(df)
        
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_clean_removes_null_values(self):
        """
        🎯 TESTE CRÍTICO DA ENTREVISTA
        Verifica se valores nulos são removidos corretamente
        """
        df = pd.DataFrame({
            'qtd': [1, None, 3, None],
            'preco': [10.0, 20.0, 30.0, 40.0]
        })
        
        result = self.cleaner.clean(df)
        
        # Não deve ter valores nulos
        self.assertEqual(result.isnull().sum().sum(), 0)
        
        # Deve ter apenas 2 linhas (removeu 2 com None)
        self.assertEqual(len(result), 2)
    
    def test_clean_creates_total_column(self):
        """
        🎯 TESTE CRÍTICO DA ENTREVISTA
        Verifica se a coluna 'total' é criada corretamente
        """
        df = pd.DataFrame({
            'qtd': [1, 2, 3],
            'preco': [10.0, 20.0, 30.0]
        })
        
        result = self.cleaner.clean(df)
        
        # Deve ter a coluna total
        self.assertIn('total', result.columns)
    
    def test_clean_calculates_total_correctly(self):
        """
        🎯 TESTE CRÍTICO DA ENTREVISTA
        Verifica se o cálculo de total = qtd * preco está correto
        """
        df = pd.DataFrame({
            'qtd': [1, 2, 3],
            'preco': [10.0, 20.0, 30.0]
        })
        
        result = self.cleaner.clean(df)
        
        # Verifica cada valor
        expected_totals = [10.0, 40.0, 90.0]  # 1*10, 2*20, 3*30
        actual_totals = result['total'].tolist()
        
        self.assertEqual(actual_totals, expected_totals)
    
    def test_clean_preserves_original_columns(self):
        """Verifica se as colunas originais são mantidas"""
        df = pd.DataFrame({
            'qtd': [1, 2],
            'preco': [10.0, 20.0]
        })
        
        result = self.cleaner.clean(df)
        
        # Deve manter qtd e preco
        self.assertIn('qtd', result.columns)
        self.assertIn('preco', result.columns)
    
    def test_clean_does_not_modify_original_dataframe(self):
        """
        🏆 BOA PRÁTICA
        Verifica se o DataFrame original não é modificado (imutabilidade)
        """
        df = pd.DataFrame({
            'qtd': [1, None, 3],
            'preco': [10.0, 20.0, 30.0]
        })
        
        original_len = len(df)
        original_cols = list(df.columns)
        
        # Limpa
        result = self.cleaner.clean(df)
        
        # Original não deve ter mudado
        self.assertEqual(len(df), original_len)
        self.assertListEqual(list(df.columns), original_cols)
        self.assertNotIn('total', df.columns)  # Original não tem 'total'
    
    @patch('builtins.print')
    def test_clean_prints_feedback(self, mock_print):
        """Verifica se clean() imprime feedback para o usuário"""
        df = pd.DataFrame({'qtd': [1], 'preco': [10.0]})
        self.cleaner.clean(df)
        
        # Verifica se print foi chamado
        mock_print.assert_called()
        
        # Verifica se a mensagem é sobre limpeza
        call_args = str(mock_print.call_args)
        self.assertIn("Limpando", call_args.lower())


class TestSalesDataCleanerEdgeCases(unittest.TestCase):
    """Testes de casos extremos - cruciais para identificar bugs"""
    
    def setUp(self):
        self.cleaner = SalesDataCleaner()
    
    def test_clean_with_all_null_rows(self):
        """
        🐛 EDGE CASE IMPORTANTE
        Verifica comportamento quando todas as linhas têm nulos
        """
        df = pd.DataFrame({
            'qtd': [None, None, None],
            'preco': [10.0, 20.0, 30.0]
        })
        
        result = self.cleaner.clean(df)
        
        # Deve retornar DataFrame vazio
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_clean_with_zero_values(self):
        """
        🐛 EDGE CASE IMPORTANTE
        Zero é diferente de None - deve ser mantido
        """
        df = pd.DataFrame({
            'qtd': [0, 1, 2],
            'preco': [10.0, 20.0, 30.0]
        })
        
        result = self.cleaner.clean(df)
        
        # Deve manter linha com zero
        self.assertEqual(len(result), 3)
        self.assertEqual(result['qtd'].iloc[0], 0)
        self.assertEqual(result['total'].iloc[0], 0.0)  # 0 * 10 = 0
    
    def test_clean_with_negative_values(self):
        """
        🐛 EDGE CASE IMPORTANTE
        Valores negativos (devoluções?) devem ser tratados?
        """
        df = pd.DataFrame({
            'qtd': [-1, 2],
            'preco': [10.0, 20.0]
        })
        
        result = self.cleaner.clean(df)
        
        # Na implementação atual, negativos são mantidos
        self.assertEqual(len(result), 2)
        self.assertEqual(result['total'].iloc[0], -10.0)  # -1 * 10
    
    def test_clean_with_very_large_numbers(self):
        """
        🐛 EDGE CASE
        Verifica overflow com números grandes
        """
        df = pd.DataFrame({
            'qtd': [1e10, 2e10],
            'preco': [1e10, 2e10]
        })
        
        result = self.cleaner.clean(df)
        
        # Não deve ter overflow
        self.assertEqual(len(result), 2)
        self.assertTrue(np.isfinite(result['total'].iloc[0]))
    
    def test_clean_with_single_row(self):
        """Verifica comportamento com apenas uma linha"""
        df = pd.DataFrame({
            'qtd': [5],
            'preco': [100.0]
        })
        
        result = self.cleaner.clean(df)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result['total'].iloc[0], 500.0)
    
    def test_clean_with_mixed_null_positions(self):
        """
        🐛 EDGE CASE IMPORTANTE
        Nulos em diferentes colunas
        """
        df = pd.DataFrame({
            'qtd': [1, None, 3, 4],
            'preco': [10.0, 20.0, None, 40.0]
        })
        
        result = self.cleaner.clean(df)
        
        # Deve remover linhas com None em qualquer coluna
        self.assertEqual(len(result), 2)  # Linhas 0 e 3
        self.assertEqual(result['qtd'].iloc[0], 1)
        self.assertEqual(result['qtd'].iloc[1], 4)


class TestSalesDataCleanerDataIntegrity(unittest.TestCase):
    """
    Testes de integridade de dados - críticos para ML
    Dados sujos = modelo ruim
    """
    
    def setUp(self):
        self.cleaner = SalesDataCleaner()
    
    def test_clean_maintains_data_types(self):
        """Verifica se os tipos de dados são preservados"""
        df = pd.DataFrame({
            'qtd': [1, 2, 3],
            'preco': [10.0, 20.0, 30.0]
        })
        
        result = self.cleaner.clean(df)
        
        # Tipos numéricos devem ser mantidos
        self.assertTrue(pd.api.types.is_numeric_dtype(result['qtd']))
        self.assertTrue(pd.api.types.is_numeric_dtype(result['preco']))
        self.assertTrue(pd.api.types.is_numeric_dtype(result['total']))
    
    def test_clean_no_duplicate_rows(self):
        """Verifica se limpeza não cria duplicatas"""
        df = pd.DataFrame({
            'qtd': [1, 2, 3],
            'preco': [10.0, 20.0, 30.0]
        })
        
        result = self.cleaner.clean(df)
        
        # Não deve criar linhas duplicadas
        self.assertEqual(len(result), len(result.drop_duplicates()))
    
    def test_clean_preserves_row_order(self):
        """Verifica se a ordem das linhas é mantida após limpeza"""
        df = pd.DataFrame({
            'qtd': [1, None, 3, 4],
            'preco': [10.0, 20.0, 30.0, 40.0]
        })
        
        result = self.cleaner.clean(df)
        
        # Linhas válidas devem manter ordem relativa
        # [1, 3, 4] -> qtd deve estar em ordem crescente
        qtd_values = result['qtd'].tolist()
        self.assertEqual(qtd_values, [1, 3, 4])
    
    def test_clean_total_calculation_precision(self):
        """
        🐛 TESTE DE PRECISÃO
        Verifica precisão de ponto flutuante
        """
        df = pd.DataFrame({
            'qtd': [1, 2],
            'preco': [0.1, 0.2]  # Números que causam problemas de precisão
        })
        
        result = self.cleaner.clean(df)
        
        # Verifica valores com tolerância
        self.assertAlmostEqual(result['total'].iloc[0], 0.1, places=5)
        self.assertAlmostEqual(result['total'].iloc[1], 0.4, places=5)


class TestCleanerStrategyProtocol(unittest.TestCase):
    """Testes para validar o contrato do Protocol CleanerStrategy"""
    
    def test_custom_cleaner_can_implement_protocol(self):
        """Verifica se podemos criar um cleaner customizado"""
        
        class CustomCleaner:
            def clean(self, df: pd.DataFrame) -> pd.DataFrame:
                # Lógica customizada
                df = df.copy()
                df['processed'] = True
                return df
        
        custom_cleaner = CustomCleaner()
        df = pd.DataFrame({'col': [1, 2, 3]})
        
        # Deve funcionar como CleanerStrategy
        result = custom_cleaner.clean(df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('processed', result.columns)


class TestSalesDataCleanerIntegration(unittest.TestCase):
    """
    Testes de integração simulando uso real na pipeline
    """
    
    def test_cleaner_output_ready_for_training(self):
        """
        🎯 TESTE DE INTEGRAÇÃO CRÍTICO
        Verifica se output está pronto para o ModelTrainer
        """
        cleaner = SalesDataCleaner()
        
        # Simula dados que viriam do loader
        raw_data = pd.DataFrame({
            'qtd': [1, None, 3, 4],
            'preco': [10.0, 20.0, 30.0, 40.0]
        })
        
        # Limpa
        clean_data = cleaner.clean(raw_data)
        
        # Verifica se está pronto para treino
        # 1. Sem valores nulos
        self.assertEqual(clean_data.isnull().sum().sum(), 0)
        
        # 2. Tem todas as colunas necessárias
        required_columns = ['qtd', 'preco', 'total']
        for col in required_columns:
            self.assertIn(col, clean_data.columns)
        
        # 3. Tipos corretos (numéricos para ML)
        self.assertTrue(pd.api.types.is_numeric_dtype(clean_data['qtd']))
        self.assertTrue(pd.api.types.is_numeric_dtype(clean_data['preco']))
        self.assertTrue(pd.api.types.is_numeric_dtype(clean_data['total']))
        
        # 4. Tem dados suficientes
        self.assertGreater(len(clean_data), 0)
    
    def test_cleaner_handles_loader_output(self):
        """
        Testa com dados reais que viriam do CsvLoader
        """
        from loaders import CsvLoader
        
        cleaner = SalesDataCleaner()
        loader = CsvLoader()
        
        # Simula pipeline
        raw_data = loader.load("fake.csv")
        clean_data = cleaner.clean(raw_data)
        
        # Deve processar com sucesso
        self.assertIsInstance(clean_data, pd.DataFrame)
        self.assertGreater(len(clean_data), 0)
        self.assertIn('total', clean_data.columns)


class TestSalesDataCleanerRegressionTests(unittest.TestCase):
    """
    Testes de regressão - previnem bugs já corrigidos
    """
    
    def test_bug_total_column_overwrite(self):
        """
        Simula bug: e se DataFrame já tiver coluna 'total'?
        """
        df = pd.DataFrame({
            'qtd': [1, 2],
            'preco': [10.0, 20.0],
            'total': [999, 999]  # Valor errado pré-existente
        })
        
        cleaner = SalesDataCleaner()
        result = cleaner.clean(df)
        
        # Deve recalcular, não manter o errado
        self.assertEqual(result['total'].iloc[0], 10.0)  # 1*10, não 999
        self.assertEqual(result['total'].iloc[1], 40.0)  # 2*20, não 999


if __name__ == '__main__':
    # Executa os testes com verbosidade
    unittest.main(verbosity=2)
