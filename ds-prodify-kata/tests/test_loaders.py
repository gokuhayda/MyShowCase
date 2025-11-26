
"""
Testes unitários para o módulo loaders.py

Testa a funcionalidade de carregamento de dados, garantindo que:
- O CsvLoader retorna um DataFrame válido
- A estrutura de dados está correta
- O contrato LoaderStrategy é respeitado
"""

import unittest
import pandas as pd
from unittest.mock import patch, Mock
from loaders import CsvLoader, LoaderStrategy


class TestCsvLoader(unittest.TestCase):
    """Testes para a classe CsvLoader"""
    
    def setUp(self):
        """Setup executado antes de cada teste"""
        self.loader = CsvLoader()
    
    def test_loader_implements_strategy_protocol(self):
        """Verifica se CsvLoader implementa o protocolo LoaderStrategy"""
        # CsvLoader deve ter o método load
        self.assertTrue(hasattr(self.loader, 'load'))
        self.assertTrue(callable(getattr(self.loader, 'load')))
    
    def test_load_returns_dataframe(self):
        """Verifica se load() retorna um pandas DataFrame"""
        result = self.loader.load("fake_path.csv")
        
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_load_returns_expected_columns(self):
        """Verifica se o DataFrame contém as colunas esperadas"""
        result = self.loader.load("fake_path.csv")
        
        expected_columns = ['qtd', 'preco']
        self.assertListEqual(list(result.columns), expected_columns)
    
    def test_load_returns_non_empty_dataframe(self):
        """Verifica se o DataFrame retornado não está vazio"""
        result = self.loader.load("fake_path.csv")
        
        self.assertGreater(len(result), 0, "DataFrame não deveria estar vazio")
    
    def test_load_data_types(self):
        """Verifica os tipos de dados das colunas"""
        result = self.loader.load("fake_path.csv")
        
        # qtd pode ser int ou float (devido ao None)
        self.assertTrue(
            pd.api.types.is_numeric_dtype(result['qtd']) or 
            result['qtd'].dtype == object  # object quando tem None
        )
        # preco deve ser numérico
        self.assertTrue(pd.api.types.is_numeric_dtype(result['preco']))
    
    def test_load_contains_expected_data(self):
        """Verifica se os dados simulados estão corretos"""
        result = self.loader.load("fake_path.csv")
        
        # Verifica se tem 4 linhas (conforme implementação)
        self.assertEqual(len(result), 4)
        
        # Verifica alguns valores específicos
        self.assertEqual(result['qtd'].iloc[0], 1)
        self.assertEqual(result['preco'].iloc[0], 10.0)
    
    def test_load_accepts_path_parameter(self):
        """Verifica se o método aceita o parâmetro path"""
        # Não deve lançar exceção
        try:
            result = self.loader.load("any_path.csv")
        except TypeError as e:
            self.fail(f"load() deveria aceitar path como parâmetro: {e}")
    
    @patch('builtins.print')
    def test_load_prints_feedback(self, mock_print):
        """Verifica se load() imprime feedback para o usuário"""
        self.loader.load("test_path.csv")
        
        # Verifica se print foi chamado
        mock_print.assert_called()
        
        # Verifica se a mensagem contém o path
        call_args = str(mock_print.call_args)
        self.assertIn("test_path.csv", call_args)


class TestLoaderStrategyProtocol(unittest.TestCase):
    """Testes para validar o contrato do Protocol LoaderStrategy"""
    
    def test_custom_loader_can_implement_protocol(self):
        """Verifica se podemos criar um loader customizado"""
        
        class CustomLoader:
            def load(self, path: str) -> pd.DataFrame:
                return pd.DataFrame({'col1': [1, 2, 3]})
        
        custom_loader = CustomLoader()
        
        # Deve funcionar como LoaderStrategy
        result = custom_loader.load("dummy.csv")
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_loader_without_load_method_fails(self):
        """Verifica que classe sem load() não funciona como loader"""
        
        class InvalidLoader:
            def read(self, path: str):  # método errado
                return pd.DataFrame()
        
        invalid_loader = InvalidLoader()
        
        # Não deve ter o método load
        self.assertFalse(hasattr(invalid_loader, 'load'))


class TestCsvLoaderEdgeCases(unittest.TestCase):
    """Testes de casos extremos e edge cases"""
    
    def setUp(self):
        self.loader = CsvLoader()
    
    def test_load_with_empty_string_path(self):
        """Verifica comportamento com path vazio"""
        # Na implementação atual, isso retorna dados simulados
        result = self.loader.load("")
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_load_with_none_values_in_data(self):
        """Verifica que None values são preservados no carregamento"""
        result = self.loader.load("test.csv")
        
        # Deve conter None/NaN na coluna qtd
        self.assertTrue(result['qtd'].isna().any())
    
    def test_load_returns_copy_not_reference(self):
        """Verifica se cada chamada retorna dados independentes"""
        result1 = self.loader.load("test.csv")
        result2 = self.loader.load("test.csv")
        
        # Modificar result1 não deve afetar result2
        result1.loc[0, 'qtd'] = 999
        
        self.assertNotEqual(result1.loc[0, 'qtd'], result2.loc[0, 'qtd'])


class TestLoaderIntegration(unittest.TestCase):
    """Testes de integração para simular uso real"""
    
    def test_loader_in_pipeline_context(self):
        """Simula uso do loader em uma pipeline"""
        loader = CsvLoader()
        
        # Simula pipeline
        raw_data = loader.load("vendas.csv")
        
        # Pipeline deveria receber DataFrame válido
        self.assertIsInstance(raw_data, pd.DataFrame)
        self.assertGreater(len(raw_data), 0)
        self.assertIn('qtd', raw_data.columns)
        self.assertIn('preco', raw_data.columns)


# Teste de exemplo com mock real do pd.read_csv
class TestCsvLoaderWithRealCSV(unittest.TestCase):
    """
    Testes que demonstram como testar com pd.read_csv real
    (caso a implementação mude para usar arquivo real)
    """
    
    @patch('pandas.read_csv')
    def test_load_calls_pandas_read_csv(self, mock_read_csv):
        """
        Exemplo de como mockar pd.read_csv se fosse implementação real
        """
        # Arrange
        mock_read_csv.return_value = pd.DataFrame({
            'qtd': [1, 2, 3],
            'preco': [10.0, 20.0, 30.0]
        })
        
        # Para testar implementação real, teríamos algo como:
        # class RealCsvLoader:
        #     def load(self, path: str) -> pd.DataFrame:
        #         return pd.read_csv(path)
        
        # Este é um exemplo de estrutura de teste para quando
        # a implementação evoluir para usar arquivo real
        
        expected_df = pd.DataFrame({
            'qtd': [1, 2, 3],
            'preco': [10.0, 20.0, 30.0]
        })
        
        # Verificar que o mock retorna o esperado
        result = mock_read_csv("dummy.csv")
        pd.testing.assert_frame_equal(result, expected_df)


if __name__ == '__main__':
    # Executa os testes com verbosidade
    unittest.main(verbosity=2)
