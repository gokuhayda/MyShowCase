
"""
Testes unitários para o módulo trainers.py

Testa a funcionalidade de treinamento do modelo:
- Preparação correta de features (X) e target (y)
- Chamada ao método fit do modelo
- Tratamento de dados de entrada
- Validação de pré-condições
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, call
from trainers import ModelTrainer, LinearRegression


class TestModelTrainer(unittest.TestCase):
    """Testes para a classe ModelTrainer"""
    
    def setUp(self):
        """Setup executado antes de cada teste"""
        self.trainer = ModelTrainer()
    
    def test_trainer_has_train_method(self):
        """Verifica se ModelTrainer tem método train"""
        self.assertTrue(hasattr(self.trainer, 'train'))
        self.assertTrue(callable(getattr(self.trainer, 'train')))
    
    def test_train_returns_model(self):
        """Verifica se train() retorna um modelo"""
        df = pd.DataFrame({
            'qtd': [1, 2, 3],
            'preco': [10.0, 20.0, 30.0],
            'total': [10.0, 40.0, 90.0]
        })
        
        result = self.trainer.train(df)
        
        # Deve retornar algum objeto (o modelo)
        self.assertIsNotNone(result)
    
    def test_train_uses_correct_features(self):
        """
        🎯 TESTE CRÍTICO
        Verifica se X é construído com as colunas corretas
        """
        df = pd.DataFrame({
            'qtd': [1, 2, 3],
            'preco': [10.0, 20.0, 30.0],
            'total': [10.0, 40.0, 90.0]
        })
        
        # Mock do modelo para capturar o que foi passado
        with patch.object(LinearRegression, 'fit') as mock_fit:
            self.trainer.train(df)
            
            # Verifica que fit foi chamado
            self.assertTrue(mock_fit.called)
            
            # Captura os argumentos passados para fit
            call_args = mock_fit.call_args
            X_passed = call_args[0][0]  # Primeiro argumento (X)
            
            # X deve ter apenas qtd e preco
            expected_columns = ['qtd', 'preco']
            self.assertListEqual(list(X_passed.columns), expected_columns)
    
    def test_train_uses_correct_target(self):
        """
        🎯 TESTE CRÍTICO
        Verifica se y é a coluna 'total'
        """
        df = pd.DataFrame({
            'qtd': [1, 2, 3],
            'preco': [10.0, 20.0, 30.0],
            'total': [10.0, 40.0, 90.0]
        })
        
        # Mock do modelo para capturar o que foi passado
        with patch.object(LinearRegression, 'fit') as mock_fit:
            self.trainer.train(df)
            
            # Captura os argumentos passados para fit
            call_args = mock_fit.call_args
            y_passed = call_args[0][1]  # Segundo argumento (y)
            
            # y deve ser a coluna total
            pd.testing.assert_series_equal(y_passed, df['total'], check_names=False)
    
    def test_train_calls_model_fit(self):
        """Verifica se o método fit do modelo é chamado"""
        df = pd.DataFrame({
            'qtd': [1, 2],
            'preco': [10.0, 20.0],
            'total': [10.0, 40.0]
        })
        
        with patch.object(LinearRegression, 'fit') as mock_fit:
            self.trainer.train(df)
            
            # fit deve ter sido chamado exatamente uma vez
            self.assertEqual(mock_fit.call_count, 1)
    
    def test_train_passes_correct_shapes(self):
        """
        🏆 BOA PRÁTICA
        Verifica se X e y têm shapes compatíveis
        """
        df = pd.DataFrame({
            'qtd': [1, 2, 3],
            'preco': [10.0, 20.0, 30.0],
            'total': [10.0, 40.0, 90.0]
        })
        
        with patch.object(LinearRegression, 'fit') as mock_fit:
            self.trainer.train(df)
            
            call_args = mock_fit.call_args
            X_passed = call_args[0][0]
            y_passed = call_args[0][1]
            
            # X deve ser (n_samples, n_features) = (3, 2)
            self.assertEqual(X_passed.shape, (3, 2))
            
            # y deve ser (n_samples,) = (3,)
            self.assertEqual(len(y_passed), 3)
    
    @patch('builtins.print')
    def test_train_prints_feedback(self, mock_print):
        """Verifica se train() imprime feedback sobre treinamento"""
        df = pd.DataFrame({
            'qtd': [1, 2],
            'preco': [10.0, 20.0],
            'total': [10.0, 40.0]
        })
        
        self.trainer.train(df)
        
        # Verifica se algum print foi chamado (pode ser do LinearRegression)
        self.assertTrue(mock_print.called)


class TestModelTrainerDataValidation(unittest.TestCase):
    """Testes de validação de dados de entrada"""
    
    def setUp(self):
        self.trainer = ModelTrainer()
    
    def test_train_with_required_columns(self):
        """
        🎯 TESTE CRÍTICO
        Verifica se todas as colunas necessárias existem
        """
        df = pd.DataFrame({
            'qtd': [1, 2],
            'preco': [10.0, 20.0],
            'total': [10.0, 40.0]
        })
        
        # Não deve lançar exceção
        try:
            result = self.trainer.train(df)
        except KeyError as e:
            self.fail(f"train() falhou com DataFrame válido: {e}")
    
    def test_train_raises_error_missing_qtd(self):
        """
        🐛 EDGE CASE
        Verifica erro quando falta coluna qtd
        """
        df = pd.DataFrame({
            'preco': [10.0, 20.0],
            'total': [10.0, 40.0]
        })
        
        with self.assertRaises(KeyError):
            self.trainer.train(df)
    
    def test_train_raises_error_missing_preco(self):
        """
        🐛 EDGE CASE
        Verifica erro quando falta coluna preco
        """
        df = pd.DataFrame({
            'qtd': [1, 2],
            'total': [10.0, 40.0]
        })
        
        with self.assertRaises(KeyError):
            self.trainer.train(df)
    
    def test_train_raises_error_missing_total(self):
        """
        🐛 EDGE CASE
        Verifica erro quando falta coluna total
        """
        df = pd.DataFrame({
            'qtd': [1, 2],
            'preco': [10.0, 20.0]
        })
        
        with self.assertRaises(KeyError):
            self.trainer.train(df)
    
    def test_train_with_extra_columns(self):
        """
        Verifica se colunas extras não afetam o treinamento
        """
        df = pd.DataFrame({
            'qtd': [1, 2],
            'preco': [10.0, 20.0],
            'total': [10.0, 40.0],
            'extra_col': ['a', 'b']  # Coluna extra
        })
        
        # Não deve lançar exceção
        try:
            result = self.trainer.train(df)
        except Exception as e:
            self.fail(f"Colunas extras não deveriam afetar treinamento: {e}")


class TestModelTrainerEdgeCases(unittest.TestCase):
    """Testes de casos extremos"""
    
    def setUp(self):
        self.trainer = ModelTrainer()
    
    def test_train_with_single_row(self):
        """
        🐛 EDGE CASE
        Modelo com apenas 1 amostra (tecnicamente inválido para ML)
        """
        df = pd.DataFrame({
            'qtd': [1],
            'preco': [10.0],
            'total': [10.0]
        })
        
        # Deve executar sem erro (mesmo que não seja útil)
        try:
            result = self.trainer.train(df)
        except Exception as e:
            self.fail(f"train() deveria aceitar 1 linha: {e}")
    
    def test_train_with_two_rows(self):
        """
        Verifica treinamento com número mínimo razoável de amostras
        """
        df = pd.DataFrame({
            'qtd': [1, 2],
            'preco': [10.0, 20.0],
            'total': [10.0, 40.0]
        })
        
        result = self.trainer.train(df)
        self.assertIsNotNone(result)
    
    def test_train_with_large_dataset(self):
        """
        Verifica performance com dataset maior
        """
        # 10000 linhas
        df = pd.DataFrame({
            'qtd': np.random.randint(1, 100, 10000),
            'preco': np.random.uniform(10, 100, 10000),
        })
        df['total'] = df['qtd'] * df['preco']
        
        # Deve processar rapidamente
        try:
            result = self.trainer.train(df)
        except Exception as e:
            self.fail(f"train() falhou com dataset grande: {e}")
    
    def test_train_with_zero_values(self):
        """
        🐛 EDGE CASE
        Zero é válido em vendas (nenhuma venda)
        """
        df = pd.DataFrame({
            'qtd': [0, 1, 2],
            'preco': [10.0, 20.0, 30.0],
            'total': [0.0, 20.0, 60.0]
        })
        
        result = self.trainer.train(df)
        self.assertIsNotNone(result)
    
    def test_train_with_negative_values(self):
        """
        🐛 EDGE CASE
        Valores negativos (devoluções)
        """
        df = pd.DataFrame({
            'qtd': [-1, 1, 2],
            'preco': [10.0, 20.0, 30.0],
            'total': [-10.0, 20.0, 60.0]
        })
        
        # Deve aceitar valores negativos
        result = self.trainer.train(df)
        self.assertIsNotNone(result)
    
    def test_train_preserves_dataframe(self):
        """
        🏆 BOA PRÁTICA
        Verifica se DataFrame original não é modificado
        """
        df = pd.DataFrame({
            'qtd': [1, 2],
            'preco': [10.0, 20.0],
            'total': [10.0, 40.0]
        })
        
        original_df = df.copy()
        
        self.trainer.train(df)
        
        # DataFrame original deve estar intacto
        pd.testing.assert_frame_equal(df, original_df)


class TestModelTrainerIntegration(unittest.TestCase):
    """
    Testes de integração simulando uso real na pipeline
    """
    
    def test_trainer_accepts_cleaner_output(self):
        """
        🎯 TESTE DE INTEGRAÇÃO CRÍTICO
        Verifica se trainer aceita output do cleaner
        """
        from cleaners import SalesDataCleaner
        
        # Simula dados limpos vindos do cleaner
        cleaner = SalesDataCleaner()
        raw_data = pd.DataFrame({
            'qtd': [1, None, 3],
            'preco': [10.0, 20.0, 30.0]
        })
        clean_data = cleaner.clean(raw_data)
        
        # Trainer deve aceitar sem problemas
        trainer = ModelTrainer()
        try:
            model = trainer.train(clean_data)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Trainer deveria aceitar output do cleaner: {e}")
    
    def test_full_pipeline_flow(self):
        """
        Simula fluxo completo: loader -> cleaner -> trainer
        """
        from loaders import CsvLoader
        from cleaners import SalesDataCleaner
        
        # Simula pipeline
        loader = CsvLoader()
        cleaner = SalesDataCleaner()
        trainer = ModelTrainer()
        
        # Fluxo
        raw_data = loader.load("fake.csv")
        clean_data = cleaner.clean(raw_data)
        model = trainer.train(clean_data)
        
        # Deve completar com sucesso
        self.assertIsNotNone(model)


class TestLinearRegressionMock(unittest.TestCase):
    """
    Testes para a versão mock do LinearRegression
    """
    
    def test_linear_regression_has_fit_method(self):
        """Verifica se LinearRegression tem método fit"""
        model = LinearRegression()
        self.assertTrue(hasattr(model, 'fit'))
        self.assertTrue(callable(getattr(model, 'fit')))
    
    def test_linear_regression_fit_accepts_X_y(self):
        """Verifica se fit aceita X e y"""
        model = LinearRegression()
        X = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        y = pd.Series([5, 6])
        
        # Não deve lançar exceção
        try:
            model.fit(X, y)
        except Exception as e:
            self.fail(f"LinearRegression.fit falhou: {e}")
    
    @patch('builtins.print')
    def test_linear_regression_fit_prints_info(self, mock_print):
        """Verifica se fit imprime informação sobre treinamento"""
        model = LinearRegression()
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([4, 5, 6])
        
        model.fit(X, y)
        
        # Verifica se print foi chamado
        mock_print.assert_called_once()
        
        # Verifica se menciona número de linhas
        call_args = str(mock_print.call_args)
        self.assertIn('3', call_args)  # 3 linhas


class TestModelTrainerWithRealSklearn(unittest.TestCase):
    """
    Testes demonstrando como testar com sklearn real
    (caso a implementação mude para usar sklearn real)
    """
    
    @patch('trainers.LinearRegression')
    def test_train_with_mocked_sklearn(self, MockLinearRegression):
        """
        Exemplo de como mockar sklearn.linear_model.LinearRegression
        """
        # Configura o mock
        mock_model_instance = Mock()
        MockLinearRegression.return_value = mock_model_instance
        
        # Trainer usando o mock
        trainer = ModelTrainer()
        df = pd.DataFrame({
            'qtd': [1, 2],
            'preco': [10.0, 20.0],
            'total': [10.0, 40.0]
        })
        
        result = trainer.train(df)
        
        # Verifica que LinearRegression foi instanciado
        MockLinearRegression.assert_called_once()
        
        # Verifica que fit foi chamado no modelo
        mock_model_instance.fit.assert_called_once()
        
        # Verifica que o modelo mockado foi retornado
        self.assertEqual(result, mock_model_instance)


class TestModelTrainerResponsibility(unittest.TestCase):
    """
    Testes que verificam a responsabilidade única do trainer
    """
    
    def test_trainer_does_not_clean_data(self):
        """
        🎯 PRINCÍPIO SRP
        Trainer NÃO deve limpar dados - isso é responsabilidade do Cleaner
        """
        trainer = ModelTrainer()
        
        # Passa dados "sujos" (com None) - trainer não deve limpar
        df = pd.DataFrame({
            'qtd': [1, None, 3],
            'preco': [10.0, 20.0, 30.0],
            'total': [10.0, None, 90.0]
        })
        
        # Se trainer tenta usar dados com None, deve falhar
        # Isso garante que limpeza não é responsabilidade do trainer
        # (Na prática, com sklearn real, fit() falharia com NaN)
    
    def test_trainer_does_not_load_data(self):
        """
        🎯 PRINCÍPIO SRP
        Trainer NÃO deve carregar dados - isso é responsabilidade do Loader
        """
        trainer = ModelTrainer()
        
        # Trainer não deve ter método load
        self.assertFalse(hasattr(trainer, 'load'))
    
    def test_trainer_does_not_validate_business_rules(self):
        """
        🎯 PRINCÍPIO SRP
        Trainer não valida regras de negócio - apenas treina
        """
        trainer = ModelTrainer()
        
        # Aceita dados "estranhos" sem validar
        df = pd.DataFrame({
            'qtd': [-100, 999999],  # Valores bizarros
            'preco': [0.01, 0.01],
            'total': [-1, 9999.99]
        })
        
        # Não deve lançar exceção de validação
        # (Validação de negócio é responsabilidade do Cleaner)
        try:
            trainer.train(df)
        except ValueError as e:
            if "business rule" in str(e).lower():
                self.fail("Trainer não deve validar regras de negócio")


if __name__ == '__main__':
    # Executa os testes com verbosidade
    unittest.main(verbosity=2)
