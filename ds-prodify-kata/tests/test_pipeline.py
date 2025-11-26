
"""
Testes de integração para o módulo orchestrator.py

Este módulo testa a TrainingPipeline, que é o orquestrador principal.
Foco em:
- Integração entre componentes
- Fluxo de dados correto
- Dependency Injection
- Comportamento end-to-end
"""

import unittest
import pandas as pd
from unittest.mock import Mock, patch, call
from dataclasses import dataclass

from orchestrator import TrainingPipeline
from loaders import CsvLoader, LoaderStrategy
from cleaners import SalesDataCleaner, CleanerStrategy
from trainers import ModelTrainer


class TestTrainingPipelineStructure(unittest.TestCase):
    """Testes da estrutura da TrainingPipeline"""
    
    def test_pipeline_is_dataclass(self):
        """Verifica se TrainingPipeline é um dataclass"""
        # dataclass gera __dataclass_fields__
        self.assertTrue(hasattr(TrainingPipeline, '__dataclass_fields__'))
    
    def test_pipeline_has_required_attributes(self):
        """Verifica se pipeline tem os atributos esperados"""
        pipeline = TrainingPipeline(
            loader=Mock(),
            cleaner=Mock(),
            trainer=Mock()
        )
        
        self.assertTrue(hasattr(pipeline, 'loader'))
        self.assertTrue(hasattr(pipeline, 'cleaner'))
        self.assertTrue(hasattr(pipeline, 'trainer'))
    
    def test_pipeline_has_run_method(self):
        """Verifica se pipeline tem método run"""
        pipeline = TrainingPipeline(
            loader=Mock(),
            cleaner=Mock(),
            trainer=Mock()
        )
        
        self.assertTrue(hasattr(pipeline, 'run'))
        self.assertTrue(callable(getattr(pipeline, 'run')))
    
    def test_pipeline_constructor_accepts_dependencies(self):
        """
        🎯 TESTE DE DEPENDENCY INJECTION
        Verifica se pipeline aceita dependências no construtor
        """
        loader = CsvLoader()
        cleaner = SalesDataCleaner()
        trainer = ModelTrainer()
        
        # Não deve lançar exceção
        try:
            pipeline = TrainingPipeline(
                loader=loader,
                cleaner=cleaner,
                trainer=trainer
            )
        except Exception as e:
            self.fail(f"Pipeline deveria aceitar dependências: {e}")


class TestTrainingPipelineOrchestration(unittest.TestCase):
    """
    Testes do comportamento de orquestração
    🎯 TESTE CRÍTICO: Pipeline deve chamar componentes na ordem correta
    """
    
    def test_pipeline_calls_loader_first(self):
        """Verifica se loader é chamado primeiro"""
        mock_loader = Mock()
        mock_loader.load.return_value = pd.DataFrame({'qtd': [1], 'preco': [10.0]})
        
        mock_cleaner = Mock()
        mock_cleaner.clean.return_value = pd.DataFrame({
            'qtd': [1], 'preco': [10.0], 'total': [10.0]
        })
        
        mock_trainer = Mock()
        mock_trainer.train.return_value = "mock_model"
        
        pipeline = TrainingPipeline(
            loader=mock_loader,
            cleaner=mock_cleaner,
            trainer=mock_trainer
        )
        
        pipeline.run("test.csv")
        
        # Loader deve ter sido chamado
        mock_loader.load.assert_called_once_with("test.csv")
    
    def test_pipeline_calls_cleaner_second(self):
        """Verifica se cleaner é chamado após loader"""
        mock_loader = Mock()
        raw_data = pd.DataFrame({'qtd': [1], 'preco': [10.0]})
        mock_loader.load.return_value = raw_data
        
        mock_cleaner = Mock()
        mock_cleaner.clean.return_value = pd.DataFrame({
            'qtd': [1], 'preco': [10.0], 'total': [10.0]
        })
        
        mock_trainer = Mock()
        mock_trainer.train.return_value = "mock_model"
        
        pipeline = TrainingPipeline(
            loader=mock_loader,
            cleaner=mock_cleaner,
            trainer=mock_trainer
        )
        
        pipeline.run("test.csv")
        
        # Cleaner deve ter sido chamado com o retorno do loader
        mock_cleaner.clean.assert_called_once()
        
        # Verifica se recebeu DataFrame do loader
        call_args = mock_cleaner.clean.call_args[0][0]
        pd.testing.assert_frame_equal(call_args, raw_data)
    
    def test_pipeline_calls_trainer_third(self):
        """Verifica se trainer é chamado por último"""
        mock_loader = Mock()
        mock_loader.load.return_value = pd.DataFrame({'qtd': [1], 'preco': [10.0]})
        
        mock_cleaner = Mock()
        clean_data = pd.DataFrame({
            'qtd': [1], 'preco': [10.0], 'total': [10.0]
        })
        mock_cleaner.clean.return_value = clean_data
        
        mock_trainer = Mock()
        mock_trainer.train.return_value = "mock_model"
        
        pipeline = TrainingPipeline(
            loader=mock_loader,
            cleaner=mock_cleaner,
            trainer=mock_trainer
        )
        
        pipeline.run("test.csv")
        
        # Trainer deve ter sido chamado com o retorno do cleaner
        mock_trainer.train.assert_called_once()
        
        # Verifica se recebeu DataFrame do cleaner
        call_args = mock_trainer.train.call_args[0][0]
        pd.testing.assert_frame_equal(call_args, clean_data)
    
    def test_pipeline_calls_components_in_order(self):
        """
        🎯 TESTE CRÍTICO DE ORQUESTRAÇÃO
        Verifica se componentes são chamados na ordem correta
        """
        call_order = []
        
        mock_loader = Mock()
        mock_loader.load.side_effect = lambda x: (
            call_order.append('loader'),
            pd.DataFrame({'qtd': [1], 'preco': [10.0]})
        )[1]
        
        mock_cleaner = Mock()
        mock_cleaner.clean.side_effect = lambda x: (
            call_order.append('cleaner'),
            pd.DataFrame({'qtd': [1], 'preco': [10.0], 'total': [10.0]})
        )[1]
        
        mock_trainer = Mock()
        mock_trainer.train.side_effect = lambda x: (
            call_order.append('trainer'),
            "model"
        )[1]
        
        pipeline = TrainingPipeline(
            loader=mock_loader,
            cleaner=mock_cleaner,
            trainer=mock_trainer
        )
        
        pipeline.run("test.csv")
        
        # Ordem deve ser: loader -> cleaner -> trainer
        self.assertEqual(call_order, ['loader', 'cleaner', 'trainer'])
    
    def test_pipeline_returns_trained_model(self):
        """Verifica se pipeline retorna o modelo treinado"""
        mock_loader = Mock()
        mock_loader.load.return_value = pd.DataFrame({'qtd': [1], 'preco': [10.0]})
        
        mock_cleaner = Mock()
        mock_cleaner.clean.return_value = pd.DataFrame({
            'qtd': [1], 'preco': [10.0], 'total': [10.0]
        })
        
        expected_model = "trained_model_object"
        mock_trainer = Mock()
        mock_trainer.train.return_value = expected_model
        
        pipeline = TrainingPipeline(
            loader=mock_loader,
            cleaner=mock_cleaner,
            trainer=mock_trainer
        )
        
        result = pipeline.run("test.csv")
        
        # Deve retornar o modelo
        self.assertEqual(result, expected_model)


class TestTrainingPipelineDataFlow(unittest.TestCase):
    """
    Testes do fluxo de dados entre componentes
    """
    
    def test_pipeline_raw_data_flows_to_cleaner(self):
        """
        🎯 TESTE DE FLUXO DE DADOS
        Verifica se dados brutos do loader chegam ao cleaner
        """
        raw_data = pd.DataFrame({
            'qtd': [1, None, 3],
            'preco': [10.0, 20.0, 30.0]
        })
        
        mock_loader = Mock()
        mock_loader.load.return_value = raw_data
        
        captured_data = None
        
        def capture_clean_input(df):
            nonlocal captured_data
            captured_data = df.copy()
            return pd.DataFrame({
                'qtd': [1, 3], 'preco': [10.0, 30.0], 'total': [10.0, 90.0]
            })
        
        mock_cleaner = Mock()
        mock_cleaner.clean.side_effect = capture_clean_input
        
        mock_trainer = Mock()
        mock_trainer.train.return_value = "model"
        
        pipeline = TrainingPipeline(
            loader=mock_loader,
            cleaner=mock_cleaner,
            trainer=mock_trainer
        )
        
        pipeline.run("test.csv")
        
        # Cleaner deve ter recebido dados do loader
        pd.testing.assert_frame_equal(captured_data, raw_data)
    
    def test_pipeline_clean_data_flows_to_trainer(self):
        """
        🎯 TESTE DE FLUXO DE DADOS
        Verifica se dados limpos do cleaner chegam ao trainer
        """
        clean_data = pd.DataFrame({
            'qtd': [1, 3],
            'preco': [10.0, 30.0],
            'total': [10.0, 90.0]
        })
        
        mock_loader = Mock()
        mock_loader.load.return_value = pd.DataFrame({'qtd': [1], 'preco': [10.0]})
        
        mock_cleaner = Mock()
        mock_cleaner.clean.return_value = clean_data
        
        captured_data = None
        
        def capture_train_input(df):
            nonlocal captured_data
            captured_data = df.copy()
            return "model"
        
        mock_trainer = Mock()
        mock_trainer.train.side_effect = capture_train_input
        
        pipeline = TrainingPipeline(
            loader=mock_loader,
            cleaner=mock_cleaner,
            trainer=mock_trainer
        )
        
        pipeline.run("test.csv")
        
        # Trainer deve ter recebido dados do cleaner
        pd.testing.assert_frame_equal(captured_data, clean_data)


class TestTrainingPipelineWithRealComponents(unittest.TestCase):
    """
    Testes de integração usando componentes reais (não mocks)
    🎯 TESTES END-TO-END
    """
    
    def test_pipeline_with_all_real_components(self):
        """
        Teste end-to-end com todos os componentes reais
        """
        pipeline = TrainingPipeline(
            loader=CsvLoader(),
            cleaner=SalesDataCleaner(),
            trainer=ModelTrainer()
        )
        
        # Não deve lançar exceção
        try:
            model = pipeline.run("vendas.csv")
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Pipeline completa falhou: {e}")
    
    def test_pipeline_end_to_end_data_transformation(self):
        """
        Verifica transformação completa dos dados
        """
        # Captura estados intermediários
        original_loader_load = CsvLoader.load
        original_cleaner_clean = SalesDataCleaner.clean
        
        raw_data_captured = None
        clean_data_captured = None
        
        def capture_loader(self, path):
            nonlocal raw_data_captured
            result = original_loader_load(self, path)
            raw_data_captured = result.copy()
            return result
        
        def capture_cleaner(self, df):
            nonlocal clean_data_captured
            result = original_cleaner_clean(self, df)
            clean_data_captured = result.copy()
            return result
        
        with patch.object(CsvLoader, 'load', capture_loader):
            with patch.object(SalesDataCleaner, 'clean', capture_cleaner):
                pipeline = TrainingPipeline(
                    loader=CsvLoader(),
                    cleaner=SalesDataCleaner(),
                    trainer=ModelTrainer()
                )
                
                model = pipeline.run("test.csv")
        
        # Verificações
        self.assertIsNotNone(raw_data_captured)
        self.assertIsNotNone(clean_data_captured)
        
        # Raw data deve ter None
        self.assertTrue(raw_data_captured['qtd'].isna().any())
        
        # Clean data NÃO deve ter None
        self.assertFalse(clean_data_captured['qtd'].isna().any())
        
        # Clean data deve ter coluna total
        self.assertIn('total', clean_data_captured.columns)
    
    @patch('builtins.print')
    def test_pipeline_prints_progress(self, mock_print):
        """Verifica se pipeline imprime progresso"""
        pipeline = TrainingPipeline(
            loader=CsvLoader(),
            cleaner=SalesDataCleaner(),
            trainer=ModelTrainer()
        )
        
        pipeline.run("test.csv")
        
        # Deve ter múltiplos prints (início, cada componente, fim)
        self.assertGreater(mock_print.call_count, 1)


class TestTrainingPipelineErrorHandling(unittest.TestCase):
    """
    Testes de tratamento de erros
    """
    
    def test_pipeline_propagates_loader_error(self):
        """Verifica se erros do loader são propagados"""
        mock_loader = Mock()
        mock_loader.load.side_effect = FileNotFoundError("Arquivo não encontrado")
        
        pipeline = TrainingPipeline(
            loader=mock_loader,
            cleaner=Mock(),
            trainer=Mock()
        )
        
        with self.assertRaises(FileNotFoundError):
            pipeline.run("nonexistent.csv")
    
    def test_pipeline_propagates_cleaner_error(self):
        """Verifica se erros do cleaner são propagados"""
        mock_loader = Mock()
        mock_loader.load.return_value = pd.DataFrame({'qtd': [1]})
        
        mock_cleaner = Mock()
        mock_cleaner.clean.side_effect = KeyError("Coluna faltando")
        
        pipeline = TrainingPipeline(
            loader=mock_loader,
            cleaner=mock_cleaner,
            trainer=Mock()
        )
        
        with self.assertRaises(KeyError):
            pipeline.run("test.csv")
    
    def test_pipeline_propagates_trainer_error(self):
        """Verifica se erros do trainer são propagados"""
        mock_loader = Mock()
        mock_loader.load.return_value = pd.DataFrame({'qtd': [1]})
        
        mock_cleaner = Mock()
        mock_cleaner.clean.return_value = pd.DataFrame({'qtd': [1]})
        
        mock_trainer = Mock()
        mock_trainer.train.side_effect = ValueError("Dados inválidos")
        
        pipeline = TrainingPipeline(
            loader=mock_loader,
            cleaner=mock_cleaner,
            trainer=mock_trainer
        )
        
        with self.assertRaises(ValueError):
            pipeline.run("test.csv")


class TestTrainingPipelineExtensibility(unittest.TestCase):
    """
    Testes que demonstram extensibilidade (Open/Closed Principle)
    🎯 PRINCÍPIO SOLID: Open for extension, closed for modification
    """
    
    def test_pipeline_accepts_custom_loader(self):
        """
        Verifica se podemos usar loader customizado sem modificar pipeline
        """
        class CustomLoader:
            def load(self, path: str) -> pd.DataFrame:
                return pd.DataFrame({
                    'qtd': [100, 200],
                    'preco': [50.0, 75.0]
                })
        
        # Pipeline aceita novo loader SEM MODIFICAÇÃO
        pipeline = TrainingPipeline(
            loader=CustomLoader(),
            cleaner=SalesDataCleaner(),
            trainer=ModelTrainer()
        )
        
        try:
            model = pipeline.run("dummy.csv")
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Pipeline deveria aceitar loader customizado: {e}")
    
    def test_pipeline_accepts_custom_cleaner(self):
        """
        Verifica se podemos usar cleaner customizado sem modificar pipeline
        """
        class CustomCleaner:
            def clean(self, df: pd.DataFrame) -> pd.DataFrame:
                # Lógica diferente
                df = df.copy()
                df = df[df['qtd'] > 0]  # Remove qtd <= 0
                df['total'] = df['qtd'] * df['preco'] * 1.1  # Adiciona taxa
                return df
        
        # Pipeline aceita novo cleaner SEM MODIFICAÇÃO
        pipeline = TrainingPipeline(
            loader=CsvLoader(),
            cleaner=CustomCleaner(),
            trainer=ModelTrainer()
        )
        
        try:
            model = pipeline.run("dummy.csv")
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Pipeline deveria aceitar cleaner customizado: {e}")
    
    def test_pipeline_accepts_custom_trainer(self):
        """
        Verifica se podemos usar trainer customizado sem modificar pipeline
        """
        class CustomTrainer:
            def train(self, df: pd.DataFrame):
                # Simula outro algoritmo
                return "custom_model_xgboost"
        
        # Pipeline aceita novo trainer SEM MODIFICAÇÃO
        pipeline = TrainingPipeline(
            loader=CsvLoader(),
            cleaner=SalesDataCleaner(),
            trainer=CustomTrainer()
        )
        
        model = pipeline.run("dummy.csv")
        self.assertEqual(model, "custom_model_xgboost")


class TestTrainingPipelineResponsibility(unittest.TestCase):
    """
    Testes que verificam Single Responsibility Principle
    🎯 Pipeline deve APENAS orquestrar, não fazer o trabalho
    """
    
    def test_pipeline_does_not_load_data_itself(self):
        """Pipeline NÃO deve ter lógica de carregamento"""
        pipeline = TrainingPipeline(
            loader=Mock(),
            cleaner=Mock(),
            trainer=Mock()
        )
        
        # Pipeline não deve ter métodos de carregamento
        self.assertFalse(hasattr(pipeline, 'read_csv'))
        self.assertFalse(hasattr(pipeline, 'load_data'))
    
    def test_pipeline_does_not_clean_data_itself(self):
        """Pipeline NÃO deve ter lógica de limpeza"""
        pipeline = TrainingPipeline(
            loader=Mock(),
            cleaner=Mock(),
            trainer=Mock()
        )
        
        # Pipeline não deve ter métodos de limpeza
        self.assertFalse(hasattr(pipeline, 'dropna'))
        self.assertFalse(hasattr(pipeline, 'clean_data'))
    
    def test_pipeline_does_not_train_model_itself(self):
        """Pipeline NÃO deve ter lógica de treinamento"""
        pipeline = TrainingPipeline(
            loader=Mock(),
            cleaner=Mock(),
            trainer=Mock()
        )
        
        # Pipeline não deve ter métodos de ML
        self.assertFalse(hasattr(pipeline, 'fit'))
        self.assertFalse(hasattr(pipeline, 'train_model'))
    
    def test_pipeline_only_orchestrates(self):
        """
        🎯 TESTE FILOSÓFICO
        Pipeline deve apenas chamar métodos dos componentes
        """
        mock_loader = Mock()
        mock_loader.load.return_value = pd.DataFrame({'qtd': [1], 'preco': [10.0]})
        
        mock_cleaner = Mock()
        mock_cleaner.clean.return_value = pd.DataFrame({
            'qtd': [1], 'preco': [10.0], 'total': [10.0]
        })
        
        mock_trainer = Mock()
        mock_trainer.train.return_value = "model"
        
        pipeline = TrainingPipeline(
            loader=mock_loader,
            cleaner=mock_cleaner,
            trainer=mock_trainer
        )
        
        pipeline.run("test.csv")
        
        # Pipeline deve ter chamado cada componente exatamente uma vez
        self.assertEqual(mock_loader.load.call_count, 1)
        self.assertEqual(mock_cleaner.clean.call_count, 1)
        self.assertEqual(mock_trainer.train.call_count, 1)
        
        # E nada mais - pipeline não faz trabalho pesado


class TestTrainingPipelineScenarios(unittest.TestCase):
    """
    Testes de cenários reais de uso
    """
    
    def test_scenario_changing_data_source(self):
        """
        Cenário: Cliente muda fonte de dados de CSV para BigQuery
        """
        # Simula BigQueryLoader
        class BigQueryLoader:
            def load(self, query: str) -> pd.DataFrame:
                # Simula query do BigQuery
                return pd.DataFrame({
                    'qtd': [10, 20, 30],
                    'preco': [100.0, 200.0, 300.0]
                })
        
        # Troca loader SEM MUDAR a pipeline
        pipeline = TrainingPipeline(
            loader=BigQueryLoader(),  # ← Única mudança
            cleaner=SalesDataCleaner(),
            trainer=ModelTrainer()
        )
        
        model = pipeline.run("SELECT * FROM sales")
        self.assertIsNotNone(model)
    
    def test_scenario_ab_testing_cleaners(self):
        """
        Cenário: A/B test entre duas estratégias de limpeza
        """
        class ConservativeCleaner:
            def clean(self, df: pd.DataFrame) -> pd.DataFrame:
                # Remove apenas nulls
                df = df.dropna().copy()
                df['total'] = df['qtd'] * df['preco']
                return df
        
        class AggressiveCleaner:
            def clean(self, df: pd.DataFrame) -> pd.DataFrame:
                # Remove nulls E outliers
                df = df.dropna().copy()
                df = df[df['qtd'] < 1000]  # Remove outliers
                df['total'] = df['qtd'] * df['preco']
                return df
        
        # Pipeline A (conservativa)
        pipeline_a = TrainingPipeline(
            loader=CsvLoader(),
            cleaner=ConservativeCleaner(),
            trainer=ModelTrainer()
        )
        
        # Pipeline B (agressiva)
        pipeline_b = TrainingPipeline(
            loader=CsvLoader(),
            cleaner=AggressiveCleaner(),
            trainer=ModelTrainer()
        )
        
        # Ambas funcionam sem modificar código da pipeline
        model_a = pipeline_a.run("test.csv")
        model_b = pipeline_b.run("test.csv")
        
        self.assertIsNotNone(model_a)
        self.assertIsNotNone(model_b)


if __name__ == '__main__':
    # Executa os testes com verbosidade
    unittest.main(verbosity=2)
