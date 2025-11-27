import unittest
import pandas as pd
# Importa a função load_data do seu script de treino
# (Precisamos garantir que train.py está acessível como módulo)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import load_data

class TestDataQuality(unittest.TestCase):
    
    def setUp(self):
        self.df = load_data()

    def test_column_names(self):
        """Schema Validation: Colunas esperadas existem?"""
        expected_cols = [
            'sepal length (cm)', 'sepal width (cm)', 
            'petal length (cm)', 'petal width (cm)', 
            'target'
        ]
        # Verifica se todas as colunas esperadas estão no DataFrame
        for col in expected_cols:
            self.assertIn(col, self.df.columns)

    def test_no_missing_values(self):
        """Data Quality: Não pode haver nulos"""
        self.assertFalse(self.df.isnull().values.any())

    def test_target_values(self):
        """Data Quality: Target deve ser válido (0, 1, 2 para Iris)"""
        unique_targets = sorted(self.df['target'].unique())
        self.assertEqual(unique_targets, [0, 1, 2])

    def test_data_volume(self):
        """Data Volume: Temos dados suficientes?"""
        self.assertGreater(len(self.df), 100, "Dataset muito pequeno!")

if __name__ == "__main__":
    unittest.main()
