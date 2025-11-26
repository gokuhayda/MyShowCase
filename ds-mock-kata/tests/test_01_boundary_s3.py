import unittest
from unittest.mock import patch
import pandas as pd
from katas.b01_boundary_s3.data_loader import S3DataLoader

class TestS3Boundary(unittest.TestCase):
    """
    REGRA DE OURO: o mocker é USADO (no data_loader), não onde é DEFINIDO (pandas)

    """

    @patch("katas.b01_boundary_s3.data_loader.pd.read_csv")
    def test_load_csv(self, mock_read_csv):
        fake_df = pd.DataFrame({
               'id':[1, 2, 3],
               'product':['A', 'B', 'C']
        })
        
        mock_read_csv.return_value = fake_df

        loader = S3DataLoader()
        path = "s3//bucket-test/dados.csv"

        result = loader.load_csv(path)

        # 1 Verificação do Estado: O resultado é do nosso DF falso
        pd.testing.assert_frame_equal(result, fake_df)
        
        # 2. Verificação do Comportamento:  O pandas foi chamado com o caminho certo?
        mock_read_csv.assert_called_once_with(path)

        print("\n Kata 01: S3 Boundry isolado com sucesso!")