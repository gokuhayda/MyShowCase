from tests.test_01_boundary_s3 import TestS3Boundary
from tests.test_02_ml_pipeline import TestMLPipeline
from tests.test_03_legacy_rescue import TestLegacyRescue
import unittest

def load_suite(option: int):
    match option:
        case 1:
            print("Rodando testes do Kata 01...")
            return unittest.TestLoader().loadTestsFromTestCase(TestS3Boundary)

        case 2:
            print("Rodando testes do Kata 02...")
            return unittest.TestLoader().loadTestsFromTestCase(TestMLPipeline)

        case 3:
            print("Rodando testes do Kata 03...")
            return unittest.TestLoader().loadTestsFromTestCase(TestLegacyRescue)
        
        case 4:
            print("Rodando TODOS os testes...")
            suite = unittest.TestSuite()
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestS3Boundary))
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMLPipeline))
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLegacyRescue))
            return suite

        case _:
            raise ValueError("Opção inválida. Escolha 1, 2 ou 3.")

if __name__ == "__main__":
    print("Escolha o kata para rodar:")
    print("1 = Boundary S3")
    print("2 = ML Pipeline")
    print("3 = Legacy Rescue")
    print("4 = Todos")

    opcao = int(input("Digite a opção: "))
    suite = load_suite(opcao)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    result.wasSuccessful()
