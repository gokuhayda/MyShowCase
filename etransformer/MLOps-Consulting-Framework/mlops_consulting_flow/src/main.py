#!/usr/bin/env python
import sys
import asyncio
import logging
import warnings
from datetime import datetime

from crews.poem_crew.poem_crew import MLOpsCrew  # Importa a configuração do Crew
from flow import MLOpsConsultingFlow  # Importa o fluxo principal



# Ignora warnings desnecessários para um output mais limpo
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Caminho para a configuração global
CONFIG_PATH = "crews/poem_crew/config/mlops_globals.yaml"


def run():
    """
    Executa o fluxo MLOps Consulting.
    """
    logging.info("Iniciando o fluxo MLOps Consulting...")

    # Instancia o fluxo
    mlops_flow = MLOpsConsultingFlow(config_path=CONFIG_PATH)

    try:
        asyncio.run(mlops_flow.run())
        logging.info("Fluxo MLOps Consulting concluído com sucesso!")
    except Exception as e:
        logging.error(f"Erro ao executar o fluxo: {e}")
        raise


def train():
    """
    Treina o Crew para um número específico de iterações.
    Exemplo de uso: `python main.py train 5 treinamento.pkl`
    """
    if len(sys.argv) < 3:
        print("Uso: python main.py train <n_iterations> <output_file>")
        sys.exit(1)

    n_iterations = int(sys.argv[2])
    output_file = sys.argv[3]

    logging.info(f"Iniciando o treinamento com {n_iterations} iterações...")

    try:
        MLOpsCrew().crew().train(n_iterations=n_iterations, filename=output_file)
        logging.info("Treinamento concluído!")
    except Exception as e:
        logging.error(f"Erro durante o treinamento: {e}")
        raise


def replay():
    """
    Reexecuta o Crew a partir de uma tarefa específica.
    Exemplo de uso: `python main.py replay <task_id>`
    """
    if len(sys.argv) < 2:
        print("Uso: python main.py replay <task_id>")
        sys.exit(1)

    task_id = sys.argv[2]

    logging.info(f"Reexecutando o Crew a partir da tarefa {task_id}...")

    try:
        MLOpsCrew().crew().replay(task_id=task_id)
        logging.info("Replay concluído!")
    except Exception as e:
        logging.error(f"Erro ao reexecutar o Crew: {e}")
        raise


def test():
    """
    Testa a execução do Crew e retorna os resultados.
    Exemplo de uso: `python main.py test 5 gpt-3.5-turbo`
    """
    if len(sys.argv) < 3:
        print("Uso: python main.py test <n_iterations> <openai_model_name>")
        sys.exit(1)

    n_iterations = int(sys.argv[2])
    model_name = sys.argv[3]

    logging.info(f"Iniciando testes com {n_iterations} iterações no modelo {model_name}...")

    try:
        MLOpsCrew().crew().test(n_iterations=n_iterations, openai_model_name=model_name)
        logging.info("Testes concluídos!")
    except Exception as e:
        logging.error(f"Erro durante os testes: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main.py <run|train|replay|test>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "run":
        run()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"Comando desconhecido: {command}")
        sys.exit(1)
