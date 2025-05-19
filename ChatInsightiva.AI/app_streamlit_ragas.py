"""
App final unificado com parâmetros de modelo, estilo e controle total para avaliação com RAGAS.
# streamlit run app_streamlit_ragas.py --server.runOnSave false
"""

from dotenv import load_dotenv
load_dotenv()

from app_modules.qa_interface import executar_interface_streamlit

if __name__ == "__main__":
    executar_interface_streamlit()
