"""
Módulo de utilidades para extração e visualização de perguntas e respostas, com foco em suporte à avaliação via RAGAS.

Funções:
- extrair_qa_por_regex: usa expressão regular para extrair perguntas e respostas de um texto bruto.
- dividir_em_contextos: divide o conteúdo de texto em blocos lógicos.
- mostrar_resultados_ragas: exibe os resultados da avaliação RAGAS no Streamlit.
"""

import json
import re
import streamlit as st
from datasets import Dataset
import pandas as pd
import matplotlib.pyplot as plt

def extrair_qa_por_regex(conteudo, n_qas=5):
    """
    Extrai pares de pergunta e resposta de um texto bruto usando regex.

    Args:
        conteudo (str): Texto contendo perguntas e respostas.
        n_qas (int): Número máximo de pares a retornar.

    Returns:
        list: Lista de dicionários com campos 'pergunta' e 'resposta'.
    """
    padrao = re.compile(r"[Qq][:：]\s*(.*?)\s*(?:\r?\n)+[Aa][:：]\s*(.*?)(?=(?:\r?\n){2,}|$)", re.DOTALL)
    matches = padrao.findall(conteudo)
    return [{"pergunta": q.strip(), "resposta": a.strip()} for q, a in matches][:n_qas]

def dividir_em_contextos(conteudo):
    """
    Divide um texto em blocos de contexto separados por linhas em branco.

    Args:
        conteudo (str): Texto completo.

    Returns:
        list: Lista de blocos de texto.
    """
    return [p.strip() for p in conteudo.split("\n\n") if p.strip()]

def mostrar_resultados_ragas(resultados):
    """
    Exibe os resultados da avaliação RAGAS no Streamlit com histogramas.

    Args:
        resultados (EvaluationResult): Objeto de avaliação retornado pelo RAGAS.
    """
    try:
        df_ragas = resultados.to_pandas()
        metricas = df_ragas.mean(numeric_only=True)
        st.markdown("### 📈 Métricas médias")
        st.json({k: round(v, 3) for k, v in metricas.items()})
        st.markdown("### 📋 Resultados completos")
        st.dataframe(df_ragas)

        for metrica in metricas.index:
            fig, ax = plt.subplots()
            ax.hist(df_ragas[metrica].dropna(), bins=10)
            ax.set_title(metrica)
            ax.set_xlabel(metrica)
            ax.set_ylabel("Frequência")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Erro ao processar resultados do RAGAS: {e}")
