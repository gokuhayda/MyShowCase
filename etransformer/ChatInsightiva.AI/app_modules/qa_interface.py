"""
Interface unificada Streamlit para avaliar respostas com RAGAS,
com barra de progresso, reescrita, resposta do bot e resumo final.
"""

import streamlit as st
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from app_modules.qa_extractor import extrair_dados_do_json
from app_modules.qa_rewriter import reescrever_perguntas
from utils_tools.qa_utils import mostrar_resultados_ragas

def executar_interface_streamlit():
    st.set_page_config(page_title="RAGAS QA Avaliação", layout="wide")
    st.title("📊 Avaliação de Perguntas e Respostas com RAGAS")

    with st.expander("📖 O que são essas métricas e como configurar os parâmetros?"):
        st.markdown("""
        ### 📊 Entendendo as Métricas de Qualidade (RAGAS)

        As métricas abaixo avaliam a **qualidade das respostas do chatbot** com base em critérios técnicos e de linguagem natural:

        - **🧠 Faithfulness (Fidelidade)**: Avalia se a resposta está fiel ao conteúdo apresentado nos contextos. Respostas inventadas ou distorcidas reduzem essa métrica.
        
        - **🎯 Answer Relevancy (Relevância da Resposta)**: Mede o quanto a resposta realmente responde à pergunta feita. Uma resposta genérica ou fora do tema reduz essa nota.
        
        - **📌 Context Precision (Precisão do Contexto)**: Verifica se os trechos usados para gerar a resposta são realmente úteis ou se há excesso de informação desnecessária.
        
        - **🔍 Context Recall (Abrangência do Contexto)**: Mede se os contextos utilizados cobrem tudo o que a resposta precisa. Se algo importante ficou de fora, essa métrica cai.

        ---

        ### ⚙️ Como configurar os parâmetros do modelo LLM

        Estes parâmetros ajudam a controlar o estilo e a forma como as perguntas serão reescritas:

        - **🔥 Temperatura**: Controla a criatividade do modelo. Valores mais baixos geram respostas mais seguras e previsíveis. Valores mais altos (ex: 0.9) tornam as respostas mais criativas e variadas.
        
        - **🌐 Top-p**: Determina o quanto o modelo deve considerar alternativas além das mais prováveis. Use 1.0 para permitir maior variedade e inclusão de ideias.
        
        - **🔁 Presence Penalty**: Penaliza o modelo se ele repetir temas já abordados. Aumente se quiser menos repetição de ideias.
        
        - **🔁 Frequency Penalty**: Penaliza palavras que se repetem com frequência. Útil para evitar redundância.
        
        - **🎭 Estilo da Reescrita**: Escolha se deseja perguntas mais formais, resumidas ou diretas. Isso afeta como o modelo reformula a pergunta original para fazer mais sentido ao chatbot.

        ---
        🧪 Dica: experimente diferentes combinações para ver como isso afeta os resultados. Cada modelo tem um "comportamento criativo" diferente!
        """)


    st.sidebar.header("⚙️ Parâmetros do Modelo")

    provedor = st.sidebar.radio("🧠 Provedor", ["OpenAI", "Ollama"], horizontal=True)
    modelo = st.sidebar.selectbox("Modelo", ["gpt-4", "gpt-3.5-turbo"] if provedor == "OpenAI" else ["llama3", "mistral", "codellama"])

    temperatura = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.7)
    top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 1.0)
    presence_penalty = st.sidebar.slider("Presence Penalty", 0.0, 2.0, 0.0)
    frequency_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 2.0, 0.0)

    estilos = st.sidebar.multiselect("🎭 Estilo da reescrita", ["informal","formal", "resumido", "instrucional", "assertivo"])

    modo_qas = st.sidebar.radio("Modo de Seleção", ["Amostra Aleatória", "Sequencial", "Usar todo o dataset"])
    n_qas = None
    if modo_qas != "Usar todo o dataset":
        n_qas = st.sidebar.slider("Nº máximo de pares QA", 5, 100, 30)

    perguntas_pers = ""
    usar_personalizadas = st.sidebar.checkbox("✍️ Inserir perguntas personalizadas")
    if usar_personalizadas:
        perguntas_pers = st.sidebar.text_area("Cole as perguntas (uma por linha):")

    reescrever = st.checkbox("🔁 Reescrever perguntas com LLM")

    if st.button("▶️ Executar avaliação"):
        with st.spinner("🔄 Extraindo dados..."):
            dataset = extrair_dados_do_json()

        if reescrever:
            
                    bloco_texto = st.empty()
        dataset = reescrever_perguntas(
            dataset,
            modelo=modelo,
            provedor=provedor,
            temperatura=temperatura,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            estilos=estilos,
            perguntas_personalizadas=perguntas_pers.splitlines() if perguntas_pers else None,
            max_qas=n_qas if n_qas else 9999,
            modo_qas=modo_qas,
            bloco_texto=bloco_texto
        )

        with st.spinner("⚙️ Calculando métricas com RAGAS..."):
            resultados = evaluate(dataset, metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ])
            mostrar_resultados_ragas(resultados)

            if resultados:
                df_ragas = resultados.to_pandas()
                st.download_button(
                    label="📥 Baixar resultados RAGAS (.csv)",
                    data=df_ragas.to_csv(index=False).encode('utf-8'),
                    file_name="avaliacao_ragas.csv",
                    mime="text/csv"
                )

# fim do módulo
