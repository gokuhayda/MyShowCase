import os
import streamlit as st
from app.utils import select_folder, process_pdfs
from app.indexing import create_index_with_rerankers, query_index_with_rerankers

# Configuração inicial da interface
st.title("Sistema de Consulta a PDFs com GPT-4")
st.sidebar.header("Configuração de API e Seleção de Pasta")

# Configuração da API OpenAI
openai_api_key = st.sidebar.text_input("Digite sua chave de API OpenAI", type="password")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    st.success("Chave de API configurada com sucesso!")

# Seleção de pasta para PDFs
if st.sidebar.button("Selecionar Pasta", key="select_folder_button"):
    base_directory = select_folder()
    if base_directory:
        # Criar a estrutura pdf-gpt4-query dentro da pasta selecionada
        pdf_directory = os.path.join(base_directory)
        os.makedirs(pdf_directory, exist_ok=True)

        # Contar PDFs na pasta base
        pdf_files = [f for f in os.listdir(base_directory) if f.endswith(".pdf")]
        if not pdf_files:
            st.sidebar.error("Nenhum arquivo PDF encontrado na pasta selecionada.")
        else:
            st.sidebar.write(f"Pasta selecionada: {base_directory}")
            st.sidebar.write(f"Número de arquivos PDF encontrados: {len(pdf_files)}")
    else:
        st.sidebar.error("Nenhuma pasta foi selecionada.")

# Criação de índice vetorial
if "pdf_directory" in locals() and pdf_directory:
    if "vector_index" not in st.session_state:
        with st.spinner("Criando índice vetorial..."):
            try:
                # Criar o índice vetorial usando os PDFs processados
                st.session_state.vector_index = create_index_with_rerankers(base_directory)
                st.success("Índice criado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao criar o índice: {e}")

# Interface para consulta
st.header("Faça uma consulta")
question = st.text_input("Digite sua pergunta sobre os documentos", key="question_input")
if st.button("Consultar", key="query_button") and question:
    try:
        with st.spinner("Buscando resposta..."):
            # Consultar o índice com a pergunta fornecida
            answer = query_index_with_rerankers(question)
            st.session_state.answer = answer  # Armazenar resposta no estado
    except ValueError as e:
        st.error(f"Erro ao realizar consulta: {e}")
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")

# Exibir a resposta armazenada
if "answer" in st.session_state:
    st.write("**Resposta:**")
    st.write(st.session_state.answer)
