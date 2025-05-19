import os
import fitz  # PyMuPDF para leitura de PDFs
import re
import pdfplumber
from typing import List

def select_folder():
    """
    Abre um diálogo para o usuário selecionar uma pasta (exclusivo para interfaces desktop).
    Retorna o caminho da pasta selecionada.
    """
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()  # Oculta a janela principal
    folder_path = filedialog.askdirectory(title="Selecione a pasta com PDFs")
    root.destroy()
    return folder_path


def pdf_to_text(pdf_path: str) -> str:
    """
    Extrai texto de um arquivo PDF usando a biblioteca pdfplumber.

    Args:
        pdf_path (str): Caminho para o arquivo PDF.

    Returns:
        str: Texto extraído do PDF.
    """
    text = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
                else:
                    print(f"⚠️ Página {i + 1} de '{pdf_path}' sem texto extraível.")

        return "\n\n".join(text).strip()

    except Exception as e:
        raise RuntimeError(f"Erro ao extrair texto de '{pdf_path}': {e}")


def process_pdfs(base_dir: str, output_dir: str) -> List[str]:
    """
    Processa arquivos PDF de um diretório, extrai o texto e salva como arquivos `.txt`.

    Args:
        base_dir (str): Caminho do diretório com PDFs.
        output_dir (str): Caminho do diretório de saída dos arquivos `.txt`.

    Returns:
        list: Lista dos caminhos dos arquivos `.txt` gerados.
    """
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(base_dir) if f.lower().endswith(".pdf")]
    output_files = []

    for file_name in pdf_files:
        try:
            pdf_path = os.path.join(base_dir, file_name)
            output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.txt")

            text = pdf_to_text(pdf_path)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)

            output_files.append(output_file)
            print(f"✅ Sucesso: '{file_name}' ➜ '{output_file}'")

        except Exception as e:
            print(f"❌ Falha ao processar '{file_name}': {e}")

    if not output_files:
        print("⚠️ Nenhum arquivo PDF foi processado com sucesso.")

    return output_files


def extract_tables_from_pdf(pdf_path: str) -> list:
    """
    Extrai tabelas de um PDF usando pdfplumber.

    Args:
        pdf_path (str): Caminho para o arquivo PDF.

    Returns:
        list: Lista de tabelas extraídas, onde cada tabela é uma lista de listas (linhas e colunas).
    """
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            if page_tables:
                print(f"📄 Página {i + 1}: {len(page_tables)} tabela(s) extraída(s).")
                tables.extend(page_tables)

    return tables


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Divide um texto em chunks de tamanho definido, com sobreposição opcional.

    Args:
        text (str): Texto a ser dividido.
        chunk_size (int): Tamanho máximo de cada chunk (em caracteres).
        overlap (int): Quantidade de caracteres a sobrepor entre os chunks.

    Returns:
        list: Lista de chunks de texto.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= chunk_size:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap // len(word.split()):]  # Adicionar sobreposição
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def clean_text(text):
    """
    Remove caracteres indesejados e normaliza o texto.

    Args:
        text (str): Texto a ser limpo.

    Returns:
        str: Texto limpo.
    """
    text = re.sub(r"\s+", " ", text)  # Remove espaços extras
    text = re.sub(r"[^a-zA-Z0-9.,!?;:()\[\] ]", "", text)  # Remove caracteres especiais
    return text.strip()


def save_chunks(chunks, output_file):
    """
    Salva uma lista de chunks de texto em um arquivo.

    Args:
        chunks (list): Lista de chunks de texto.
        output_file (str): Caminho do arquivo para salvar os chunks.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")
    print(f"Chunks salvos em: {output_file}")


def extract_metadata_from_text(text):
    """
    Extração de metadados simples de um texto (exemplo: FAT, SOAP).

    Args:
        text (str): Texto de onde extrair os metadados.

    Returns:
        dict: Metadados extraídos.
    """
    metadata = {
        "FAT": re.findall(r"\bFAT\b.*", text, re.IGNORECASE),
        "SOAP": re.findall(r"\bSOAP\b.*", text, re.IGNORECASE),
    }
    return metadata


def summarize_text(text, max_length=150):
    """
    Resume um texto para um comprimento máximo.

    Args:
        text (str): Texto a ser resumido.
        max_length (int): Comprimento máximo do resumo.

    Returns:
        str: Resumo do texto.
    """
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    summary = ""
    for sentence in sentences:
        if len(summary) + len(sentence) <= max_length:
            summary += sentence + " "
        else:
            break
    return summary.strip()

def log_interaction(question, response, source, similarity):
    with open("logs/conversas.log", "a") as f:
        f.write(f"{question} | {similarity:.2f} | {source} | {response}\n")


