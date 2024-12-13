import os
import fitz  # PyMuPDF para leitura de PDFs
import re

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


def pdf_to_text(pdf_path):
    """
    Extrai o texto de um arquivo PDF.
    
    Args:
        pdf_path (str): Caminho do arquivo PDF.
    
    Returns:
        str: Texto extraído do PDF.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def process_pdfs(base_directory, output_directory):
    """
    Processa todos os PDFs em um diretório, extrai texto e salva como arquivos `.txt`.

    Args:
        base_directory (str): Diretório contendo os arquivos PDF.
        output_directory (str): Diretório para salvar os arquivos de texto extraídos.

    Returns:
        list: Lista de caminhos dos arquivos de texto gerados.
    """
    os.makedirs(output_directory, exist_ok=True)

    pdf_files = [f for f in os.listdir(base_directory) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"Nenhum arquivo PDF encontrado em {base_directory}.")

    output_files = []
    for file_name in pdf_files:
        pdf_path = os.path.join(base_directory, file_name)
        text = pdf_to_text(pdf_path)
        output_file_path = os.path.join(output_directory, f"{os.path.splitext(file_name)[0]}.txt")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(text)
        output_files.append(output_file_path)
        print(f"Arquivo processado: {file_name}")

    return output_files


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


if __name__ == "__main__":
    # Exemplo de uso
    BASE_DIR = "./datasets/raw_data"
    OUTPUT_DIR = "./datasets/processed_texts"

    # Processar PDFs
    print("Processando PDFs...")
    processed_files = process_pdfs(BASE_DIR, OUTPUT_DIR)

    # Dividir em chunks
    for file_path in processed_files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text, chunk_size=500, overlap=50)

        # Salvar chunks
        chunk_output_path = file_path.replace(".txt", "_chunks.txt")
        save_chunks(chunks, chunk_output_path)
