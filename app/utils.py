import os
import fitz
from tkinter import Tk, filedialog

def select_folder():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Selecione a pasta com PDFs")
    root.destroy()
    return folder_path

# Função para converter PDFs em texto
def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Função para processar PDFs e salvar os textos
def process_pdfs(base_directory):
    output_dir = os.path.join(base_directory, "processed_texts")
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(base_directory) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"Nenhum arquivo PDF encontrado em {base_directory}.")

    for file_name in pdf_files:
        pdf_path = os.path.join(base_directory, file_name)
        text = pdf_to_text(pdf_path)
        text_file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.txt")
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Arquivo processado: {file_name}")
    
    print(f"Todos os arquivos PDF foram processados e salvos em {output_dir}.")
    return output_dir




