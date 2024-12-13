import yaml
import os
import fitz  # PyMuPDF para extração de PDFs
import requests
from bs4 import BeautifulSoup
import logging
from utils import process_pdfs


def scrape_web_pages(config):
    print("Iniciando scraping de páginas da web...")
    urls = config["web_pages"]
    output_directory = config["storage"]["processed_texts_directory"]
    os.makedirs(output_directory, exist_ok=True)
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text(separator="\n", strip=True)

                file_name = url.split("/")[-1].split(".")[0] + ".txt"
                file_path = os.path.join(output_directory, file_name)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                logging.info(f"Conteúdo da página salvo: {file_name}")
            else:
                logging.warning(f"Erro ao acessar {url}: Status {response.status_code}")
        except Exception as e:
            logging.error(f"Erro ao processar {url}: {e}")

def process_pdfs_text(config):
    raw_df = config["storage"]["raw_pdf_directory"]
    output_directory = config["storage"]["processed_texts_directory"]
    os.makedirs(raw_df, exist_ok=True)
    print("Iniciando processamento de PDFs...")
    process_pdfs(raw_df,output_directory)
    print("Processamento do PDFs concluído.")
