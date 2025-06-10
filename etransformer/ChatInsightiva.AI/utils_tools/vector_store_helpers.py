"""
Funções auxiliares para manipulação do vetor store (DeepLake) e leitura de arquivos.
"""

import os
import shutil
from pathlib import Path

def should_rebuild_index(config: dict) -> bool:
    """
    Verifica se o índice vetorial deve ser reconstruído com base na configuração.

    Parâmetros:
        config (dict): Configuração contendo a flag 'force_rebuild_vector_index'.

    Retorna:
        bool: True se o índice deve ser reconstruído, False caso contrário.
    """
    return config.get("force_rebuild_vector_index", False)

def clear_deeplake_vector_store(path: str):
    """
    Remove o diretório onde está armazenado o vetor store DeepLake, se existir.

    Parâmetros:
        path (str): Caminho do vetor store.

    Retorna:
        None
    """
    if os.path.exists(path):
        print(f"🧹 Limpando banco DeepLake em: {path}")
        shutil.rmtree(path)
    else:
        print(f"⚠️ Nenhum diretório encontrado para limpar em: {path}")

def is_valid_deeplake_dataset(path: str) -> bool:
    """
    Verifica se o caminho fornecido corresponde a um dataset DeepLake válido.

    Parâmetros:
        path (str): Caminho do diretório.

    Retorna:
        bool: True se válido, False se não.
    """
    return Path(path).is_dir() and (Path(path) / "meta.json").exists()

def ler_conteudo_dos_arquivos(nomes_arquivos, pasta_base="./storage/datasets/processed_texts"):
    """
    Lê o conteúdo de arquivos de texto localizados em uma pasta base.

    Parâmetros:
        nomes_arquivos (list): Lista de nomes de arquivos.
        pasta_base (str): Caminho da pasta onde os arquivos estão localizados.

    Retorna:
        str: Conteúdo concatenado de todos os arquivos encontrados, ou aviso se vazio.
    """
    conteudo_total = ""
    for nome in nomes_arquivos:
        caminho = Path(pasta_base) / nome
        if caminho.exists():
            with open(caminho, "r", encoding="utf-8") as f:
                conteudo = f.read()
                conteudo_total += f"\n---\n📄 {nome}:\n{conteudo.strip()}\n"
    return conteudo_total if conteudo_total else "⚠️ Nenhum conteúdo foi encontrado nos arquivos listados."
