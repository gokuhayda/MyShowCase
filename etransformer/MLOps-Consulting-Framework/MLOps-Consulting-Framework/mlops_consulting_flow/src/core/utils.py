# core/utils.py

import os
import yaml
from typing import Dict, Any
from bs4 import BeautifulSoup
import re
from markdown import markdown

def load_yaml_config(file_path: str) -> dict:
    """
    Carrega um arquivo YAML e retorna seu conteúdo como dicionário Python.
    Lança exceções específicas em caso de arquivo inexistente ou erro de parsing YAML.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            print(f"Configuração carregada com sucesso de {file_path}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Erro ao analisar o arquivo YAML {file_path}: {e}")

def load_file_content(file_path: str) -> str:
    """
    Lê todo o conteúdo de um arquivo de texto (UTF-8) e retorna como string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"Conteúdo do arquivo {file_path} carregado com sucesso.")
        return content

def parse_project_markdown(md_text: str) -> Dict[str, Any]:
    """
    Lê uma string em formato Markdown contendo informações do projeto 
    e retorna um dicionário com campos extraídos.
    """
    html_content = markdown(md_text)
    soup = BeautifulSoup(html_content, 'html.parser')

    parsed_data = {
        "project_type": "",
        "industry": "",
        "objectives": "",
        "team_members": "",
        "requirements": "",
        "dependencies": [],
        "deliverables": []
    }
    
    # Project Type
    project_type_elem = soup.find(lambda tag: tag.name in ["h1","h2","h3"] and "Project Type" in tag.text)
    if project_type_elem:
        next_node = project_type_elem.find_next_sibling()
        if next_node:
            parsed_data["project_type"] = next_node.get_text(strip=True)

    # Industry
    industry_elem = soup.find(lambda tag: tag.name in ["h1","h2","h3"] and "Industry" in tag.text)
    if industry_elem:
        next_node = industry_elem.find_next_sibling()
        if next_node:
            parsed_data["industry"] = next_node.get_text(strip=True)

    # Project Objectives
    objectives_section = soup.find(lambda tag: tag.name in ["h1","h2","h3"] and "Project Objectives" in tag.text)
    if objectives_section:
        objectives_text = []
        pointer = objectives_section.find_next_sibling()
        while pointer and pointer.name not in ["h1","h2","h3"]:
            objectives_text.append(pointer.get_text(strip=True))
            pointer = pointer.find_next_sibling()
        parsed_data["objectives"] = "\n".join(objectives_text)
    
    # Team Members
    team_section = soup.find(lambda tag: tag.name in ["h1","h2","h3"] and "Team Members" in tag.text)
    if team_section:
        pointer = team_section.find_next_sibling()
        current_role = None
        team_members_dict = {}

        while pointer and pointer.name not in ["h1","h2","h3"]:
            if pointer.name == "p" and pointer.find("strong"):
                # Captura o título do cargo (ex.: "Gerenciamento de Projeto")
                current_role = pointer.get_text(strip=True).replace(":", "")
                team_members_dict[current_role] = []
            elif pointer.name == "ul" and current_role:
                for li in pointer.find_all("li"):
                    team_members_dict[current_role].append(li.get_text(strip=True))
            pointer = pointer.find_next_sibling()

        # Converter para formato string legível (string única)
        team_members_list = [f"{role}: {', '.join(members)}" for role, members in team_members_dict.items()]
        parsed_data["team_members"] = "\n".join(team_members_list)

    # Project Requirements
    requirements_dict = {}

    req_section = soup.find(lambda tag: tag.name in ["h1","h2","h3"] and "Project Requirements" in tag.text)
    if req_section:
        pointer = req_section.find_next_sibling()
        req_key = None
        while pointer and pointer.name not in ["h1","h2","h3"]:
            if pointer.name == "p" and re.match(r"^\d+\.\s+\*\*", pointer.get_text(strip=True)):
                # e.g. "1. **Infraestrutura de Dados**"
                content = pointer.get_text(strip=True).replace("**", "")  # Remove negrito
                req_key = content
                requirements_dict[req_key] = []
            elif pointer.name == "ul" and req_key:
                for li in pointer.find_all("li"):
                    requirements_dict[req_key].append(li.get_text(strip=True))
            pointer = pointer.find_next_sibling()

    # Converter para string formatada
    parsed_data["requirements"] = "\n".join(
        [f"{key}:\n  - " + "\n  - ".join(values) for key, values in requirements_dict.items()]
    )

    # Dependencies
    deps_section = soup.find(lambda tag: tag.name in ["h1","h2","h3"] and "Dependencies" in tag.text)
    if deps_section:
        pointer = deps_section.find_next_sibling()
        while pointer and pointer.name not in ["h1","h2","h3"]:
            if pointer.name == "ul":
                for li in pointer.find_all("li"):
                    parsed_data["dependencies"].append(li.get_text(strip=True))
            pointer = pointer.find_next_sibling()

    # Expected Deliverables
    deliverables_section = soup.find(lambda tag: tag.name in ["h1","h2","h3"] and "Expected Deliverables" in tag.text)
    if deliverables_section:
        pointer = deliverables_section.find_next_sibling()
        current_deliverable = None
        while pointer and pointer.name not in ["h1","h2","h3"]:
            if pointer.name == "p" and re.match(r"^\d+\.\s+\*\*", pointer.get_text(strip=True)):
                current_deliverable = pointer.get_text(strip=True)
                parsed_data["deliverables"].append({current_deliverable: []})
            elif pointer.name == "ul" and current_deliverable:
                for li in pointer.find_all("li"):
                    parsed_data["deliverables"][-1][current_deliverable].append(li.get_text(strip=True))
            pointer = pointer.find_next_sibling()

    return parsed_data

def load_project_description(project_description_path: str) -> Dict[str, Any]:
    """Lê e processa a descrição do projeto a partir do Markdown."""
    if not os.path.exists(project_description_path):
        raise FileNotFoundError(f"Arquivo de descrição do projeto não encontrado: {project_description_path}")

    with open(project_description_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    return md_content, parse_project_markdown(md_content)
