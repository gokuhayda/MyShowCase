# Intelligent Repository Explorer

## Descrição

Este projeto é uma aplicação Python que utiliza LLMs (como OpenAI GPT e Llama) para análise e consulta em repositórios do GitHub. Foi projetado para facilitar a exploração de repositórios e melhorar a eficiência na recuperação de informações.

---

## Funcionalidades

- **Carregamento de Repositórios GitHub**: Lê repositórios e filtra arquivos por extensão.
- **Integração com LLMs**: Suporte para modelos OpenAI GPT e Meta-Llama.
- **Persistência de Índices**: Salva dados para uso futuro.
- **Configuração Personalizável**: Via arquivo `config.yaml`.

---

## Requisitos

- Python 3.9+
- Dependências listadas em `requirements.txt`.

---

## Configuração

1. Clone o repositório:
   ```bash
   git clone https://github.com/gokuhayda/eTechShowCase.git

Crie e ative um ambiente virtual:

python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

Instale as dependências:

pip install -r requirements/requirements.txt

Configure o arquivo .env com suas chaves de API:

GITHUB_TOKEN=seu_github_token

OPENAI_API_KEY=sua_openai_api_key

HUGGINGFACEHUB_API_TOKEN=sua_huggingfacehub_api_key

Como Usar

    Atualize o arquivo config.yaml para apontar para o repositório GitHub desejado.
    Execute o script principal:

python src/main.py



Estrutura de Arquivos

📦 Intelligent Repository Explorer

 ┣ 📂 src
 
 ┃ ┗ 📜 main.py
 
 ┣ 📂 config
 
 ┃ ┗ 📜 config.yaml
 
 ┣ 📂 docs
 
 ┃ ┗ 📜 README.md
 
 ┣ 📂 requirements
 
    ┗ 📜 requirements.txt


