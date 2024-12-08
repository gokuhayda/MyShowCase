<<<<<<< HEAD

# Sistema de Consulta a PDFs com GPT-4

Este projeto permite realizar consultas em arquivos PDF utilizando a tecnologia GPT-4 e técnicas de reranking para melhorar a precisão das respostas.

## Funcionalidades

- Conversão de PDFs em texto.
- Criação de um índice vetorial para buscas rápidas.
- Consulta ao conteúdo dos documentos PDF com suporte a rerankers avançados.

---

## Pré-requisitos

- **Python 3.8 ou superior.**
- **Chave de API da OpenAI** (para acesso ao GPT-4).

---

## Instalação

1. **Clone o repositório**:
    ```bash
    git clone https://github.com/seu-usuario/pdf-gpt4-query.git
    cd pdf-gpt4-query
    ```

2. **Crie e ative um ambiente virtual**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3. **Instale as dependências do projeto**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Configuração do PYTHONPATH**:
    Configure o `PYTHONPATH` para que o Python reconheça os módulos do projeto.

    - No terminal:
      ```bash
      export PYTHONPATH=$(pwd):$PYTHONPATH
      ```

    - Para tornar a configuração permanente, edite o arquivo do shell (`~/.bashrc`, `~/.zshrc` ou equivalente):
      ```bash
      echo 'export PYTHONPATH=/caminho/para/pdf-gpt4-query:$PYTHONPATH' >> ~/.bashrc
      source ~/.bashrc
      ```

---

## Uso

### 1. **Executar o aplicativo Streamlit**
Inicie o sistema:
```bash
streamlit run app/main.py
```

### 2. **Configurar a chave de API da OpenAI**
No painel lateral do Streamlit:
- Insira sua chave de API da OpenAI.
- Pressione Enter para validar.

### 3. **Testar com o PDF de Exemplo**
Para facilitar os testes, um PDF de exemplo está incluído no diretório `data`. Siga os passos:
- Clique no botão **"Usar PDF de Exemplo"**.
- O sistema processará o PDF e criará o índice vetorial.

### 4. **Realizar consultas**
- Insira perguntas no campo de texto principal.
- O sistema retornará respostas baseadas no conteúdo do PDF.

---

## Teste com o PDF de Exemplo

Certifique-se de que o arquivo `example.pdf` está presente no diretório `data`. Caso ele tenha sido movido ou deletado, restaure-o ou adicione outro arquivo PDF ao mesmo local.

---

## Estrutura do Projeto

```plaintext
pdf-gpt4-query/
│
├── app/
│   ├── main.py                  # Código principal do aplicativo Streamlit
│   ├── utils.py                 # Funções auxiliares
│   ├── indexing.py              # Funções de criação de índices vetoriais
│
├── data/                        # Pasta para arquivos de entrada (inclui example.pdf)
│   └── example.pdf              # PDF de exemplo para testes
│
├── pdf-gpt4-query/              # Saída de processamento
│   ├── processed_texts/         # Arquivos de texto extraídos dos PDFs
│   └── deep_lake_db/            # Banco de dados vetorial criado
│
├── tests/                       # Testes automatizados
├── requirements.txt             # Dependências do projeto
├── README.md                    # Documentação do projeto
├── .gitignore                   # Arquivos ignorados pelo Git
└── LICENSE                      # Licença do projeto
```

---

## Problemas Comuns e Soluções

### Nenhum PDF encontrado:
- Certifique-se de que a pasta selecionada contém arquivos `.pdf`.
- Use o botão **"Usar PDF de Exemplo"** para validar a funcionalidade.

### API não configurada:
- Insira sua chave da OpenAI na barra lateral.

### Índice não criado:
- Verifique se os PDFs foram processados corretamente na pasta `processed_texts`.

---

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir um pull request ou relatar problemas na aba de **Issues**.

---

## Licença

Este projeto está licenciado sob a MIT License.
# Nova Linha
=======
# eTechShowCase

Welcome to **eTechShowCase**, a public repository showcasing my technical experiments, projects, and learning journey across various fields of technology.

## 📌 About the Project
This repository aims to:
- Share knowledge and insights from personal coding explorations.
- Demonstrate technical skills in programming, data analysis, and machine learning.
- Foster collaboration and learning within the developer community.

Feel free to explore, use, and contribute to the projects available here!

## 📂 Repository Structure
The repository is organized into categories based on programming languages and themes:

eTechShowCase/
├── etsml/

## 📖 How to Use
1. Browse through the repository to find a project or code snippet of interest.
2. Clone the repository using:
   ```bash
   git clone https://github.com/gokuhayda/eTechShowCase.git

3. Explore the code, modify it, and adapt it to your own projects.

📋 License

This repository is licensed under the MIT License. You are free to use the code, provided proper attribution is given.

🤝 Contributions

Contributions, suggestions, and feedback are highly encouraged! If you'd like to collaborate, feel free to open an issue or submit a pull request.

📫 Connect with Me

Find me on LinkedIn (https://www.linkedin.com/in/éric-sena) or GitHub to discuss tech, projects, or collaboration opportunities.
>>>>>>> 6e630e0815839ff75bb0e312906729002d7a7433
