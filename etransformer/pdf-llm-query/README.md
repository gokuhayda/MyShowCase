
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

### 5. **Consultas com Subperguntas**

O sistema oferece suporte a consultas avançadas com **subperguntas**, permitindo explorar documentos em maior profundidade, particularmente úteis para perguntas complexas ou multi-facetadas.

**Como funciona:**
- A funcionalidade divide uma pergunta principal em subperguntas específicas, realizando consultas separadas para cada uma delas.
- As respostas às subperguntas são combinadas para formar uma resposta consolidada à pergunta principal.
- Esse método é particularmente eficaz ao lidar com grandes volumes de informações ou perguntas que envolvem várias partes.

**Exemplo com o PDF de Exemplo**
O arquivo `currículo_exemplo.pdf`, incluído na pasta `data`, contém informações sobre a trajetória profissional de um cientista de dados.

Se você perguntar: *"Quais projetos relacionados a Machine Learning foram realizados e qual o impacto de cada um?"*, o sistema pode dividir a pergunta em:
1. Quais projetos de Machine Learning estão listados no documento?
2. Qual foi o impacto relatado para cada projeto?

O sistema pode retornar:
- **Projeto:** Reconhecimento Biométrico Infantil - Natosafe.  
  **Impacto:** Aumento de 20% na precisão de identificação biométrica.
- **Projeto:** Sistema preditivo para rastreadores solares - FIT.  
  **Impacto:** Aumento de 10% na eficiência operacional.
- **Projeto:** Previsão de Demanda e Clusterização - MadeiraMadeira.  
  **Impacto:** Redução de 10% nos custos logísticos.

**Passo a Passo para Utilizar:**
1. Insira sua pergunta no campo **"Consulta com Subperguntas"**.
2. Clique no botão **"Consultar (Subperguntas)"**.
3. O sistema processará a pergunta, realizará as consultas, e exibirá a resposta consolidada.
4. Se houver fontes disponíveis, elas serão exibidas abaixo da resposta.

**Nota:** Certifique-se de que o índice vetorial foi criado antes de realizar consultas. Caso contrário, o sistema exibirá uma mensagem de erro.

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
