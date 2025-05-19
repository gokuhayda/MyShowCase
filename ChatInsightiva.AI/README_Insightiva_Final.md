# ChatInsightiva.AI – Plataforma Inteligente de Respostas com Diagnóstico Organizacional

**ChatInsightiva.AI** é uma plataforma de inteligência artificial desenvolvida para oferecer respostas inteligentes e avaliadas automaticamente com base nos conteúdos institucionais da sua empresa. Com uma arquitetura robusta e modular, integra tecnologias de ponta em **IA generativa**, **busca vetorial**, **validação semântica**, **avaliação de qualidade** e **agentes especializados**.

---

## 🚀 Visão Geral do Projeto

- Plataforma de Perguntas e Respostas com RAG (Retrieval-Augmented Generation)
- Arquitetura baseada em agentes CrewAI para tarefas especializadas
- Validação de escopo e semântica com ferramentas próprias (guardrails)
- Avaliação automática de qualidade com RAGAS (faithfulness, relevância, precisão, recall)
- Interface visual amigável com Streamlit
- Infraestrutura pronta para integração com WhatsApp, Google Drive, SQL, e n8n

---

## 🧠 Objetivo Estratégico

Permitir que organizações respondam automaticamente a dúvidas internas e externas com base em seus próprios documentos, garantindo:

- Conformidade com conteúdo oficial
- Agilidade na comunicação
- Monitoramento da qualidade das respostas
- Escalabilidade com automações

---

## 📁 Estrutura do Projeto

```
├── .gitignore
├── FAQ_Exemplo_LIMPO.json
├── LICENSE
├── MANIFEST.in
├── README.md
├── __init__.py
├── api_server
    ├── ChatbotInsightiva.credentials.ts
    ├── ChatbotInsightiva.node.ts
    ├── ChatbotInsightiva.png
    ├── Chatbot_Insightiva_Fluxo_Exemplo.json
    ├── Dockerfile
    ├── README_ChatbotInsightiva.md
    ├── api.py
    ├── docker-compose.yaml
    ├── setup_n8n_chatbot.sh
├── app_modules
    ├── qa_extractor.py
    ├── qa_interface.py
    ├── qa_rewriter.py
├── app_streamlit_ragas.py
├── build_category_indexes.py
├── chat_history_manager.py
├── chatbot.js
├── chatbot.py
├── config
    ├── agents.yaml
    ├── index.yaml
    ├── parameters.yaml
    ├── tasks.yaml
├── core
    ├── __init__.py
    ├── crew_manager.py
    ├── graph_loader.py
    ├── load_tools.py
    ├── query_router.py
    ├── router.py
├── data_ingestion.py
├── docs_storage_setup.py
├── env.example
├── indexing.py
├── model_loader.py
├── pip install upgrade pip.txt
├── query_index.py
├── requirements.txt
├── sessions_manager.py
├── setup.py
├── storage
    ├── datasets
        ├── knowledge_base
            ├── FAQ_Exemplo_LIMPO.json
        ├── raw_data
            ├── .~lock.FAQs - 2.0.5 - Próximos Passos depois do Diagnóstico Insightiva.docx#
├── tests
    ├── deploy_test_chatbot_backend.py
    ├── teste_ragas_simulado.py
├── tools
    ├── __init__.py
    ├── anti_prompt_injection_tool.py
    ├── segmented_semantic_guard_tool.py
    ├── semantic_guard_tool.py
├── utils.py
├── utils_tools
    ├── __init__.py
    ├── config_loader.py
    ├── guard_validator.py
    ├── langfuse_logger.py
    ├── model_logger.py
    ├── model_logger_com_callback.py
    ├── qa_utils.py
    ├── vector_index_utils.py
    ├── vector_store_helpers.py
```

---

## ⚙️ Componentes e Arquitetura

### Núcleo Inteligente
- `chatbot.py`: pipeline de execução do modelo RAG + reranking + validação + fallback com agentes
- `query_router.py` e `router.py`: roteamento e orquestração das respostas com fallback inteligente
- `crew_manager.py`: configuração e gerenciamento dos agentes especialistas
- `qa_interface.py`: interface de perguntas e respostas com avaliação

### Módulos de Avaliação
- `app_streamlit_ragas.py`: dashboard interativo para avaliar a qualidade de respostas com RAGAS
- `qa_extractor.py`, `qa_rewriter.py`: extração, reescrita e avaliação de dados
- `teste_ragas_simulado.py`: simulações automatizadas para validação em lote

### Validação e Segurança
- `semantic_guard_tool.py`, `segmented_semantic_guard_tool.py`: garantem que só perguntas válidas recebam resposta
- `anti_prompt_injection_tool.py`: proteção contra manipulação por prompt injection

### Utilitários
- `model_loader.py`, `config_loader.py`, `langfuse_logger.py`, `vector_index_utils.py`: carregamento de modelos, logs, indexação e vetorização
- `data_ingestion.py`, `indexing.py`, `docs_storage_setup.py`: ingestão e armazenamento vetorial (DeepLake)

### API e Deploy
- `api_server/api.py`: disponibiliza a plataforma como serviço via API REST
- `setup.py`: instalação de dependências como pacote Python
- `n8n`, `Docker` e integrações externas possíveis

---

## 📊 Avaliação com RAGAS

A plataforma integra métricas automáticas de avaliação de respostas:

- **Faithfulness** – A resposta é fiel ao conteúdo?
- **Answer Relevancy** – A resposta é útil e pertinente?
- **Context Precision & Recall** – Quão bem o sistema usou o conteúdo vetorizado?

Esses dados podem ser usados para treinar novos modelos, detectar inconsistências e refinar continuamente a experiência do usuário.

---

## ✅ Diferenciais Estratégicos

- Garantia de respostas **sem alucinação**, com base em documentos da empresa
- Estrutura extensível para múltiplas áreas temáticas (ex: onboarding, clima organizacional, liderança)
- Transparência com logs via **Langfuse**
- Avaliação integrada para **medir ROI de implantação da IA**

---

## 📄 Licenciamento e Contato

Este projeto é licenciado sob modelo privado e pode ser customizado para empresas que desejam:

- Implementar IA institucional
- Treinar modelos com seus próprios documentos
- Integrar workflows com plataformas já utilizadas

**Contato:** [NextGen Analytics Solutions](https://gokuhayda.github.io/nextgen_frontend)

---
---

## 🎓 Contexto de Desenvolvimento

Este projeto foi desenvolvido como parte de um portfólio pessoal com fins educacionais e de experimentação técnica. A proposta é explorar e aplicar, de forma prática, tecnologias de ponta em inteligência artificial generativa e sistemas de perguntas e respostas (RAG).

A aplicação **ChatInsightiva.AI** simula um ambiente real de atendimento automatizado por chatbot, voltado para resposta de FAQs institucionais com apoio em documentos internos. Foi uma oportunidade de:

- Aprofundar conhecimentos em **RAG**, **CrewAI**, **RAGAS** e **guardrails semânticos**
- Aprimorar a construção de pipelines robustos para **validação de escopo e conteúdo**
- Experimentar arquiteturas modernas para **sistemas de resposta automáticos baseados em documentos**
- Integrar serviços com ferramentas como **Streamlit**, **n8n**, **Docker** e **Langfuse**

> ⚠️ **Nota:** Este não é um projeto comercial nem vinculado a nenhuma organização real. A empresa “Insightiva” é fictícia, criada unicamente para fins ilustrativos.

Este projeto está publicado dentro do repositório de portfólio [MyShowCase](https://github.com/gokuhayda/MyShowCase), na pasta [`etransformer/Insightiva.AI`](https://github.com/gokuhayda/MyShowCase/tree/main/etransformer/Insightiva.AI).
