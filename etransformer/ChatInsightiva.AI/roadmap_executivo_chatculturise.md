
# 🚀 Roadmap Executivo – Projeto ChatInsightiva.AI

---

## ✅ MVP 1 – Núcleo Funcional

| Etapa                            | Descrição                                                                 | Status       | Entrega Técnica                                      |
|----------------------------------|---------------------------------------------------------------------------|--------------|------------------------------------------------------|
| 🧭 Planejamento e Kick-off       | Alinhamento de escopo, definição de temas, estrutura modular              | ✅ Concluído | Documento de escopo, wireframes                     |
| 🏗️ Montagem da Arquitetura       | Estrutura com RAG, CrewAI, YAMLs, fallback, vetorização com DeepLake      | ✅ Concluído | `chatbot.py`, `query_router.py`, indexações         |
| 🧠 Agentes e Orquestração        | Concierge inteligente + especialistas com roteamento e fallback           | ✅ Concluído | `crew_manager.py`, `agents.yaml`, fallback manager  |
| 🛡️ Guardrails Básicos            | Validação semântica com `semantic_guard_tool` e segmentação temática      | ✅ Concluído | `semantic_guard_tool.py`, `segmented_guard_tool.py` |
| 📚 Documentação Técnica          | `README.md`, imagens explicativas, roadmap gráfico                        | ✅ Concluído | Documentação + slides executivos                    |
| 📦 Empacotamento da MVP 1        | Estrutura final de pastas e revisão para deploy ou entrega zipada         | ⚙️ Em andamento | `.zip`, reestruturação do projeto                   |
| ☁️ Ambiente de Testes Inicial    | Setup inicial em Colab-like (sem GPU) com 4vCPU, 16GB RAM, ngrok          | ⚠️ A iniciar | Acesso remoto, teste de performance básico          |

---

## 🔄 MVP 2 – Interface, APIs e Expansão

| Etapa                            | Descrição                                                                 | Status         | Entrega Esperada                                    |
|----------------------------------|---------------------------------------------------------------------------|----------------|-----------------------------------------------------|
| 🖥️ Interface de Avaliação        | Dashboard Streamlit para avaliação com RAGAS                              | ⚠️ Estruturação | `app_streamlit_ragas.py`                            |
| 🔌 Integração com API externa    | Backend REST via FastAPI, pronto para integração com frontend e n8n       | ⚠️ Estruturação | `api_server/api.py`                                 |
| 🧰 Guardrails Avançados          | Filtros por tema, validações refinadas e personalização                   | ⚠️ Em análise    | `guard_validator.py` refinado                       |
| ⚛️ Interface em React            | Frontend moderno, personalizado, conectado ao backend do chatbot          | ⚠️ Planejado    | Página web do chatbot integrada à API               |
| 🔗 Integrações n8n e externos    | Conectores com Google Drive, Sheets, SQL e WhatsApp                       | ⚠️ Planejado    | Webhooks + configuração de automações via n8n       |
| 📊 Avaliação com RAGAS           | Métricas: relevância, fidelidade, precisão, recall                        | ⚠️ Estruturação | Integração completa com pipeline e histórico        |
| 🧾 Armazenamento de Sessões e Logs | Registro de interações para auditoria e análise histórica                 | ⚠️ Planejado    | Google Sheets ou banco relacional (PostgreSQL etc.) |
| ☁️ Testes em Cloud               | Escalabilidade em cloud se necessário                                     | ⚠️ Estudo inicial| Alternativas a Colab, ex: GCP, AWS ou DigitalOcean  |

---

## 📌 Resumo Estratégico

- **MVP 1 entrega**: arquitetura modular robusta, fallback com RAG, orquestração por agentes e validação semântica.
- **MVP 2 expande**: APIs externas, dashboard de avaliação, frontend React e automações via n8n.
- Estrutura escalável e pronta para ambientes corporativos.
- Deploy inicial com fallback via **ngrok**, performance monitorada.
- Logs e sessões poderão ser salvos em banco ou Google Drive para análise futura.

---

## ✅ Pronto para escalar

Seja para integração com sistemas legados, visualização em tempo real ou controle de qualidade por métricas, o ChatInsightiva.AI está preparado para crescer junto com sua organização.
