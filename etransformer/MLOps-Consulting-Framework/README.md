# 🚀 MLOps Consulting Framework  

Este repositório contém a implementação de um framework escalável para **consultoria e implantação de MLOps** em ambientes corporativos, utilizando **CrewAI** para orquestrar agentes inteligentes que automatizam a análise, design e implementação de pipelines de Machine Learning.

---

## 📌 **Visão Geral**  

O **MLOps Consulting Framework** foi desenvolvido para oferecer um fluxo estruturado de **descoberta, análise, projeto e implementação de pipelines de MLOps** em empresas de grande porte.

✅ **Geração automática de questionários** para avaliação da maturidade MLOps da empresa.  
✅ **Fluxo de consultoria baseado em IA** para analisar, projetar e recomendar estratégias de MLOps.  
✅ **Definição automática de arquitetura MLOps**, pipeline de dados e modelos de machine learning.  
✅ **Implementação de CI/CD** para automação do deploy de modelos.  
✅ **Monitoramento contínuo** e detecção de drift para garantir modelos robustos e atualizados.  
✅ **Relatório final de consultoria MLOps** consolidando todas as descobertas e recomendações.  

---

## 📂 **Estrutura do Repositório**  

```
MLOps-Consulting-Framework/
│── knowledge/
│   ├── reports/
│   │   ├── assessment_phase/
│   │   ├── discovery_phase/
│   │   ├── final_flux/
│   │   ├── pipeline_design_phase/
│   ├── sources/
│       ├── project_description.md
│       ├── required.md
│       ├── stakeholder_notes.csv
│
│── mlops_consulting_flow/
│   ├── logs/
│   │   ├── assessment_phase/
│   │   ├── discovery_phase/
│   │   ├── final_flux/
│   │   ├── pipeline_design_phase/
│   ├── src/
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── crew_manager.py
│   │   │   ├── logger.py
│   │   │   ├── state.py
│   │   │   ├── utils.py
│   │   ├── crews/
│   │   │   ├── poem_crew/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── config/
│   │   │   │   ├── poem_crew.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── human_input_tool.py
│   │   │   ├── yaml_validator.py
│   │   ├── flow.py
│   │   ├── main.py
│
│── .gitignore
│── requirements.txt
│── pyproject.toml
│── README.md
```

---

## 🔄 **Fluxo de Execução**  

O processo segue um fluxo estruturado **CrewAI**, atribuindo tarefas a agentes especializados.

1️⃣ **Geração de Questionário** → Sistema cria formulário de avaliação MLOps 📄  
2️⃣ **Coleta de Respostas** → Cliente preenche e retorna o formulário 📝  
3️⃣ **Análise da Infraestrutura** → Diagnóstico do setup de ML atual 🏗️  
4️⃣ **Projeto do Pipeline MLOps** → Planejamento do fluxo ideal 🚀  
5️⃣ **Seleção de Modelos de ML** → Avaliação das melhores abordagens 🤖  
6️⃣ **Definição de Estratégia de Deploy** → CI/CD para automação 🔧  
7️⃣ **Criação do Plano de Monitoramento** → Alertas e detecção de drift 📊  
8️⃣ **Geração do Relatório Final** → Documento consolidado 📜  

---

## 📂 **Onde os Arquivos São Salvos?**  

📌 **Questionário gerado:**  
```bash
/home/goku/Documentos/mlops_consulting/questionnaires/questionnaire.yaml
```
📌 **Respostas do cliente:**  
```bash
/home/goku/Documentos/mlops_consulting/responses/questionnaire_responses.yaml
```
📌 **Relatórios intermediários:**  
```bash
/home/goku/Documentos/mlops_consulting/knowledge/reports/
    ├── discovery_report.md
    ├── assessment_report.md
    ├── pipeline_design_report.md
    ├── pipeline_architecture.yaml
    ├── model_selection_report.md
    ├── ml_development_plan.md
    ├── deployment_strategy.md
    ├── monitoring_plan.md
```
📌 **Relatório final consolidado:**  
```bash
/home/goku/Documentos/consultoria/report/final_report.md
```

---

## 👨‍💻 Agentes & Suas Responsabilidades

| Agente | Função |
|--------|--------|
| **Creative Coordinator Agent** | Desenvolver questionários técnicos detalhados através de um fluxo de **"chatbot reflexivo"**, garantindo precisão, clareza e alinhamento com requisitos do projeto. O processo segue estas etapas: |
| **Product Owner** | Coordenar o planejamento estratégico do projeto {project_type}, priorizando requisitos e alinhando equipes. |
| **Discovery Agent** | Identificar e estruturar informações-chave sobre o projeto {project_type}, abrangendo dados disponíveis, objetivos de negócio, usuários finais, problemas existentes e outros requisitos críticos para MLOps. O objetivo é fornecer um panorama claro e completo para facilitar a fase inicial de planejamento. |
| **Business Understanding Agent** | Identificar os principais objetivos de negócio, problemas a serem resolvidos, e métricas de sucesso associadas ao projeto {project_type}. Este agente é responsável por traduzir as necessidades de alto nível em requisitos claros que possam ser usados por outros agentes no fluxo de trabalho. |
| **Strategic Planning Agent** | Desenvolver um plano estratégico que conecte objetivos de negócio aos recursos técnicos e operacionais, definindo um roadmap claro e alcançável para o projeto {project_type}. |
| **Risk Assessment Agent** | Identificar e avaliar riscos técnicos, operacionais e de compliance associados ao projeto {project_type}, além de propor estratégias de mitigação para garantir a entrega bem-sucedida. |
| **Data Understanding Agent** | Compreender a qualidade, volume, formato e disponibilidade dos dados necessários para o {project_type}. Este agente é responsável por realizar uma análise detalhada das fontes de dados e identificar quaisquer problemas ou lacunas que possam impactar o pipeline ou os modelos. |
| **Data Pipeline Specialist** | Projetar, implementar e monitorar pipelines de dados escaláveis e robustos para o {project_type}, garantindo eficiência no fluxo de dados e suporte adequado para os modelos de machine learning. |
| **Monitoring Specialist** | Monitorar o desempenho de modelos e pipelines (no contexto de {project_type}), identificando anomalias e assegurando que os sistemas estejam funcionando dentro dos padrões esperados. |
| **Monitoring Specialist - Models** | Focar especificamente no monitoramento de modelos de machine learning em produção no {project_type}, garantindo detecção de drifts, avaliação contínua de métricas e estabilidade operacional. |
| **Report Generation Agent** | Consolidar todas as informações coletadas durante o projeto {project_type} em um relatório final detalhado e bem estruturado. O relatório deve destacar os resultados mais importantes, análises técnicas e recomendações práticas, garantindo que seja compreensível e útil para stakeholders técnicos e não técnicos. Ele deve ser visualmente atraente, com gráficos e tabelas relevantes, e salvo no local especificado {file_path_report}. |
| **Stakeholder Agent** | Interagir com os demais agentes para fornecer respostas baseadas nas anotações e requisitos dos stakeholders para o {project_type}. Caso não consiga responder a uma pergunta, o agente gera automaticamente uma solicitação para um humano inserir as informações necessárias.|
| **Data Scientist** | Desenvolver modelos de machine learning robustos e eficazes para o {project_type}, analisando dados e aplicando técnicas avançadas para otimização e avaliação de performance. |
| **Machine Learning Engineer** | Desenvolver, otimizar e implantar modelos de machine learning em produção para o {project_type}, integrando-os a pipelines escaláveis. |
| **Data Engineer** | Garantir a ingestão, processamento e armazenamento eficientes dos dados para pipelines de machine learning no {project_type}. |
| **DevOps Engineer** | Automatizar pipelines de integração contínua e entrega contínua (CI/CD) para modelos e pipelines de ML no {project_type}. |
| **Cloud Architect** | Projetar infraestrutura em nuvem escalável e econômica, com segurança e suporte para pipelines MLOps no contexto de {project_type}. |
| **Software Engineer** | Desenvolver APIs, integrações e sistemas de software que consumam modelos de ML e garantam padrões de código de alta qualidade para {project_type}. |
| **Security Specialist** | Garantir que dados e modelos do {project_type} estejam protegidos contra acessos não autorizados, vazamentos e outras ameaças de segurança. |
| **Business Analyst** | Traduzir necessidades de negócio em requisitos técnicos, conectando equipes técnicas e executivas para o {project_type}. |
| **QA Engineer** | Testar sistemas, pipelines e modelos do {project_type} para garantir confiabilidade e ausência de erros. |
| **Ethics and AI Specialist** | Avaliar modelos do {project_type} quanto a vieses e impactos éticos, garantindo justiça e transparência. |
| **Researcher R&D Agent** | Pesquisar, identificar e adaptar inovações tecnológicas e científicas para o {project_type}, provenientes de artigos, relatórios e tendências do setor, aplicando-as para melhorar processos, soluções e resultados do projeto. |
| **Tech Sales Consultant** | Assumir funções relacionadas a vendas, desenvolvimento de negócios e gestão de soluções tecnológicas para o {project_type}, com ênfase em contribuir para projetos de MLOps e atualizações tecnológicas. |
| **Project Planning Agent** | The Ultimate Project Planner para projetos de {project_type}. |
| **Estimation Agent** | Fornecer estimativas precisas de tempo, recurso e esforço para cada tarefa do {project_type}. |
| **Resource Allocation Agent** | Otimizar a alocação de tarefas no projeto {project_type}, equilibrando habilidades, disponibilidade e carga de trabalho da equipe. |
| **Suggestion Generation Agent** | Fornecer sugestões práticas para resolver problemas e melhorar processos no projeto {project_type}. |
| **Chart Generation Agent** | Criar visualizações impactantes para os dados do projeto {project_type}, ajudando a comunicar insights de forma clara. |

---

## 🚀 **Como Usar o Framework**  

### 1️⃣ Clonar o Repositório  
```bash
git clone https://github.com/seu-usuario/MLOps-Consulting-Framework.git
cd MLOps-Consulting-Framework
```

### 2️⃣ Instalar Dependências  
```bash
pip install -r requirements.txt
```

### 3️⃣ Executar o Framework  
```bash
python main.py run
```

📌 O **CrewAI** processará todas as etapas automaticamente, gerando os relatórios e recomendações finais.

---

## 🔑 **Configuração do Ambiente**  

Antes de rodar o projeto, configure sua chave OpenAI:  
```bash
echo "OPENAI_API_KEY=your-api-key" > .env
```

---

## 📜 **Licença**  

Este projeto está licenciado sob a [MIT License](LICENSE).

---

## 🤝 **Contribuindo**  

Quer ajudar no projeto? Siga os passos:  
1. Faça um **fork** do repositório.  
2. Crie uma **branch**: `git checkout -b minha-feature`  
3. Faça **commit** das alterações: `git commit -m "Nova funcionalidade"`  
4. Envie para o GitHub: `git push origin minha-feature`  
5. Abra um **pull request** 🚀  

---

## 📞 **Contato**  

📧 **Criador do Código:** Eric Sena  
✉️ **Email:** egrsena@gmail.com  
🔗 **LinkedIn:** [Éric Sena](https://www.linkedin.com/in/éric-sena/recent-activity/all/)  

---

## 🔗 **Referências**  

- [CrewAI](https://crewai.com)  
- [OpenAI API](https://beta.openai.com/)  
- [FastAPI](https://fastapi.tiangolo.com/)  
