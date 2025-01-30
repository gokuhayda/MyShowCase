# 🚀 MLOps Consulting Framework

Este repositório contém a implementação de um framework escalável para consultoria e implantação de MLOps em ambientes corporativos.

---

## 📌 **Visão Geral**

O **MLOps Consulting Framework** foi desenvolvido para oferecer um fluxo estruturado de **descoberta, análise, projeto e implementação de pipelines de MLOps** em empresas de grande porte.

Este projeto automatiza **consultoria MLOps** utilizando **CrewAI**, organizando tarefas para agentes especializados. O sistema guia empresas na implementação de **pipelines de machine learning escaláveis**, com **boas práticas de automação, monitoramento e deployment**.

### **Principais Funcionalidades:**
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

## 👨‍💻 **Agentes & Suas Responsabilidades**

| Agente | Função |
|--------|----------------|
| **Creative Coordinator Agent** | Gera o questionário inicial de avaliação |
| **Discovery Agent** | Processa as respostas do cliente |
| **Business Understanding Agent** | Define KPIs e ROI do projeto |
| **Data Engineer** | Mapeia fontes de dados e arquitetura ETL |
| **Pipeline Specialist** | Projeta o pipeline MLOps completo |
| **Data Scientist** | Seleciona os melhores modelos de ML |
| **ML Engineer** | Define estratégias de treino e validação |
| **DevOps Engineer** | Planeja deploy, CI/CD e automação |
| **Monitoring Specialist** | Implementa monitoramento e alerta de drift |
| **Report Generation Agent** | Consolida todas as análises no relatório final |

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

## 📝 **Licença**

Este projeto está licenciado sob a [MIT License](LICENSE).

---

📧 **Autor:** Eric Gustavo Reis de Sena  
✉️ **Email:** egrsena@gmail.com  
