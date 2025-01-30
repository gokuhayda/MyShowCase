# MLOps Consulting Framework

Este repositório contém a implementação de um framework escalável para consultoria e implantação de MLOps em ambientes corporativos.

## 📌 Visão Geral

O **MLOps Consulting Framework** foi desenvolvido para oferecer um fluxo estruturado de descoberta, análise, projeto e implementação de pipelines de MLOps em empresas de grande porte.

## 📂 Estrutura do Repositório

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

## 🚀 Como Usar

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

## ⚙️ Configuração

Os arquivos `agents.yaml`, `tasks.yaml` e `mlops_globals.yaml` permitem configurar as diferentes etapas do framework.

## 📝 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

📧 Autor: **Eric Gustavo Reis de Sena**  
✉️ Email: egrsena@gmail.com
