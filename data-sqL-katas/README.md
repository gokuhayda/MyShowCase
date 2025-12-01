# 🎯 SQL Katas for Senior Data Scientists

Um guia completo e prático de SQL Avançado para Cientistas de Dados que querem dominar análises complexas e se preparar para entrevistas técnicas de alto nível.

---

## 📂 Estrutura do Projeto

```
sql-katas/
├── README.md
├── .gitignore
├── setup/
│   ├── docker-compose.yml
│   ├── init_database.sql
│   └── README_SETUP.md
├── fundamentals/
│   ├── 01_window_functions/
│   │   ├── README.md
│   │   ├── examples.sql
│   │   ├── exercises.sql
│   │   └── solutions.sql
│   ├── 02_ctes/
│   │   ├── README.md
│   │   ├── examples.sql
│   │   ├── exercises.sql
│   │   └── solutions.sql
│   └── 03_self_joins/
│       ├── README.md
│       ├── examples.sql
│       ├── exercises.sql
│       └── solutions.sql
├── real_world_problems/
│   ├── 01_ecommerce/
│   │   ├── README.md
│   │   ├── schema.sql
│   │   ├── problems.md
│   │   └── solutions.sql
│   ├── 02_saas_metrics/
│   │   ├── README.md
│   │   ├── schema.sql
│   │   ├── problems.md
│   │   └── solutions.sql
│   └── 03_financial_analysis/
│       ├── README.md
│       ├── schema.sql
│       ├── problems.md
│       └── solutions.sql
├── interview_prep/
│   ├── thoughtworks_style/
│   │   ├── README.md
│   │   ├── problem_1_top_n.sql
│   │   ├── problem_2_cohort.sql
│   │   └── problem_3_sequential.sql
│   └── common_patterns/
│       ├── running_totals.sql
│       ├── gaps_and_islands.sql
│       └── hierarchical_queries.sql
├── cheatsheets/
│   ├── window_functions_cheatsheet.md
│   ├── cte_patterns.md
│   └── join_types_visual.md
├── datasets/
│   ├── ecommerce_sample.csv
│   ├── saas_events.csv
│   └── README.md
└── tests/
    ├── test_setup.sql
    └── README.md
```

---

## 🎓 Para Quem é Este Repositório?

- Já sabe SQL básico  
- Quer aprender Window Functions, CTEs, Self-Joins  
- Vai fazer entrevistas na ThoughtWorks, Nubank, Google, Meta, Uber  
- Quer escrever queries elegantes e eficientes  

---

## 🚀 Por Que SQL Avançado Importa?

SQL bem usado resolve problemas que Pandas não escala, economiza recursos e traz performance de nível profissional.

---

## 📚 Seções do Repositório

### 1. Fundamentals  
Conceitos essenciais com teoria, analogias, exemplos e exercícios.

### 2. Real World Problems  
Problemas reais: e-commerce, SaaS e finanças.

### 3. Interview Prep  
Desafios no estilo de entrevistas da ThoughtWorks, Google e Meta.

### 4. Cheatsheets  
Referências rápidas para estudo e revisão.

---

## 🛠️ Setup Rápido

### Docker
```bash
cd setup
docker-compose up -d
```

### Local (PostgreSQL)
```bash
psql -U postgres -f setup/init_database.sql
```

---

## 🎯 Roadmap de Estudos

- Semana 1: Window Functions, CTEs, Self-Joins  
- Semana 2: Problemas Reais de negócio  
- Semana 3: Entrevistas + otimização  

---

## 🤝 Contribuindo

1. Fork  
2. Nova branch  
3. Pull request  

---

## 📝 Licença

MIT License.

---

**SQL bem escrito é arte. Este repositório é seu ateliê.**
